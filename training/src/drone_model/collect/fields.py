"""
Block reader training frames from inside task arenas, the way the experts see them, from views at random distances,
heights, and angles.
harvest_crops looks at its field's plots and keeps frames showing wheat, into data/fields.
follow_mob and hunt_mobs look around their walled floor and keep frames showing a mob, plus a share without one, so the
gray concrete wall is seen as a wall, into data/arenas.
Labeled and sharded as collect.studio does. The plan is fixed by the seed, so a stopped run picks up at the first unfinished shard

    python -m drone_model.collect.fields --arenas 150
    python -m drone_model.collect.fields --tasks follow_mob,hunt_mobs --arenas 150
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from quadcraft import BridgeError, DroneEnv

from ..paths import DATA
from ..perception.reader import (
    CLASS_HASH,
    FRAME,
    INDEX,
    MOB_KINDS,
    entity_table,
    state_tables,
)
from .studio import FIELDS, MIN_SUBJECT_PIXELS, TERRAIN_SHARE, TERRAINS, label

ARENAS_PER_SHARD = 20
VIEWS = 24
CROP = "minecraft:wheat"
MOBS = np.array([INDEX[m] for m in MOB_KINDS])
# frames of a mob arena kept without any mob in them
MOBLESS_SHARE = 0.3


def views(region: list[float], top: float, bounds: list[float], rng: np.random.Generator) -> list[list[float]]:
    """
    Camera poses as [x, y, z, yaw, pitch], each looking at a random spot of region's floor at height top, from up to 8
    blocks past its edge and 1 to 7 over it, inside bounds, the geofence, since a view outside it ends the episode
    """
    x0, z0, x1, z1 = region
    cx, cz = (x0 + x1 + 1) / 2, (z0 + z1 + 1) / 2
    half = max(x1 - x0, z1 - z0) / 2 + 1
    out = []
    for _ in range(VIEWS):
        target = (rng.uniform(x0, x1 + 1), top, rng.uniform(z0, z1 + 1))
        angle = rng.uniform(0, 2 * math.pi)
        reach = rng.uniform(0, half + 8)
        eye = (
            float(np.clip(cx + math.cos(angle) * reach, bounds[0] + 1, bounds[3] - 1)),
            min(top + rng.uniform(1.0, 7.0), bounds[4] - 1),
            float(np.clip(cz + math.sin(angle) * reach, bounds[2] + 1, bounds[5] - 1)),
        )
        dx, dy, dz = target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]
        yaw = math.degrees(math.atan2(-dx, dz))
        pitch = -math.degrees(math.atan2(dy, max(math.hypot(dx, dz), 1e-3)))
        out.append([*eye, yaw, pitch])
    return out


def keep(task: str, cls: np.ndarray, rng: np.random.Generator) -> str | None:
    """The frame's subject when it is worth keeping, None to drop it"""
    if task == "harvest_crops":
        return CROP if (cls == INDEX[CROP]).sum() >= MIN_SUBJECT_PIXELS else None
    if np.isin(cls, MOBS).sum() >= MIN_SUBJECT_PIXELS:
        return "mobs"
    return "wall" if rng.random() < MOBLESS_SHARE else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="harvest_crops", help="harvest_crops, or follow_mob and hunt_mobs separated by commas")
    parser.add_argument("--arenas", type=int, default=150)
    parser.add_argument("--seed", type=int, default=380_000, help="first arena seed, clear of studio, demo, and evaluation seeds")
    parser.add_argument("--out", type=Path, default=None, help="data/fields for harvest_crops, data/arenas otherwise")
    args = parser.parse_args()
    tasks = args.tasks.split(",")
    out = args.out or DATA / ("fields" if tasks == ["harvest_crops"] else "arenas")

    rng = np.random.default_rng(args.seed)
    arenas = [(args.seed + i, tasks[i % len(tasks)], TERRAINS[rng.choice(len(TERRAINS), p=TERRAIN_SHARE)]) for i in range(args.arenas)]
    n_shards = (len(arenas) + ARENAS_PER_SHARD - 1) // ARENAS_PER_SHARD
    print(f"{len(arenas)} arenas in {n_shards} shards", flush=True)
    env = DroneEnv(task=tasks[0], tools=False, streams=("rgb", "depth", "mask", "state"), action_pause_ms=0, width=FRAME[1], height=FRAME[0])
    client = env.client
    classes, ripe = state_tables(client.state_names)
    entities = entity_table(client.mask_ids["entities"])
    base = client.mask_ids["entityBase"]
    failed = []
    try:
        for k in range(n_shards):
            shard = out / f"shard-{k:04d}"
            if (shard / "done").exists():
                continue
            parts: dict[str, list] = {f: [] for f in FIELDS}
            subjects = []
            for seed, task, terrain in arenas[k * ARENAS_PER_SHARD : (k + 1) * ARENAS_PER_SHARD]:
                try:
                    obs = client.reset(seed=seed, task=task, terrain=terrain, size=5)
                except BridgeError as e:
                    failed.append({"seed": seed, "task": task, "error": str(e)})
                    continue
                bounds = obs.state["bounds"]
                if task == "harvest_crops":
                    x0, _, z0, x1, y1, z1 = client.arena()["job"]["region"]
                    region, top = [x0, z0, x1, z1], y1 + 0.4
                else:
                    region, top = [bounds[0] + 1, bounds[2] + 1, bounds[3] - 2, bounds[5] - 2], bounds[1] + 1.5
                pick = np.random.default_rng(seed)
                for pose in views(region, top, bounds, pick):
                    try:
                        frame = client.view(pose)
                    except BridgeError:
                        # a hunt's mobs can end the episode by wrecking the drone
                        break
                    cls, rp = label(frame, classes, ripe, entities, base)
                    subject = keep(task, cls, pick)
                    if subject is None:
                        continue
                    parts["rgb"].append(frame.rgb)
                    parts["depth"].append(frame.depth.astype(np.float16))
                    parts["cls"].append(cls)
                    parts["ripe"].append(rp)
                    subjects.append(subject)
            if not subjects:
                continue
            shard.mkdir(parents=True, exist_ok=True)
            for f, v in parts.items():
                np.save(shard / f"{f}.npy", np.stack(v))
            (shard / "meta.json").write_text(json.dumps({"classes": CLASS_HASH, "frame": FRAME, "source": "game", "subjects": subjects}), encoding="utf-8")
            (shard / "done").touch()
            print(f"shard {k + 1}/{n_shards}: {len(subjects)} frames, {len(failed)} arenas failed so far", flush=True)
    finally:
        env.close()
        if failed:
            out.mkdir(parents=True, exist_ok=True)
            (out / "failed.json").write_text(json.dumps(failed, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
