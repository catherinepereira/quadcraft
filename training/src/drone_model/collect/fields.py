"""
Block reader training frames of crop fields, the way the harvest expert sees one.
Each harvest arena's field is seen from views around and over it at random distances, heights, and angles.
Labeled and sharded as collect.studio does, into data/fields/<shard>/.
The plan is fixed by the seed, so a stopped run picks up at the first unfinished shard

    python -m drone_model.collect.fields --arenas 150
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from quadcraft import BridgeError, DroneEnv

from ..paths import DATA
from ..perception.reader import CLASS_HASH, FRAME, INDEX, entity_table, state_tables
from .studio import FIELDS, MIN_SUBJECT_PIXELS, TERRAINS, TERRAIN_SHARE, label

ARENAS_PER_SHARD = 20
VIEWS = 24
CROP = "minecraft:wheat"


def views(region: list[int], bounds: list[float], rng: np.random.Generator) -> list[list[float]]:
    """
    Camera poses as [x, y, z, yaw, pitch], each looking at a random plot from up to 8 blocks out and 1 to 7 over the crops,
    inside bounds, the geofence, since a view outside it ends the episode
    """
    x0, _, z0, x1, y1, z1 = region
    cx, cz = (x0 + x1 + 1) / 2, (z0 + z1 + 1) / 2
    half = max(x1 - x0, z1 - z0) / 2 + 1
    out = []
    for _ in range(VIEWS):
        target = (rng.integers(x0, x1 + 1) + 0.5, y1 + 0.4, rng.integers(z0, z1 + 1) + 0.5)
        angle = rng.uniform(0, 2 * math.pi)
        reach = rng.uniform(0, half + 8)
        eye = (
            float(np.clip(cx + math.cos(angle) * reach, bounds[0] + 1, bounds[3] - 1)),
            min(y1 + rng.uniform(1.0, 7.0), bounds[4] - 1),
            float(np.clip(cz + math.sin(angle) * reach, bounds[2] + 1, bounds[5] - 1)),
        )
        dx, dy, dz = target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]
        yaw = math.degrees(math.atan2(-dx, dz))
        pitch = -math.degrees(math.atan2(dy, max(math.hypot(dx, dz), 1e-3)))
        out.append([*eye, yaw, pitch])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arenas", type=int, default=150)
    parser.add_argument("--seed", type=int, default=380_000, help="first arena seed, clear of studio, demo, and evaluation seeds")
    parser.add_argument("--out", type=Path, default=DATA / "fields")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    arenas = [(args.seed + i, TERRAINS[rng.choice(len(TERRAINS), p=TERRAIN_SHARE)]) for i in range(args.arenas)]
    n_shards = (len(arenas) + ARENAS_PER_SHARD - 1) // ARENAS_PER_SHARD
    print(f"{len(arenas)} arenas in {n_shards} shards", flush=True)
    env = DroneEnv(task="harvest_crops", tools=False, streams=("rgb", "depth", "mask", "state"), action_pause_ms=0, width=FRAME[1], height=FRAME[0])
    client = env.client
    classes, ripe = state_tables(client.state_names)
    entities = entity_table(client.mask_ids["entities"])
    base = client.mask_ids["entityBase"]
    failed = []
    try:
        for k in range(n_shards):
            shard = args.out / f"shard-{k:04d}"
            if (shard / "done").exists():
                continue
            parts: dict[str, list] = {f: [] for f in FIELDS}
            subjects = []
            for seed, terrain in arenas[k * ARENAS_PER_SHARD : (k + 1) * ARENAS_PER_SHARD]:
                try:
                    obs = client.reset(seed=seed, task="harvest_crops", terrain=terrain, size=5)
                except BridgeError as e:
                    failed.append({"seed": seed, "error": str(e)})
                    continue
                region = client.arena()["job"]["region"]
                for pose in views(region, obs.state["bounds"], np.random.default_rng(seed)):
                    frame = client.view(pose)
                    cls, rp = label(frame, classes, ripe, entities, base)
                    if (cls == INDEX[CROP]).sum() < MIN_SUBJECT_PIXELS:
                        continue
                    parts["rgb"].append(frame.rgb)
                    parts["depth"].append(frame.depth.astype(np.float16))
                    parts["cls"].append(cls)
                    parts["ripe"].append(rp)
                    subjects.append(CROP)
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
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "failed.json").write_text(json.dumps(failed, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
