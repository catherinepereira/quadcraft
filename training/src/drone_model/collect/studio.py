"""
Block reader training frames from studio scenes: for every block and mob the reader learns, scenes of it with random
neighbors, backdrop, light, and time of day, each captured from the views the mod picks (task/Studio.java).
Every frame is labeled from the state and mask streams and written straight into shards of training arrays,
data/studio/<shard>/{rgb,depth,cls,ripe}.npy, with meta.json naming the class list and each frame's subject.
The plan is fixed by the seed, so a stopped run picks up at the first unfinished shard

Blocks the offline renderer draws (collect.render) get only a few scenes here, enough for validation frames from the game

    python -m drone_model.collect.studio --block-scenes 20 --rendered-block-scenes 2 --mob-scenes 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quadcraft import BridgeError, DroneEnv

from ..paths import DATA
from ..perception.reader import (
    BLOCK_KINDS,
    CLASS_HASH,
    FRAME,
    INDEX,
    MOB_KINDS,
    entity_table,
    state_tables,
)

SHARD = 512
FIELDS = ("rgb", "depth", "cls", "ripe")
# a frame where the subject covers fewer pixels than this went to its neighbors, it's dropped
MIN_SUBJECT_PIXELS = 12
TERRAINS = ("flat", "rough", "cave")
TERRAIN_SHARE = (0.5, 0.3, 0.2)


def plan(block_scenes: int, rendered_block_scenes: int, mob_scenes: int, seed: int) -> list[tuple[str, int, str, int]]:
    """Every scene in order as subject, seed, terrain, and obstacles, rounds of every subject so each shard mixes them"""
    from ..render.assets import Assets

    assets = Assets()
    count = {b: rendered_block_scenes if assets.renderable(b) else block_scenes for b in BLOCK_KINDS}
    count.update({m: mob_scenes for m in MOB_KINDS})
    rng = np.random.default_rng(seed)
    scenes = []
    for r in range(max(count.values())):
        subjects = [k for k in (*BLOCK_KINDS, *MOB_KINDS) if r < count[k]]
        for subject in rng.permutation(subjects):
            terrain = TERRAINS[rng.choice(len(TERRAINS), p=TERRAIN_SHARE)]
            scenes.append((str(subject), seed + len(scenes), terrain, int(rng.integers(0, 4))))
    return scenes


def label(obs, classes: np.ndarray, ripe: np.ndarray, entities: np.ndarray, entity_base: int) -> tuple[np.ndarray, np.ndarray]:
    """Reader classes and ripeness per pixel, blocks from the state stream and mobs from the mask's entity ids"""
    ids = obs.block_states
    cls = classes[np.minimum(ids, len(classes) - 1)]
    mask = obs.mask.astype(np.int64)
    mob = (mask >= entity_base) & (mask - entity_base < len(entities))
    cls[mob] = entities[mask[mob] - entity_base]
    return cls, ripe[np.minimum(ids, len(ripe) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-scenes", type=int, default=20, help="scenes of each block the offline renderer can't draw, each with up to 8 views")
    parser.add_argument("--rendered-block-scenes", type=int, default=2, help="scenes of each block collect.render draws")
    parser.add_argument("--mob-scenes", type=int, default=40, help="scenes of each mob")
    parser.add_argument("--seed", type=int, default=300_000, help="first scene seed, clear of demo and evaluation seeds")
    parser.add_argument("--out", type=Path, default=DATA / "studio")
    args = parser.parse_args()

    scenes = plan(args.block_scenes, args.rendered_block_scenes, args.mob_scenes, args.seed)
    n_shards = (len(scenes) + SHARD - 1) // SHARD
    print(f"{len(scenes)} scenes in {n_shards} shards", flush=True)
    env = DroneEnv(task="studio", tools=False, streams=("rgb", "depth", "mask", "state"), action_pause_ms=0, width=FRAME[1], height=FRAME[0])
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
            for subject, seed, terrain, obstacles in scenes[k * SHARD : (k + 1) * SHARD]:
                try:
                    client.reset(seed=seed, task="studio", subject=subject, terrain=terrain, obstacles=obstacles)
                except BridgeError as e:
                    failed.append({"subject": subject, "seed": seed, "error": str(e)})
                    continue
                for pose in (client.arena() or {}).get("views", []):
                    frame = client.view(pose)
                    cls, rp = label(frame, classes, ripe, entities, base)
                    if (cls == INDEX[subject]).sum() < MIN_SUBJECT_PIXELS:
                        continue
                    parts["rgb"].append(frame.rgb)
                    parts["depth"].append(frame.depth.astype(np.float16))
                    parts["cls"].append(cls)
                    parts["ripe"].append(rp)
                    subjects.append(subject)
            shard.mkdir(parents=True, exist_ok=True)
            for f, v in parts.items():
                np.save(shard / f"{f}.npy", np.stack(v))
            (shard / "meta.json").write_text(json.dumps({"classes": CLASS_HASH, "frame": FRAME, "source": "game", "subjects": subjects}), encoding="utf-8")
            (shard / "done").touch()
            print(f"shard {k + 1}/{n_shards}: {len(subjects)} frames, {len(failed)} scenes failed so far", flush=True)
    finally:
        env.close()
        if failed:
            (args.out / "failed.json").write_text(json.dumps(failed, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
