"""
Block reader training frames rendered offline on the GPU, without the game: studio scenes of every block that draws
from its model (drone_model.render), written as shards in collect.studio's format with "source": "render" in meta.json.
Blocks drawn by their own renderers (chests, heads, banners, shulker boxes) and mobs still come from collect.studio.
The plan is fixed by the seed, so a stopped run picks up at the first unfinished shard

    python -m drone_model.collect.render --block-scenes 50 --workers 8
    python -m drone_model.collect.render --preview 12
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..paths import DATA
from ..perception.reader import BLOCK_KINDS, CLASS_HASH, DEPTH_MAX, FRAME, INDEX
from .studio import FIELDS, MIN_SUBJECT_PIXELS, TERRAIN_SHARE, TERRAINS

SHARD = 1024
VIEWS = 4


def plan(blocks: list[str], block_scenes: int, seed: int) -> list[tuple[str, int, str]]:
    """Every scene in order as subject, seed, and terrain, rounds of every block so each shard mixes them"""
    rng = np.random.default_rng(seed)
    scenes = []
    for _ in range(block_scenes):
        for subject in rng.permutation(blocks):
            scenes.append((str(subject), seed + len(scenes), TERRAINS[rng.choice(len(TERRAINS), p=TERRAIN_SHARE)]))
    return scenes


class Studio:
    """One GPU context with every block texture, staging and rendering scenes"""

    def __init__(self) -> None:
        from ..render.assets import Assets
        from ..render.gl import Renderer
        from ..render.scene import Stager

        self.assets = Assets()
        self.blocks = [b for b in BLOCK_KINDS if self.assets.renderable(b)]
        names = set().union(*(self.assets.texture_names(b) for b in self.blocks))
        self.renderer = Renderer(self.assets, names, FRAME, DEPTH_MAX)
        self.stager = Stager(self.assets, self.renderer.textures, self.blocks)

    def scene(self, subject: str, seed: int, terrain: str, views: int):
        """Up to views frames of one scene as (rgb, depth, cls, ripe) each"""
        from ..render.mesh import build, far_ground

        scene = self.stager.stage(subject, seed, terrain)
        rng = random.Random(seed + 1)
        frames = {name: rng.randrange(1 << 16) for name in self.renderer.textures.base}
        extra = []
        if scene.far_ground is not None:
            block, y = scene.far_ground
            top = next(p for p in scene.cells.values() if p.block == block)
            part = top.parts[0]
            up = int(np.flatnonzero(part.face == 1)[0])
            tint = top.tints[part.tint[up]] if part.tint[up] >= 0 else np.ones(3, dtype=np.float32)
            extra.append(far_ground(self.renderer.textures.layer(part.texture[up], 0), tint, y + 0.99))
        mesh = build(scene.cells, self.renderer.textures, frames, extra)
        self.renderer.load(mesh)
        target = INDEX[subject]
        out = []
        for view in self.stager.views(scene, rng):
            cls, ripe, depth = self.renderer.labels(view.eye, view.forward)
            if (cls == target).sum() < MIN_SUBJECT_PIXELS:
                continue
            rgb = self.renderer.color(view.eye, view.forward, scene.env)
            out.append((rgb, depth.astype(np.float16), cls, ripe))
            if len(out) == views:
                break
        return out


def write_shard(shard: Path, parts: dict[str, list], subjects: list[str]) -> None:
    shard.mkdir(parents=True, exist_ok=True)
    for f, v in parts.items():
        np.save(shard / f"{f}.npy", np.stack(v))
    meta = {"classes": CLASS_HASH, "frame": FRAME, "source": "render", "subjects": subjects}
    (shard / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (shard / "done").touch()


def worker(rank: int, workers: int, scenes: list, n_shards: int, out: Path, views: int) -> None:
    studio = Studio()
    for k in range(rank, n_shards, workers):
        shard = out / f"shard-{k:04d}"
        if (shard / "done").exists():
            continue
        start = time.perf_counter()
        parts: dict[str, list] = {f: [] for f in FIELDS}
        subjects, failed = [], 0
        for subject, seed, terrain in scenes[k * SHARD : (k + 1) * SHARD]:
            try:
                frames = studio.scene(subject, seed, terrain, views)
            except ValueError:
                failed += 1
                continue
            for rgb, depth, cls, ripe in frames:
                for f, v in zip(FIELDS, (rgb, depth, cls, ripe)):
                    parts[f].append(v)
                subjects.append(subject)
        write_shard(shard, parts, subjects)
        print(f"shard {k + 1}/{n_shards}: {len(subjects)} frames, {failed} scenes failed, {time.perf_counter() - start:.0f}s", flush=True)


def preview(n: int, dest: Path, seed: int) -> None:
    """A sheet of n scenes, each frame beside its subject's label in magenta"""
    studio = Studio()
    rng = random.Random(seed)
    rows = []
    start = time.perf_counter()
    count = 0
    for i in range(n):
        subject = rng.choice(studio.blocks)
        frames = studio.scene(subject, seed + i, rng.choices(TERRAINS, TERRAIN_SHARE)[0], 1)
        count += len(frames)
        for rgb, _, cls, _ in frames:
            over = rgb.copy()
            hit = cls == INDEX[subject]
            over[hit] = (0.4 * over[hit] + 0.6 * np.array([255, 0, 255])).astype(np.uint8)
            rows.append(np.concatenate([rgb, over], 1))
            print(subject, int(hit.sum()), "px")
    print(f"{count} frames in {time.perf_counter() - start:.1f}s")
    Image.fromarray(np.concatenate(rows, 0)).save(dest)
    print(f"wrote {dest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-scenes", type=int, default=100, help="scenes of each block")
    parser.add_argument("--views", type=int, default=VIEWS, help="frames per scene")
    parser.add_argument("--seed", type=int, default=500_000)
    parser.add_argument("--workers", type=int, default=8, help="processes, each with its own GPU context, scene staging and meshing run on the CPU")
    parser.add_argument("--out", type=Path, default=DATA / "render")
    parser.add_argument("--preview", type=int, default=0, help="render this many scenes to a PNG sheet and stop")
    parser.add_argument("--preview-out", type=Path, default=Path("render-preview.png"))
    args = parser.parse_args()
    if args.preview:
        preview(args.preview, args.preview_out, args.seed)
        return

    from ..render.assets import Assets

    assets = Assets()
    blocks = [b for b in BLOCK_KINDS if assets.renderable(b)]
    scenes = plan(blocks, args.block_scenes, args.seed)
    n_shards = (len(scenes) + SHARD - 1) // SHARD
    print(f"{len(blocks)} blocks, {len(scenes)} scenes in {n_shards} shards", flush=True)
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(r, args.workers, scenes, n_shards, args.out, args.views)) for r in range(args.workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
