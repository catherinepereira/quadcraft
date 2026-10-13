"""
Region embeddings of a trained block reader for the embedding explorer in Developer/quadcraft-embeddings.
One point per class region in a studio frame: the decoder's last features, which feed the class head, averaged over
the region's pixels. Blocks and mobs are laid out with UMAP separately, in 2D and 3D.
Shards split into training and validation the way train.reader splits them, so each point says which side it came from

    python -m drone_model.evaluate.embeddings --checkpoint checkpoints/reader-v7.pt --dest ../../quadcraft-embeddings/public/data
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..paths import CHECKPOINTS, DATA
from ..perception.reader import CLASSES, MOB_SET, OTHER, SKY, BlockReader
from ..torch_utils import pick_device, split_episodes
from ..train.reader import load_frames

# a region smaller than this is a few pixels of edge, too small to crop or average
MIN_PIXELS = 24
THUMB = 64
# frames per forward pass, the logits over 1303 classes take 100 MB a frame
BATCH = 8


def candidates(shards: list[dict]) -> dict[int, list[tuple[int, int]]]:
    """(shard, frame) pairs for every class with a big enough region there"""
    n_cls = len(CLASSES)
    out: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for e, ep in enumerate(shards):
        for start in range(0, len(ep["cls"]), 256):
            cls = np.asarray(ep["cls"][start : start + 256]).reshape(-1, ep["cls"].shape[1] * ep["cls"].shape[2]).astype(np.int64)
            counts = np.bincount((cls + np.arange(len(cls))[:, None] * n_cls).ravel(), minlength=len(cls) * n_cls).reshape(len(cls), n_cls)
            counts[:, [SKY, OTHER]] = 0
            for n, c in zip(*np.nonzero(counts >= MIN_PIXELS)):
                out[int(c)].append((e, start + int(n)))
    return out


def pick(found: list[tuple[int, int]], k: int, val: set[int], rng: np.random.Generator) -> list[tuple[int, int]]:
    """Up to k regions, validation shards first, and spread over shards before taking a second from one"""
    order = rng.permutation(len(found))
    shuffled = sorted((found[i] for i in order), key=lambda en: en[0] not in val)
    chosen, used = [], defaultdict(int)
    for limit in (1, 2, 1_000):
        for en in shuffled:
            if len(chosen) == k:
                return chosen
            if en not in chosen and used[en[0]] < limit:
                chosen.append(en)
                used[en[0]] += 1
    return chosen


def thumbnail(rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    """A square crop around the region, the rest of the crop dimmed"""
    rows, cols = np.nonzero(mask)
    cy, cx = (rows.min() + rows.max()) / 2, (cols.min() + cols.max()) / 2
    half = max(rows.max() - rows.min(), cols.max() - cols.min()) / 2 + 4
    h, w = mask.shape
    y0, y1 = int(max(0, cy - half)), int(min(h, cy + half + 1))
    x0, x1 = int(max(0, cx - half)), int(min(w, cx + half + 1))
    crop = rgb[y0:y1, x0:x1].astype(np.float32)
    crop[~mask[y0:y1, x0:x1]] *= 0.35
    return Image.fromarray(crop.astype(np.uint8)).resize((THUMB, THUMB), Image.NEAREST)


def layout(features: np.ndarray, dims: int, seed: int) -> np.ndarray:
    import umap

    scaled = (features - features.mean(0)) / (features.std(0) + 1e-6)
    # the spectral start needs more points than a small export has
    small = len(features) <= 50
    model = umap.UMAP(
        n_components=dims, n_neighbors=min(15, len(features) - 1), min_dist=0.1, init="random" if small else "spectral", random_state=seed
    )
    coords = model.fit_transform(scaled)
    return np.round(coords, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINTS / "reader.pt")
    parser.add_argument("--data", type=Path, default=DATA / "studio")
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--blocks-per-class", type=int, default=12)
    parser.add_argument("--mobs-per-class", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0, help="the training run's seed, which picks its validation shards")
    args = parser.parse_args()

    shards = load_frames(args.data)
    _, val_ids = split_episodes(range(len(shards)), 0.1, np.random.default_rng(args.seed))
    val = set(val_ids)
    found = candidates(shards)
    print(f"{sum(len(v) for v in found.values())} regions over {len(found)} classes", flush=True)

    rng = np.random.default_rng(args.seed)
    groups = {"blocks": [], "mobs": []}
    for c, regions in sorted(found.items()):
        is_mob = CLASSES[c] in MOB_SET
        k = args.mobs_per_class if is_mob else args.blocks_per_class
        groups["mobs" if is_mob else "blocks"] += [(c, e, n) for e, n in pick(regions, k, val, rng)]

    device = pick_device()
    saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = BlockReader().to(device).eval()
    model.load_state_dict(saved["model"])
    features = {}
    model.dec1.register_forward_hook(lambda _m, _i, out: features.__setitem__("d1", out))

    for group, picks in groups.items():
        by_frame = defaultdict(list)
        for i, (c, e, n) in enumerate(picks):
            by_frame[(e, n)].append((i, c))
        frames = sorted(by_frame)
        emb = np.zeros((len(picks), model.classes.in_channels), dtype=np.float32)
        pred = np.zeros(len(picks), dtype=np.int64)
        conf = np.zeros(len(picks), dtype=np.float32)
        thumbs: dict[int, list[tuple[int, Image.Image]]] = defaultdict(list)
        with torch.no_grad():
            for start in range(0, len(frames), BATCH):
                batch = frames[start : start + BATCH]
                rgb = np.stack([np.asarray(shards[e]["rgb"][n]) for e, n in batch])
                depth = np.stack([np.asarray(shards[e]["depth"][n]).astype(np.float32) for e, n in batch])
                logits, _ = model(torch.from_numpy(rgb).to(device), torch.from_numpy(depth).to(device))
                probs = logits.float().softmax(1)
                guess = probs.argmax(1)
                d1 = features["d1"].float()
                for b, (e, n) in enumerate(batch):
                    cls = torch.from_numpy(np.asarray(shards[e]["cls"][n]).astype(np.int64)).to(device)
                    for i, c in by_frame[(e, n)]:
                        mask = cls == c
                        emb[i] = d1[b][:, mask].mean(1).cpu().numpy()
                        pred[i] = int(torch.bincount(guess[b][mask], minlength=len(CLASSES)).argmax())
                        conf[i] = float(probs[b, c][mask].mean())
                        thumbs[c].append((i, thumbnail(rgb[b], mask.cpu().numpy())))

        # one strip of thumbnails per class, a point names its slot in the strip
        slot = np.zeros(len(picks), dtype=np.int64)
        (args.dest / group).mkdir(parents=True, exist_ok=True)
        for c, items in thumbs.items():
            strip = Image.new("RGB", (THUMB * len(items), THUMB))
            for j, (i, img) in enumerate(sorted(items, key=lambda t: t[0])):
                strip.paste(img, (j * THUMB, 0))
                slot[i] = j
            strip.save(args.dest / group / f"{c}.webp", quality=85)

        print(f"{group}: {len(picks)} regions, laying out", flush=True)
        xy, xyz = layout(emb, 2, args.seed), layout(emb, 3, args.seed)
        out = {
            "cls": [c for c, _, _ in picks],
            "pred": pred.tolist(),
            "conf": np.round(conf, 3).tolist(),
            "val": [int(e in val) for _, e, _ in picks],
            "shard": [shards[e]["shard"] for _, e, _ in picks],
            "frame": [n for _, _, n in picks],
            "slot": slot.tolist(),
            "xy": xy.ravel().tolist(),
            "xyz": xyz.ravel().tolist(),
        }
        (args.dest / f"{group}.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")

    metrics = saved.get("metrics", {})
    meta = {
        "checkpoint": args.checkpoint.name,
        "step": saved.get("step"),
        "meanIou": metrics.get("mean_iou"),
        "pixelAcc": metrics.get("pixel_acc"),
        "classes": list(CLASSES),
        "iou": [metrics.get("per_class_iou", {}).get(name) for name in CLASSES],
        "thumb": THUMB,
    }
    (args.dest / "meta.json").write_text(json.dumps(meta, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {args.dest}")


if __name__ == "__main__":
    main()
