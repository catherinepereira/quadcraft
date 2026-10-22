"""
Trains the block reader on studio frames, rendered by collect.render and captured in the game by collect.studio, in
shards of rgb, depth, and per-pixel labels. Game labels come from the mod's block state ids, and for mobs from the mask
stream's entity ids.
The reader itself only gets rgb and depth
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from pathlib import Path

# fixed-size cache blocks fragment and run out of memory beside the desktop's share of the GPU
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from torch.nn import functional as F

from ..paths import CHECKPOINTS, DATA
from ..perception.reader import CLASS_HASH, CLASSES, FRAME, NOT_CROP, BlockReader
from ..torch_utils import BestCheckpoint, Throttle, autocast, pick_device, split_episodes, write_report

FIELDS = ("rgb", "depth", "cls", "ripe")
# the pretrained encoder's learning rate as a share of the decoder's, so the first steps don't undo its pretraining
ENCODER_LR = 0.1
# the share of each frame's pixels the class loss reads, full-size logits over every class don't fit in memory
LOSS_PIXELS = 0.25
EVAL_BATCH = 8
# batches read ahead by each loader thread
PREFETCH = 4
LOADERS = 2


def load_frames(dirs: list[Path]) -> list[dict]:
    """
    Every finished shard in dirs memory-mapped, shards labeled for another class list or captured at another size left out.
    Each carries its source, "game" for collect.studio and "render" for collect.render
    """
    out, stale = [], 0
    for data in dirs:
        for shard in sorted(p for p in data.glob("shard-*") if (p / "done").exists()):
            meta = json.loads((shard / "meta.json").read_text(encoding="utf-8"))
            if meta["classes"] != CLASS_HASH or tuple(meta["frame"]) != FRAME:
                stale += 1
                continue
            arrays = {k: np.load(shard / f"{k}.npy", mmap_mode="r") for k in FIELDS}
            out.append({**arrays, "shard": f"{data.name}/{shard.name}", "source": meta["source"]})
    if stale:
        print(f"left out {stale} shards labeled for another class list or captured at another size", flush=True)
    if not out:
        raise SystemExit(f"no shards in {', '.join(map(str, dirs))}, collect some with collect.studio or collect.render")
    return out


def split_frames(shards: list[dict], rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    """Validation shards from the game's frames when there are any, so the score is on what the drone sees, every rendered shard trains"""
    game = [s for s in shards if s["source"] == "game"]
    rendered = [s for s in shards if s["source"] != "game"]
    if not game:
        return split_episodes(rendered, 0.1, rng)
    train, val = split_episodes(game, 0.1, rng)
    return train + rendered, val


def sample(episodes, batch: int, rng: np.random.Generator, augment: bool) -> dict[str, torch.Tensor]:
    picks = rng.integers(0, len(episodes), size=batch)
    rows = {k: [] for k in FIELDS}
    for e in picks:
        ep = episodes[e]
        n = rng.integers(0, len(ep["cls"]))
        flip = augment and rng.random() < 0.5
        for k in FIELDS:
            v = np.asarray(ep[k][n])
            rows[k].append(v[:, ::-1].copy() if flip else v)
    out = {k: torch.from_numpy(np.stack(v).astype(np.int64) if k == "cls" else np.stack(v)) for k, v in rows.items()}
    if augment:
        scale = torch.empty(batch, 1, 1, 1).uniform_(0.7, 1.3)
        out["rgb"] = (out["rgb"].float() * scale).clamp(0, 255).to(torch.uint8)
    return out


def batches(episodes, batch: int, rng: np.random.Generator):
    """Training batches read on background threads, so pulling frames off disk overlaps the GPU's step"""
    ready: queue.Queue = queue.Queue(maxsize=PREFETCH * LOADERS)

    def fill(gen: np.random.Generator) -> None:
        while True:
            ready.put({k: v.pin_memory() for k, v in sample(episodes, batch, gen, augment=True).items()})

    for gen in rng.spawn(LOADERS):
        threading.Thread(target=fill, args=(gen,), daemon=True).start()
    while True:
        yield ready.get()


def class_weights(episodes, rng: np.random.Generator) -> torch.Tensor:
    """Inverse square root frequency, so floor and sky don't drown out a single block of bricks"""
    counts = np.zeros(len(CLASSES))
    for ep in episodes[:: max(1, len(episodes) // 40)]:
        counts += np.bincount(np.asarray(ep["cls"][:: 5]).ravel(), minlength=len(CLASSES))
    w = 1.0 / np.sqrt(np.maximum(counts / counts.sum(), 1e-5))
    # scaled by the classes the data has, most of the list may be missing from a small set
    return torch.tensor(w / w[counts > 0].mean(), dtype=torch.float32)


def evaluate(model, episodes, device, rng, throttle: Throttle | None = None) -> dict[str, float]:
    model.eval()
    n = len(CLASSES)
    inter = torch.zeros(n, dtype=torch.float64, device=device)
    union = torch.zeros(n, dtype=torch.float64, device=device)
    correct = total = ripe_ok = ripe_n = 0
    with torch.no_grad():
        for ep in episodes:
            for start in range(0, len(ep["cls"]), EVAL_BATCH):
                rgb = torch.from_numpy(np.asarray(ep["rgb"][start : start + EVAL_BATCH])).to(device)
                depth = torch.from_numpy(np.asarray(ep["depth"][start : start + EVAL_BATCH]).astype(np.float32)).to(device)
                cls = torch.from_numpy(np.asarray(ep["cls"][start : start + EVAL_BATCH]).astype(np.int64)).to(device)
                ripe = np.asarray(ep["ripe"][start : start + EVAL_BATCH])
                with autocast(device):
                    logits, ripe_logits = model(rgb, depth)
                pred = logits.argmax(1)
                hit = pred == cls
                correct += int(hit.sum())
                total += cls.numel()
                both = torch.bincount(cls[hit], minlength=n)
                inter += both
                union += torch.bincount(pred.ravel(), minlength=n) + torch.bincount(cls.ravel(), minlength=n) - both
                crop = ripe != NOT_CROP
                if crop.any():
                    ripe_pred = (ripe_logits[:, 0].float().cpu().numpy() > 0)[crop]
                    ripe_ok += (ripe_pred == (ripe[crop] == 1)).sum()
                    ripe_n += crop.sum()
                if throttle is not None:
                    throttle()
    inter, union = inter.cpu().numpy(), union.cpu().numpy()
    present = union > 0
    iou = inter[present] / union[present]
    per_class = {CLASSES[c]: float(inter[c] / union[c]) for c in np.flatnonzero(present)}
    return {"pixel_acc": float(correct / total), "mean_iou": float(iou.mean()), "ripe_acc": float(ripe_ok / max(ripe_n, 1)), "per_class_iou": per_class}


def gpu_peak(device) -> str:
    """The most memory the CUDA allocator has held, as a log suffix"""
    return f" peak {torch.cuda.max_memory_reserved(device) / 1e9:.1f} GB" if device.type == "cuda" else ""


def init_from(model: BlockReader, checkpoint: Path, device) -> None:
    """Loads an older reader's weights, its class head row by row for the classes both have"""
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    weights = saved["model"]
    model.load_state_dict({k: v for k, v in weights.items() if not k.startswith("classes.")}, strict=False)
    with torch.no_grad():
        for i, name in enumerate(saved["classes"]):
            if name in CLASSES:
                model.classes.weight[CLASSES.index(name)] = weights["classes.weight"][i]
                model.classes.bias[CLASSES.index(name)] = weights["classes.bias"][i]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, nargs="+", default=[DATA / "studio", DATA / "render"])
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=2500)
    parser.add_argument("--out", type=Path, default=CHECKPOINTS / "reader.pt")
    parser.add_argument("--init", type=Path, default=None, help="start from a trained reader, classes it doesn't have start fresh, otherwise from ImageNet weights")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-share", type=float, default=1.0, help="the share of the time the GPU works, 0.4 leaves room for a game")
    parser.add_argument("--baseline", type=Path, default=None, help="a trained reader to score on the same validation frames first, for comparison")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = pick_device()
    train, val = split_frames(load_frames([d for d in args.data if d.exists()]), rng)
    sources = ", ".join(f"{sum(e['source'] == s for e in train)} {s}" for s in sorted({e["source"] for e in train}))
    print(f"train {len(train)} shards ({sum(len(e['cls']) for e in train)} frames, {sources}), val {len(val)} {val[0]['source']} shards, device {device}", flush=True)

    baseline = None
    if args.baseline is not None:
        reference = BlockReader(pretrained=False).to(device)
        init_from(reference, args.baseline, device)
        baseline = evaluate(reference, val, device, rng)
        print(f"baseline {args.baseline.name} mean IoU {baseline['mean_iou']:.3f} ripe acc {baseline['ripe_acc']:.3f}", flush=True)
        del reference
        torch.cuda.empty_cache()

    model = BlockReader(pretrained=args.init is None).to(device)
    if args.init is not None:
        init_from(model, args.init, device)
    weights = class_weights(train, rng).to(device)
    decoder = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW([{"params": model.encoder.parameters()}, {"params": decoder}], lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[args.lr * ENCODER_LR, args.lr], total_steps=args.steps)
    best = BestCheckpoint(args.out, higher_is_better=True)
    history = []
    start = time.perf_counter()
    throttle = Throttle(args.gpu_share)
    loader = batches(train, args.batch_size, rng)
    for step in range(1, args.steps + 1):
        model.train()
        b = {k: v.to(device, non_blocking=True) for k, v in next(loader).items()}
        with autocast(device):
            d = model.features(b["rgb"], b["depth"].float())
            ripe_logits = model.ripe(d)
            pixels = torch.randint(0, d.shape[2] * d.shape[3], (d.shape[0], int(d.shape[2] * d.shape[3] * LOSS_PIXELS)), device=device)
            picked = d.flatten(2).gather(2, pixels[:, None].expand(-1, d.shape[1], -1))
            logits = model.classes(picked.unsqueeze(-1))[..., 0]
            loss = F.cross_entropy(logits.float(), b["cls"].long().flatten(1).gather(1, pixels), weight=weights)
            crop = b["ripe"] != NOT_CROP
            if crop.any():
                loss = loss + F.binary_cross_entropy_with_logits(ripe_logits[:, 0].float()[crop], (b["ripe"][crop] == 1).float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        throttle()
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, val, device, rng, throttle)
            history.append({"step": step, "loss": loss.item(), **{k: v for k, v in metrics.items() if k != "per_class_iou"}})
            saved = best.offer(metrics["mean_iou"], model=model.state_dict(), step=step, metrics=metrics, classes=CLASSES)
            if saved:
                final = metrics
            print(
                f"step {step} loss {loss.item():.3f} pixel acc {metrics['pixel_acc']:.3f} mean IoU {metrics['mean_iou']:.3f} "
                f"ripe acc {metrics['ripe_acc']:.3f} {time.perf_counter() - start:.0f}s{gpu_peak(device)}{' saved' if saved else ''}",
                flush=True,
            )
    write_report(args.out.with_suffix(".json"), args, best=final, history=history, baseline=baseline)
    worst = sorted(final["per_class_iou"].items(), key=lambda kv: kv[1])[:6]
    print(f"best mean IoU {best.best:.3f}, weakest classes: " + ", ".join(f"{n.split(':')[-1]} {v:.2f}" for n, v in worst))


if __name__ == "__main__":
    main()
