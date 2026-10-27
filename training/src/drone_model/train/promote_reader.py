"""
Promotes a reader trained with --baseline to checkpoints/reader.pt when, on the same validation frames,
it reads the named class better than the baseline and loses at most TOLERANCE mean IoU.
Exits 1 when it doesn't

    python -m drone_model.train.promote_reader checkpoints/reader-candidate.pt --class minecraft:wheat
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from ..paths import CHECKPOINTS

TOLERANCE = 0.005


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--class", dest="name", required=True)
    args = parser.parse_args()

    report = json.loads(args.candidate.with_suffix(".json").read_text(encoding="utf-8"))
    new, old = report["best"], report["baseline"]
    new_class, old_class = new["per_class_iou"].get(args.name, 0.0), old["per_class_iou"].get(args.name, 0.0)
    better = new_class > old_class and new["mean_iou"] >= old["mean_iou"] - TOLERANCE
    print(f"mean IoU {old['mean_iou']:.3f} -> {new['mean_iou']:.3f}, {args.name} {old_class:.3f} -> {new_class:.3f}, {'promoted' if better else 'kept the baseline'}")
    if not better:
        sys.exit(1)
    shutil.copyfile(args.candidate, CHECKPOINTS / "reader.pt")
    shutil.copyfile(args.candidate.with_suffix(".json"), CHECKPOINTS / "reader.json")


if __name__ == "__main__":
    main()
