"""
The brain reports in reports/brain/ as one Markdown table, task by terrain, the newest report of each
scripts/evaluate.ps1 writes them and calls this at the end

    python -m drone_model.evaluate.summary --since 20261025-120000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..paths import REPORTS


def rows(reports: list[Path]) -> list[str]:
    newest: dict[tuple[str, str], dict] = {}
    for path in sorted(reports):
        report = json.loads(path.read_text(encoding="utf-8"))
        newest[(report["task"], report["terrain"])] = report
    lines = ["| Task | Terrain | Success | Median steps | Notes |", "| --- | --- | --- | --- | --- |"]
    for (task, terrain), report in newest.items():
        results = report["results"]
        wins = sum(r["success"] for r in results)
        steps = sorted(r["steps"] for r in results if r["success"])
        median = steps[len(steps) // 2] if steps else "-"
        notes = []
        if report.get("prey_killed") is not None:
            notes.append(f"{report['prey_killed']:.0%} of prey killed")
        failures = [r["outcome"] for r in results if not r["success"]]
        if failures:
            notes.append(", ".join(f"{failures.count(o)} {o}" for o in sorted(set(failures))))
        lines.append(f"| {task} | {terrain} | {wins} of {len(results)} | {median} | {', '.join(notes)} |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", default="", help="only reports written at or after this yyyymmdd-HHMMSS")
    parser.add_argument("--out", type=Path, default=REPORTS / "brain" / "summary.md")
    args = parser.parse_args()
    reports = [p for p in (REPORTS / "brain").glob("*.json") if p.stem[-15:] >= args.since]
    if not reports:
        raise SystemExit(f"no brain reports in {REPORTS / 'brain'} since {args.since or 'ever'}")
    table = "\n".join(rows(reports))
    args.out.write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
