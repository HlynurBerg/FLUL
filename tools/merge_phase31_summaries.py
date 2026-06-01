"""Merge Phase 3.1 summary parts into a canonical phase31_summary.json.

Use after a resumed sweep: takes the "old" summary (some alpha points) and the
"new" summary (the alpha points produced by the resumed run) and writes a
single output JSON containing all unique points, sorted by descending alpha.

Example:
    python merge_phase31_summaries.py \\
        --old phase31_summary_part1.json.bak \\
        --new phase31_summary_part2.json \\
        --output phase31_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old", type=Path, required=True)
    p.add_argument("--new", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    with open(args.old, "r", encoding="utf-8") as f:
        old = json.load(f)
    with open(args.new, "r", encoding="utf-8") as f:
        new = json.load(f)

    by_tag = {}
    for pt in old.get("points", []):
        by_tag[pt["tag"]] = pt
    # New overrides old on the same tag.
    for pt in new.get("points", []):
        by_tag[pt["tag"]] = pt

    points = list(by_tag.values())

    def _sort_key(pt):
        v = pt.get("point")
        if isinstance(v, list):
            return -float(v[0])
        return -float(v)

    points.sort(key=_sort_key)

    merged = dict(new)
    merged["points"] = points
    # Carry through sanity metadata from whichever side has it set.
    for key in ("sanity_algos", "sanity_alpha"):
        if not merged.get(key):
            merged[key] = old.get(key, merged.get(key))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged {len(points)} unique points to {args.output}")
    for pt in points:
        algos_done = list((pt.get("algos") or {}).keys())
        print(f"  tag={pt['tag']} point={pt.get('point')} algos={algos_done}")


if __name__ == "__main__":
    main()
