"""Shard a LongMemEval dataset JSON into per-QA files.

Each LME dataset (oracle / s_cleaned / m_cleaned) is a single JSON
array of ~500 question items. Loading the whole 264MB file in every
subprocess just to pick one item is the dominant memory hog when
running parallel sweeps.

This one-time shard converts ``foo.json`` into ``foo_shards/<qid>.json``,
each containing a 1-element array with just that QA's data. Subprocesses
then load only ~500KB instead of 264MB — 500× memory reduction.

The dataset loader (`dataset.py:load_items`) auto-detects the shard dir
on the question_id and uses it; falls back to the full file if shards
are missing. So sharding is opt-in: run this script once before the
sweep, and parallel runs use shards automatically.

Usage:
  python scripts/shard_lme_dataset.py external/longmemeval/data/longmemeval_s_cleaned.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "dataset_path",
        help="Path to LME dataset JSON (e.g. longmemeval_s_cleaned.json)",
    )
    args = ap.parse_args()

    src = Path(args.dataset_path)
    if not src.exists():
        raise SystemExit(f"dataset not found: {src}")

    out_dir = src.with_suffix("").as_posix() + "_shards"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {src} ({src.stat().st_size / 1e6:.1f} MB)...")
    data = json.loads(src.read_text(encoding="utf-8"))
    print(f"  → {len(data)} items")

    written = 0
    skipped = 0
    index = []  # tiny manifest for orchestrator's QA-list lookup
    for entry in data:
        qid = entry.get("question_id")
        if not qid:
            skipped += 1
            continue
        shard = out_dir / f"{qid}.json"
        # Wrap as a 1-element list so load_items() works as-is on shards.
        shard.write_text(json.dumps([entry], ensure_ascii=False))
        index.append({
            "question_id": qid,
            "question_type": entry.get("question_type", ""),
        })
        written += 1

    # Tiny index file — lets the orchestrator filter/sample QAs without
    # loading the full ~264MB dataset.
    (out_dir / "_index.json").write_text(json.dumps(index, ensure_ascii=False))

    avg_size = sum(
        p.stat().st_size for p in out_dir.glob("*.json")
        if p.name != "_index.json"
    ) / max(written, 1)
    idx_size = (out_dir / "_index.json").stat().st_size
    print(f"wrote {written} shards (skipped {skipped}) → {out_dir}")
    print(f"  avg shard size: {avg_size / 1024:.1f} KB")
    print(f"  index size: {idx_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
