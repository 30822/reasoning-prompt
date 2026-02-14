#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stratified sampling over JAMA specialty JSON files.
- Input dir:  resources/data/jama
- Output:     resources/data/jama_stratified_sample.json
- Keeps per-specialty proportions by sampling ~ratio from each file.

Assumptions:
- Each input JSON is either:
  (A) a list of cases
  (B) a dict with a top-level list under one of keys: ["cases", "data", "items"]
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_KEYS = ("cases", "data", "items")


def _load_cases(path: Path) -> Tuple[List[Dict[str, Any]], str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj, "list"
    if isinstance(obj, dict):
        for k in DEFAULT_KEYS:
            if k in obj and isinstance(obj[k], list):
                return obj[k], f"dict[{k}]"
    raise ValueError(f"Unsupported JSON structure in {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="resources/data/jama")
    ap.add_argument("--out_path", type=str, default=None)
    ap.add_argument("--ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42) # 일단 고정, 나중에 바꿀 것
    ap.add_argument("--min_per_file", type=int, default=1)
    ap.add_argument("--glob", type=str, default="jama_raw_*.json")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {in_dir}")

    paths = sorted(in_dir.glob(args.glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {in_dir / args.glob}")

    rng = random.Random(args.seed)

    sampled_all: List[Dict[str, Any]] = []
    summary = []

    for p in paths:
        cases, mode = _load_cases(p)
        n = len(cases)

        # proportional per-file sampling
        k = int(round(n * args.ratio))
        if args.min_per_file is not None:
            k = max(args.min_per_file, k)
        k = min(k, n)

        idx = list(range(n))
        rng.shuffle(idx)
        pick = idx[:k]
        picked_cases = [cases[i] for i in pick]

        # annotate source specialty for traceability
        specialty = p.stem.replace("jama_raw_", "")
        for c in picked_cases:
            if isinstance(c, dict) and "source_specialty" not in c:
                c["source_specialty"] = specialty

        sampled_all.extend(picked_cases)
        summary.append((p.name, specialty, n, k, mode))

    # default output path
    if args.out_path is None:
        out_path = Path("resources/data") / f"jama_stratified_sample.json"
    else:
        out_path = Path(args.out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sampled_all, ensure_ascii=False, indent=2), encoding="utf-8")

    # print summary
    total_in = sum(s[2] for s in summary)
    total_out = sum(s[3] for s in summary)
    print(f"[OK] Wrote: {out_path}  (sampled {total_out}/{total_in} = {total_out/total_in:.3%})")
    print("Per-file breakdown:")
    for fname, specialty, n, k, mode in summary:
        print(f" - {fname:28s} ({specialty:14s})  {k:4d}/{n:4d}  [{mode}]")

    # Also write a tiny manifest next to output (optional but useful)
    manifest = {
        "seed": args.seed,
        "ratio": args.ratio,
        "min_per_file": args.min_per_file,
        "input_dir": str(in_dir),
        "files": [
            {"file": fname, "specialty": specialty, "n_total": n, "n_sampled": k, "mode": mode}
            for fname, specialty, n, k, mode in summary
        ],
        "output": str(out_path),
        "n_total_all": total_in,
        "n_sampled_all": total_out,
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()