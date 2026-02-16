#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stratified sampling over JAMA specialty JSON files.
- Input dir:  resources/data/jama (configurable via --in_dir)
- Output:     configurable via --out_path / -o (default: resources/data/jama_stratified_sample.json)
- --total N: draw exactly N samples, proportionally allocated across files
- --ratio R: when --total not set, sample R fraction from each file (default 0.1)

Assumptions:
- Each input JSON is either:
  (A) a list of cases
  (B) a dict with a top-level list under one of keys: ["cases", "data", "items"]
"""

from __future__ import annotations

import argparse
import json
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


def _allocate_proportional(target_total: int, sizes: List[int]) -> List[int]:
    """Allocate target_total across strata proportionally; sum == target_total."""
    total_n = sum(sizes)
    if total_n == 0:
        return [0] * len(sizes)

    k_list = [target_total * n // total_n for n in sizes]
    remainder = target_total - sum(k_list)

    # Give +1 to strata with largest fractional remainder
    if remainder > 0:
        frac_remainders = [
            (target_total * sizes[i] / total_n - k_list[i], i)
            for i in range(len(sizes)) if k_list[i] < sizes[i]
        ]
        frac_remainders.sort(key=lambda x: -x[0])
        for j in range(min(remainder, len(frac_remainders))):
            idx = frac_remainders[j][1]
            k_list[idx] = min(k_list[idx] + 1, sizes[idx])

    return k_list


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stratified sampling over JAMA specialty JSON files. "
        "Use --total for fixed sample count, or --ratio for proportional sampling."
    )
    ap.add_argument("--in_dir", type=str, default="resources/data/jama")
    ap.add_argument("--out_path", "--output", "-o", type=str, default=None,
                    help="Output JSON path (default: resources/data/jama_stratified_sample.json)")
    ap.add_argument("--total", "-n", type=int, default=None,
                    help="Total number of samples to draw (stratified by file size)")
    ap.add_argument("--ratio", type=float, default=0.1,
                    help="Sampling ratio per file when --total is not set (default: 0.1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_per_file", type=int, default=1,
                    help="Minimum samples per file (ignored when --total is used)")
    ap.add_argument("--glob", type=str, default="jama_raw_*.json")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {in_dir}")

    paths = sorted(in_dir.glob(args.glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {in_dir / args.glob}")

    # Load all cases and sizes first
    all_cases: List[Tuple[List[Dict[str, Any]], str, str]] = []
    sizes: List[int] = []
    for p in paths:
        cases, mode = _load_cases(p)
        specialty = p.stem.replace("jama_raw_", "")
        all_cases.append((cases, mode, specialty))
        sizes.append(len(cases))

    total_n = sum(sizes)
    if args.total is not None:
        if args.total <= 0 or args.total > total_n:
            raise ValueError(f"--total must be in (0, {total_n}], got {args.total}")
        k_per_file = _allocate_proportional(args.total, sizes)
    else:
        min_per = args.min_per_file
        k_per_file = []
        for n in sizes:
            k = max(min_per, min(n, int(round(n * args.ratio))))
            k_per_file.append(k)

    rng = random.Random(args.seed)
    sampled_all: List[Dict[str, Any]] = []
    summary: List[Tuple[str, str, int, int, str]] = []

    for (cases, mode, specialty), k, p in zip(all_cases, k_per_file, paths):
        n = len(cases)
        idx = list(range(n))
        rng.shuffle(idx)
        pick = idx[:k]
        picked_cases = [cases[i] for i in pick]

        for c in picked_cases:
            if isinstance(c, dict) and "source_specialty" not in c:
                c["source_specialty"] = specialty

        sampled_all.extend(picked_cases)
        summary.append((p.name, specialty, n, k, mode))

    if args.out_path is None:
        out_path = Path("resources/data") / "jama_stratified_sample.json"
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

    manifest = {
        "seed": args.seed,
        "total_requested": args.total,
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