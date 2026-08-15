"""Statistical analysis of judged clinician–AI dialogues.

Writes four CSVs:
  bootstrap_permutation.csv  — paired bootstrap CI and permutation tests vs P1
  component_level.csv        — matched-pair effects of CoT, CL1, CL2, SR
  interaction.csv            — two-way cell means and interaction contrasts
  correlation.csv            — rank correlation of prompt effects across models
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, t, ttest_1samp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.collaborative_performance import exp_id_to_pid, get_all_ai_utterances
from src.utils import load_json_sanitized

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIDS = [f"P{i}" for i in range(1, 17)]
COMPONENTS = ["CoT", "CL1", "CL2", "SR"]
COMPONENT_MAP = {
    "P1":  {"CoT": 0, "CL1": 0, "CL2": 0, "SR": 0},
    "P2":  {"CoT": 0, "CL1": 0, "CL2": 1, "SR": 0},
    "P3":  {"CoT": 0, "CL1": 0, "CL2": 0, "SR": 1},
    "P4":  {"CoT": 0, "CL1": 0, "CL2": 1, "SR": 1},
    "P5":  {"CoT": 1, "CL1": 0, "CL2": 0, "SR": 0},
    "P6":  {"CoT": 1, "CL1": 0, "CL2": 1, "SR": 0},
    "P7":  {"CoT": 1, "CL1": 0, "CL2": 0, "SR": 1},
    "P8":  {"CoT": 1, "CL1": 0, "CL2": 1, "SR": 1},
    "P9":  {"CoT": 0, "CL1": 1, "CL2": 0, "SR": 0},
    "P10": {"CoT": 0, "CL1": 1, "CL2": 1, "SR": 0},
    "P11": {"CoT": 0, "CL1": 1, "CL2": 0, "SR": 1},
    "P12": {"CoT": 0, "CL1": 1, "CL2": 1, "SR": 1},
    "P13": {"CoT": 1, "CL1": 1, "CL2": 0, "SR": 0},
    "P14": {"CoT": 1, "CL1": 1, "CL2": 1, "SR": 0},
    "P15": {"CoT": 1, "CL1": 1, "CL2": 0, "SR": 1},
    "P16": {"CoT": 1, "CL1": 1, "CL2": 1, "SR": 1},
}
INTERACTIONS = [
    ("CoT", "CL1"),
    ("CoT", "CL2"),
    ("CoT", "SR"),
    ("CL1", "CL2"),
    ("CL1", "SR"),
    ("CL2", "SR"),
]


@dataclass(frozen=True)
class Counts:
    tot_correct: int
    succ_accept: int
    tot_incorrect: int
    succ_argue: int


def add_counts(a: Counts, b: Counts) -> Counts:
    return Counts(
        tot_correct=a.tot_correct + b.tot_correct,
        succ_accept=a.succ_accept + b.succ_accept,
        tot_incorrect=a.tot_incorrect + b.tot_incorrect,
        succ_argue=a.succ_argue + b.succ_argue,
    )


def pcollab_from_counts(c: Counts) -> float:
    if c.tot_correct <= 0 or c.tot_incorrect <= 0:
        return float("nan")
    r_acc = c.succ_accept / c.tot_correct
    r_arg = c.succ_argue / c.tot_incorrect
    if r_acc <= 0 or r_arg <= 0:
        return 0.0
    return math.sqrt(r_acc * r_arg)


def pooled_valid_acceptance(c: Counts) -> float:
    if c.tot_correct <= 0:
        return float("nan")
    return c.succ_accept / c.tot_correct


def pooled_valid_argumentation(c: Counts) -> float:
    if c.tot_incorrect <= 0:
        return float("nan")
    return c.succ_argue / c.tot_incorrect


def holm_correction(pvals: np.ndarray) -> np.ndarray:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.zeros(m)
    max_val = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * pvals[idx]
        max_val = max(max_val, val)
        adj[idx] = min(max_val, 1.0)
    return adj


def _stat_fn(metric: str):
    if metric == "Pcollab":
        return pcollab_from_counts
    if metric == "Valid Acceptance":
        return pooled_valid_acceptance
    if metric == "Valid Argumentation":
        return pooled_valid_argumentation
    raise ValueError(metric)


def build_case_prompt_counts(annotated_path: str | Path, model_key: str) -> Dict[str, Dict[str, Counts]]:
    data = load_json_sanitized(annotated_path)
    logs = data[model_key]["logs"]
    out: Dict[str, Dict[str, Counts]] = {}

    for log in logs:
        pid = exp_id_to_pid(str(log.get("experiment_id", "")))
        if pid == "P?":
            continue
        case_id = log.get("case_id", "")
        if not case_id:
            continue
        utts = get_all_ai_utterances(log.get("dialogue", []))
        if not utts:
            continue

        case_map = out.setdefault(str(case_id), {})
        cur = case_map.get(pid, Counts(0, 0, 0, 0))
        mode = log.get("mode")

        if mode in ("incorrect", "error"):
            cur = Counts(
                tot_correct=cur.tot_correct,
                succ_accept=cur.succ_accept,
                tot_incorrect=cur.tot_incorrect + len(utts),
                succ_argue=cur.succ_argue + sum(1 for a, v in utts if a == "ARGUE" and v == "VALID"),
            )
        elif mode == "correct":
            cur = Counts(
                tot_correct=cur.tot_correct + len(utts),
                succ_accept=cur.succ_accept + sum(1 for a, v in utts if a == "ACCEPT" and v == "VALID"),
                tot_incorrect=cur.tot_incorrect,
                succ_argue=cur.succ_argue,
            )
        case_map[pid] = cur
    return out


def prompt_metrics(counts_map: Dict[str, Dict[str, Counts]]) -> pd.DataFrame:
    """Dataset-level rates per prompt (pool counts across cases)."""
    rows = []
    for pid in PIDS:
        c = Counts(0, 0, 0, 0)
        for cmap in counts_map.values():
            if pid in cmap:
                c = add_counts(c, cmap[pid])
        rows.append({
            "prompt_id": pid,
            "Valid Acceptance": pooled_valid_acceptance(c),
            "Valid Argumentation": pooled_valid_argumentation(c),
            "Pcollab": pcollab_from_counts(c),
            **COMPONENT_MAP[pid],
        })
    return pd.DataFrame(rows)


def paired_ids(counts_map: Dict[str, Dict[str, Counts]], a: str, b: str, metric: str) -> List[str]:
    ids = []
    for case_id, cmap in counts_map.items():
        if a not in cmap or b not in cmap:
            continue
        ca, cb = cmap[a], cmap[b]
        if metric == "Pcollab":
            ok = ca.tot_correct > 0 and ca.tot_incorrect > 0 and cb.tot_correct > 0 and cb.tot_incorrect > 0
        elif metric == "Valid Acceptance":
            ok = ca.tot_correct > 0 and cb.tot_correct > 0
        else:
            ok = ca.tot_incorrect > 0 and cb.tot_incorrect > 0
        if ok:
            ids.append(case_id)
    return ids


def _sum_counts(counts_map, case_ids, pid) -> Counts:
    c = Counts(0, 0, 0, 0)
    for case_id in case_ids:
        c = add_counts(c, counts_map[case_id][pid])
    return c


def paired_bootstrap_ci(
    case_ids: List[str],
    counts_map: Dict[str, Dict[str, Counts]],
    pid_a: str,
    pid_b: str,
    metric: str,
    rng: np.random.Generator,
    n_boot: int,
    alpha: float,
) -> Tuple[float, float, float]:
    stat = _stat_fn(metric)
    n = len(case_ids)
    obs = stat(_sum_counts(counts_map, case_ids, pid_a)) - stat(_sum_counts(counts_map, case_ids, pid_b))
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ids = [case_ids[j] for j in idx]
        boot[i] = stat(_sum_counts(counts_map, ids, pid_a)) - stat(_sum_counts(counts_map, ids, pid_b))
    lo = float(np.percentile(boot, 100 * (alpha / 2)))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return float(obs), lo, hi


def paired_permutation_p(
    case_ids: List[str],
    counts_map: Dict[str, Dict[str, Counts]],
    pid_a: str,
    pid_b: str,
    metric: str,
    rng: np.random.Generator,
    n_perm: int,
) -> Tuple[float, float]:
    stat = _stat_fn(metric)
    obs = stat(_sum_counts(counts_map, case_ids, pid_a)) - stat(_sum_counts(counts_map, case_ids, pid_b))
    extreme = 0
    for _ in range(n_perm):
        ca = Counts(0, 0, 0, 0)
        cb = Counts(0, 0, 0, 0)
        swaps = rng.random(len(case_ids)) < 0.5
        for case_id, sw in zip(case_ids, swaps):
            a = counts_map[case_id][pid_a]
            b = counts_map[case_id][pid_b]
            if sw:
                a, b = b, a
            ca = add_counts(ca, a)
            cb = add_counts(cb, b)
        d = stat(ca) - stat(cb)
        if abs(d) >= abs(obs):
            extreme += 1
    p = (extreme + 1) / (n_perm + 1)
    return float(obs), float(p)


def analyze_bootstrap_permutation(
    counts_map: Dict[str, Dict[str, Counts]],
    model: str,
    rng: np.random.Generator,
    n_boot: int,
    n_perm: int,
    alpha: float,
) -> pd.DataFrame:
    baseline = "P1"
    rows = []
    for metric in ("Pcollab", "Valid Argumentation", "Valid Acceptance"):
        ids0 = paired_ids(counts_map, baseline, baseline, metric)
        m0 = _stat_fn(metric)(_sum_counts(counts_map, ids0, baseline)) if ids0 else float("nan")
        rows.append({
            "Model": model,
            "metric": metric,
            "prompt_id": baseline,
            "n_cases": len(ids0),
            "value": m0,
            "value_P1": m0,
            "delta_vs_P1": np.nan,
            "CI_low": np.nan,
            "CI_high": np.nan,
            "perm_p": np.nan,
            "perm_p_holm": np.nan,
            "sig_boot": np.nan,
            "sig_perm": np.nan,
        })

        tmp = []
        pvals = []
        for pid in PIDS[1:]:
            ids = paired_ids(counts_map, pid, baseline, metric)
            print(f"  [{model}] {metric} {pid} n={len(ids)} bootstrap+permutation...", flush=True)
            if not ids:
                tmp.append({
                    "Model": model, "metric": metric, "prompt_id": pid, "n_cases": 0,
                    "value": np.nan, "value_P1": np.nan, "delta_vs_P1": np.nan,
                    "CI_low": np.nan, "CI_high": np.nan, "perm_p": np.nan,
                })
                pvals.append(np.nan)
                continue
            val = _stat_fn(metric)(_sum_counts(counts_map, ids, pid))
            base = _stat_fn(metric)(_sum_counts(counts_map, ids, baseline))
            delta, lo, hi = paired_bootstrap_ci(
                ids, counts_map, pid, baseline, metric, rng, n_boot, alpha
            )
            _, p = paired_permutation_p(ids, counts_map, pid, baseline, metric, rng, n_perm)
            tmp.append({
                "Model": model, "metric": metric, "prompt_id": pid, "n_cases": len(ids),
                "value": val, "value_P1": base, "delta_vs_P1": delta,
                "CI_low": lo, "CI_high": hi, "perm_p": p,
            })
            pvals.append(p)

        p_arr = np.array(pvals, dtype=float)
        adj = np.full_like(p_arr, np.nan)
        valid = np.isfinite(p_arr)
        if valid.any():
            adj[np.where(valid)[0]] = holm_correction(p_arr[valid])
        for r, p_adj in zip(tmp, adj):
            r["perm_p_holm"] = p_adj
            lo, hi = r["CI_low"], r["CI_high"]
            r["sig_boot"] = bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)) if np.isfinite(lo) else np.nan
            r["sig_perm"] = bool(np.isfinite(p_adj) and p_adj < 0.05) if np.isfinite(p_adj) else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def _pairs_for_component(comp: str) -> List[Tuple[str, str]]:
    pairs = []
    for i, p0 in enumerate(PIDS):
        for p1 in PIDS[i + 1:]:
            m0, m1 = COMPONENT_MAP[p0], COMPONENT_MAP[p1]
            if m0[comp] != 0 or m1[comp] != 1:
                continue
            if all(m0[k] == m1[k] for k in COMPONENTS if k != comp):
                pairs.append((p0, p1))
    return pairs


def analyze_component_level(summary: pd.DataFrame, model: str, alpha: float = 0.05) -> pd.DataFrame:
    sm = summary.set_index("prompt_id")
    rows = []
    for comp in COMPONENTS:
        pairs = _pairs_for_component(comp)
        for metric in ("Pcollab", "Valid Argumentation", "Valid Acceptance"):
            deltas = []
            pair_labels = []
            for p0, p1 in pairs:
                if p0 not in sm.index or p1 not in sm.index:
                    continue
                d = float(sm.loc[p1, metric] - sm.loc[p0, metric])
                if np.isfinite(d):
                    deltas.append(d)
                    pair_labels.append(f"{p0}->{p1}")
            n = len(deltas)
            if n == 0:
                continue
            mean = float(np.mean(deltas))
            se = float(np.std(deltas, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            tcrit = float(t.ppf(1 - alpha / 2, n - 1)) if n > 1 else float("nan")
            p = float(ttest_1samp(deltas, popmean=0.0).pvalue) if n > 1 else float("nan")
            rows.append({
                "Model": model,
                "component": comp,
                "metric": metric,
                "n_pairs": n,
                "mean_delta": mean,
                "SE": se,
                "CI_low": mean - tcrit * se if np.isfinite(tcrit) else float("nan"),
                "CI_high": mean + tcrit * se if np.isfinite(tcrit) else float("nan"),
                "p_value": p,
                "pairs": ";".join(pair_labels),
            })
    return pd.DataFrame(rows)


def analyze_interaction(summary: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = []
    for metric in ("Pcollab", "Valid Argumentation", "Valid Acceptance"):
        for a, b in INTERACTIONS:
            grouped = {(0, 0): [], (1, 0): [], (0, 1): [], (1, 1): []}
            for pid, flags in COMPONENT_MAP.items():
                row = summary.loc[summary["prompt_id"] == pid]
                if row.empty:
                    continue
                grouped[(int(flags[a]), int(flags[b]))].append(float(row.iloc[0][metric]))
            y00 = float(np.nanmean(grouped[(0, 0)])) if grouped[(0, 0)] else float("nan")
            y10 = float(np.nanmean(grouped[(1, 0)])) if grouped[(1, 0)] else float("nan")
            y01 = float(np.nanmean(grouped[(0, 1)])) if grouped[(0, 1)] else float("nan")
            y11 = float(np.nanmean(grouped[(1, 1)])) if grouped[(1, 1)] else float("nan")
            effect_a_without_b = y10 - y00
            effect_a_with_b = y11 - y01
            interaction = effect_a_with_b - effect_a_without_b
            rows.append({
                "Model": model,
                "metric": metric,
                "factor_a": a,
                "factor_b": b,
                "y_00": y00,
                "y_10": y10,
                "y_01": y01,
                "y_11": y11,
                "effect_a_without_b": effect_a_without_b,
                "effect_a_with_b": effect_a_with_b,
                "interaction": interaction,
            })
    return pd.DataFrame(rows)


CORR_COLUMNS = [
    "metric", "model_a", "model_b", "n_prompts",
    "spearman_rho", "spearman_p", "kendall_tau", "kendall_p",
]


def analyze_correlation(boot_df: pd.DataFrame) -> pd.DataFrame:
    models = list(boot_df["Model"].dropna().unique())
    rows = []
    if len(models) < 2:
        return pd.DataFrame(columns=CORR_COLUMNS)

    for metric in ("Pcollab", "Valid Argumentation", "Valid Acceptance"):
        sub = boot_df[(boot_df["metric"] == metric) & (boot_df["prompt_id"] != "P1")]
        wide = sub.pivot_table(index="prompt_id", columns="Model", values="delta_vs_P1", aggfunc="first")
        for m1, m2 in combinations(models, 2):
            if m1 not in wide.columns or m2 not in wide.columns:
                continue
            x = wide[m1].to_numpy(dtype=float)
            y = wide[m2].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            n = int(mask.sum())
            if n < 3:
                rho = p_rho = tau = p_tau = float("nan")
            else:
                rho, p_rho = spearmanr(x[mask], y[mask])
                tau, p_tau = kendalltau(x[mask], y[mask])
            rows.append({
                "metric": metric,
                "model_a": m1,
                "model_b": m2,
                "n_prompts": n,
                "spearman_rho": float(rho),
                "spearman_p": float(p_rho),
                "kendall_tau": float(tau),
                "kendall_p": float(p_tau),
            })
    return pd.DataFrame(rows, columns=CORR_COLUMNS)


def main():
    parser = argparse.ArgumentParser(description="Bootstrap, permutation, component-level, interaction, and correlation analysis.")
    parser.add_argument(
        "--annotated",
        default="output/evaluation/annotated_results.json",
        help="Path to annotated_results.json",
    )
    parser.add_argument("--out-dir", default="output/analysis")
    parser.add_argument("--model", action="append", dest="models", help="Model key; repeatable. Default: all models in the file.")
    parser.add_argument("--n-boot", type=int, default=20000)
    parser.add_argument("--n-perm", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    annotated = Path(args.annotated)
    if not annotated.is_absolute():
        annotated = PROJECT_ROOT / annotated
    if not annotated.exists():
        raise FileNotFoundError(f"Annotated file not found: {annotated}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_json_sanitized(annotated)
    models = args.models or list(data.keys())
    rng = np.random.default_rng(args.seed)

    boot_parts, comp_parts, inter_parts = [], [], []
    for model in models:
        if model not in data:
            raise KeyError(f"Model {model!r} not in {annotated}. Keys: {list(data.keys())}")
        print(f"Loading counts: {model}")
        counts_map = build_case_prompt_counts(annotated, model)
        summary = prompt_metrics(counts_map)
        boot_parts.append(
            analyze_bootstrap_permutation(
                counts_map, model, rng, args.n_boot, args.n_perm, args.alpha
            )
        )
        comp_parts.append(analyze_component_level(summary, model, args.alpha))
        inter_parts.append(analyze_interaction(summary, model))

    boot_df = pd.concat(boot_parts, ignore_index=True)
    comp_df = pd.concat(comp_parts, ignore_index=True)
    inter_df = pd.concat(inter_parts, ignore_index=True)
    corr_df = analyze_correlation(boot_df)

    paths = {
        "bootstrap_permutation": out_dir / "bootstrap_permutation.csv",
        "component_level": out_dir / "component_level.csv",
        "interaction": out_dir / "interaction.csv",
        "correlation": out_dir / "correlation.csv",
    }
    boot_df.to_csv(paths["bootstrap_permutation"], index=False)
    comp_df.to_csv(paths["component_level"], index=False)
    inter_df.to_csv(paths["interaction"], index=False)
    corr_df.to_csv(paths["correlation"], index=False)

    print("Saved:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
