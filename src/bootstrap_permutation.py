import argparse
import math
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import yaml

# 프로젝트 루트 import (기존 코드 스타일 유지)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.medcobe import get_all_ai_utterances
from src.utils import load_json_sanitized


# ============================
# Config
# ============================
EXPERIMENTS_YAML = "./resources/experiments.yaml"
MODEL_NAME = "openai/o3"

_T1_ORDER = ["B", "B_COT", "B_CL", "B_COT_CL"]
_T2_ORDER = ["B", "B_CL", "B_SR", "B_CL_SR"]

# 통계 설정
N_BOOT = 20000
N_PERM = 50000
SEED = 42

ALPHA = 0.05  # 95% CI


# ============================
# Helpers
# ============================
def load_model_map(experiments_yaml_path: str) -> dict[str, str]:
    cfg = yaml.safe_load(Path(experiments_yaml_path).read_text(encoding="utf-8"))
    model_map = cfg.get("models", cfg)
    return model_map

def resolve_short_model_name(full_model_name: str, model_map: dict[str, str]) -> str:
    inv = {v: k for k, v in model_map.items()}
    return inv.get(full_model_name, full_model_name.split("/")[-1].replace("-", "_"))

def exp_id_to_cid(experiment_id: str) -> Optional[str]:
    try:
        rhs = experiment_id.split("-", 1)[1]
        t1, t2 = rhs.split("__", 1)
        r = _T1_ORDER.index(t1)
        c = _T2_ORDER.index(t2)
        return f"C{r*4 + c + 1}"
    except Exception:
        return None

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


# ============================
# 핵심: per-case counts를 미리 만들어두고
# bootstrap/permutation에서 빠르게 합산
# ============================
@dataclass(frozen=True)
class Counts:
    # correct 모드(= confirmation)
    tot_correct: int
    succ_accept: int
    # incorrect/error 모드(= correction)
    tot_incorrect: int
    succ_argue: int

def add_counts(a: Counts, b: Counts) -> Counts:
    return Counts(
        tot_correct=a.tot_correct + b.tot_correct,
        succ_accept=a.succ_accept + b.succ_accept,
        tot_incorrect=a.tot_incorrect + b.tot_incorrect,
        succ_argue=a.succ_argue + b.succ_argue,
    )

def medcobe_from_dataset_counts(c: Counts) -> float:
    """dataset-level MEDCOBE = sqrt(R_accept * R_argue) with strict validity."""
    if c.tot_correct <= 0 or c.tot_incorrect <= 0:
        return float("nan")

    r_accept = c.succ_accept / c.tot_correct
    r_argue = c.succ_argue / c.tot_incorrect

    # 논문/기존 구현 관례: 둘 중 하나가 0이면 score 0
    if r_accept <= 0 or r_argue <= 0:
        return 0.0
    return math.sqrt(r_accept * r_argue)


def case_r_accept(c: Counts) -> float:
    """Per-case R_accept = valid ACCEPT / correct-mode AI utterances."""
    if c.tot_correct <= 0:
        return float("nan")
    return c.succ_accept / c.tot_correct


def case_r_argue(c: Counts) -> float:
    """Per-case R_argue = valid ARGUE / incorrect-mode AI utterances."""
    if c.tot_incorrect <= 0:
        return float("nan")
    return c.succ_argue / c.tot_incorrect


def pooled_r_accept(c: Counts) -> float:
    """Pooled R_accept after summing counts over cases: sum succ_accept / sum tot_correct."""
    if c.tot_correct <= 0:
        return float("nan")
    return c.succ_accept / c.tot_correct


def pooled_r_argue(c: Counts) -> float:
    """Pooled R_argue after summing counts: sum succ_argue / sum tot_incorrect."""
    if c.tot_incorrect <= 0:
        return float("nan")
    return c.succ_argue / c.tot_incorrect


def build_case_prompt_counts(
    annotated_json_path: str,
    model_key: Optional[str] = None,
) -> Dict[str, Dict[str, Counts]]:
    """
    return: case_id -> cid -> Counts
    """
    data = load_json_sanitized(annotated_json_path)
    mk = model_key if model_key is not None else MODEL_NAME
    logs = data[mk]["logs"]

    # 임시 누적용 dict
    tmp: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
    # 위는 쓰기 복잡하니, 바로 counts로 누적
    out: Dict[str, Dict[str, Counts]] = {}

    for log in logs:
        cid = exp_id_to_cid(log.get("experiment_id", ""))
        if cid is None:
            continue
        case_id = log.get("case_id", "")
        if not case_id:
            continue

        mode = log.get("mode")
        dialogue = log.get("dialogue", [])
        ai_utts = get_all_ai_utterances(dialogue)
        if not ai_utts:
            continue

        # 현재 case/cid counts 가져오기
        case_map = out.setdefault(case_id, {})
        cur = case_map.get(cid, Counts(0, 0, 0, 0))

        if mode in ("incorrect", "error"):
            tot_inc = len(ai_utts)
            succ_arg = sum(1 for action, validity in ai_utts if action == "ARGUE" and validity == "VALID")
            cur = Counts(
                tot_correct=cur.tot_correct,
                succ_accept=cur.succ_accept,
                tot_incorrect=cur.tot_incorrect + tot_inc,
                succ_argue=cur.succ_argue + succ_arg,
            )
        elif mode == "correct":
            tot_cor = len(ai_utts)
            succ_acc = sum(1 for action, validity in ai_utts if action == "ACCEPT" and validity == "VALID")
            cur = Counts(
                tot_correct=cur.tot_correct + tot_cor,
                succ_accept=cur.succ_accept + succ_acc,
                tot_incorrect=cur.tot_incorrect,
                succ_argue=cur.succ_argue,
            )
        else:
            # mode unknown -> ignore
            pass

        case_map[cid] = cur

    return out


def experiment_suffix_from_id(experiment_id: str) -> Optional[str]:
    """`o3-B_CL_RO__B_CL_LC` -> `B_CL_RO__B_CL_LC` (prompt robustness 등 C1–C16 그리드 밖 실험용)."""
    exp = (experiment_id or "").strip()
    if not exp:
        return None
    if "-" in exp:
        return exp.split("-", 1)[1]
    return exp


def build_case_experiment_suffix_counts(
    annotated_json_path: str,
    model_key: Optional[str] = None,
) -> Dict[str, Dict[str, Counts]]:
    """
    case_id -> experiment_suffix (T1__T2) -> Counts

    `build_case_prompt_counts`와 동일 로직이나 키를 `exp_id_to_cid` 대신 접미사로 둠.
    """
    data = load_json_sanitized(annotated_json_path)
    mk = model_key if model_key is not None else MODEL_NAME
    logs = data[mk]["logs"]

    out: Dict[str, Dict[str, Counts]] = {}

    for log in logs:
        suf = experiment_suffix_from_id(str(log.get("experiment_id", "")))
        if suf is None:
            continue
        case_id = log.get("case_id", "")
        if not case_id:
            continue

        mode = log.get("mode")
        dialogue = log.get("dialogue", [])
        ai_utts = get_all_ai_utterances(dialogue)
        if not ai_utts:
            continue

        case_map = out.setdefault(case_id, {})
        cur = case_map.get(suf, Counts(0, 0, 0, 0))

        if mode in ("incorrect", "error"):
            tot_inc = len(ai_utts)
            succ_arg = sum(1 for action, validity in ai_utts if action == "ARGUE" and validity == "VALID")
            cur = Counts(
                tot_correct=cur.tot_correct,
                succ_accept=cur.succ_accept,
                tot_incorrect=cur.tot_incorrect + tot_inc,
                succ_argue=cur.succ_argue + succ_arg,
            )
        elif mode == "correct":
            tot_cor = len(ai_utts)
            succ_acc = sum(1 for action, validity in ai_utts if action == "ACCEPT" and validity == "VALID")
            cur = Counts(
                tot_correct=cur.tot_correct + tot_cor,
                succ_accept=cur.succ_accept + succ_acc,
                tot_incorrect=cur.tot_incorrect,
                succ_argue=cur.succ_argue,
            )
        else:
            pass

        case_map[suf] = cur

    return out


# ============================
# Bootstrap & Permutation (paired)
# ============================
def paired_bootstrap_delta_samples(
    case_ids: List[str],
    counts_map: Dict[str, Dict[str, Counts]],
    cid_a: str,
    cid_b: str,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> Tuple[float, np.ndarray]:
    """
    delta = SMed(a) - SMed(b) on dataset-level.
    Returns observed delta and length-n_boot bootstrap sample array.
    """
    base_counts_a = Counts(0, 0, 0, 0)
    base_counts_b = Counts(0, 0, 0, 0)
    for pid in case_ids:
        base_counts_a = add_counts(base_counts_a, counts_map[pid][cid_a])
        base_counts_b = add_counts(base_counts_b, counts_map[pid][cid_b])
    delta_obs = medcobe_from_dataset_counts(base_counts_a) - medcobe_from_dataset_counts(base_counts_b)

    n = len(case_ids)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ca = Counts(0, 0, 0, 0)
        cb = Counts(0, 0, 0, 0)
        for j in idx:
            pid = case_ids[j]
            ca = add_counts(ca, counts_map[pid][cid_a])
            cb = add_counts(cb, counts_map[pid][cid_b])
        boot[i] = medcobe_from_dataset_counts(ca) - medcobe_from_dataset_counts(cb)

    return float(delta_obs), boot


def paired_bootstrap_delta(
    case_ids: List[str],
    counts_map: Dict[str, Dict[str, Counts]],
    cid_a: str,
    cid_b: str,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
    alpha: float = ALPHA,
) -> Tuple[float, float, float]:
    """
    delta = SMed(a) - SMed(b) on dataset-level.
    bootstrap: resample case_ids with replacement, recompute delta.
    """
    delta_obs, boot = paired_bootstrap_delta_samples(
        case_ids, counts_map, cid_a, cid_b, rng, n_boot=n_boot
    )
    lo = np.percentile(boot, 100 * (alpha / 2))
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    return float(delta_obs), float(lo), float(hi)

def paired_permutation_delta_pvalue(
    case_ids: List[str],
    counts_map: Dict[str, Dict[str, Counts]],
    cid_a: str,
    cid_b: str,
    rng: np.random.Generator,
    n_perm: int = N_PERM,
) -> Tuple[float, float]:
    """
    Paired permutation for nonlinear statistic:
    - for each case, with p=0.5 swap (a,b) counts
    - recompute dataset-level delta
    """
    # observed
    ca0 = Counts(0, 0, 0, 0)
    cb0 = Counts(0, 0, 0, 0)
    for pid in case_ids:
        ca0 = add_counts(ca0, counts_map[pid][cid_a])
        cb0 = add_counts(cb0, counts_map[pid][cid_b])
    obs = medcobe_from_dataset_counts(ca0) - medcobe_from_dataset_counts(cb0)

    # permutation null
    extreme = 0
    for _ in range(n_perm):
        ca = Counts(0, 0, 0, 0)
        cb = Counts(0, 0, 0, 0)
        swaps = rng.random(len(case_ids)) < 0.5
        for pid, sw in zip(case_ids, swaps):
            a = counts_map[pid][cid_a]
            b = counts_map[pid][cid_b]
            if sw:
                a, b = b, a
            ca = add_counts(ca, a)
            cb = add_counts(cb, b)

        d = medcobe_from_dataset_counts(ca) - medcobe_from_dataset_counts(cb)
        if abs(d) >= abs(obs):
            extreme += 1

    p = (extreme + 1) / (n_perm + 1)  # add-one smoothing
    return float(obs), float(p)


# ----- Pooled R_accept / R_argue (same swap/resample logic as dataset-level MedCOBE) -----


def _pooled_rate_fn(rate: str):
    if rate == "accept":
        return pooled_r_accept
    if rate == "argue":
        return pooled_r_argue
    raise ValueError('rate must be "accept" or "argue"')


def paired_bootstrap_delta_pooled_rate_samples(
    case_ids: List[str],
    counts_map: Dict[str, Dict[str, Counts]],
    cid_a: str,
    cid_b: str,
    rate: str,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> Tuple[float, np.ndarray]:
    """
    delta = pooled R(a) - pooled R(b) after summing per-case counts (figure-style pooled rates).
    """
    stat = _pooled_rate_fn(rate)
    ca0 = Counts(0, 0, 0, 0)
    cb0 = Counts(0, 0, 0, 0)
    for pid in case_ids:
        ca0 = add_counts(ca0, counts_map[pid][cid_a])
        cb0 = add_counts(cb0, counts_map[pid][cid_b])
    delta_obs = stat(ca0) - stat(cb0)

    n = len(case_ids)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ca = Counts(0, 0, 0, 0)
        cb = Counts(0, 0, 0, 0)
        for j in idx:
            pid = case_ids[j]
            ca = add_counts(ca, counts_map[pid][cid_a])
            cb = add_counts(cb, counts_map[pid][cid_b])
        boot[i] = stat(ca) - stat(cb)

    return float(delta_obs), boot


def paired_bootstrap_delta_pooled_rate(
    case_ids: List[str],
    counts_map: Dict[str, Dict[str, Counts]],
    cid_a: str,
    cid_b: str,
    rate: str,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
    alpha: float = ALPHA,
) -> Tuple[float, float, float]:
    delta_obs, boot = paired_bootstrap_delta_pooled_rate_samples(
        case_ids, counts_map, cid_a, cid_b, rate, rng, n_boot=n_boot
    )
    lo = np.percentile(boot, 100 * (alpha / 2))
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    return float(delta_obs), float(lo), float(hi)


def paired_permutation_delta_pooled_rate_pvalue(
    case_ids: List[str],
    counts_map: Dict[str, Dict[str, Counts]],
    cid_a: str,
    cid_b: str,
    rate: str,
    rng: np.random.Generator,
    n_perm: int = N_PERM,
) -> Tuple[float, float]:
    stat = _pooled_rate_fn(rate)
    ca0 = Counts(0, 0, 0, 0)
    cb0 = Counts(0, 0, 0, 0)
    for pid in case_ids:
        ca0 = add_counts(ca0, counts_map[pid][cid_a])
        cb0 = add_counts(cb0, counts_map[pid][cid_b])
    obs = stat(ca0) - stat(cb0)

    extreme = 0
    for _ in range(n_perm):
        ca = Counts(0, 0, 0, 0)
        cb = Counts(0, 0, 0, 0)
        swaps = rng.random(len(case_ids)) < 0.5
        for pid, sw in zip(case_ids, swaps):
            a = counts_map[pid][cid_a]
            b = counts_map[pid][cid_b]
            if sw:
                a, b = b, a
            ca = add_counts(ca, a)
            cb = add_counts(cb, b)

        d = stat(ca) - stat(cb)
        if abs(d) >= abs(obs):
            extreme += 1

    p = (extreme + 1) / (n_perm + 1)
    return float(obs), float(p)


# ----- Case-level mean paired difference: R_accept or R_argue -----

def paired_case_ids_for_rate(
    counts_map: Dict[str, Dict[str, Counts]],
    cid_a: str,
    cid_b: str,
    *,
    rate: str,
) -> List[str]:
    """
    Paired cases where both arms have a defined per-case rate.
    rate == \"accept\": both tot_correct > 0
    rate == \"argue\": both tot_incorrect > 0
    """
    if rate not in ("accept", "argue"):
        raise ValueError('rate must be \"accept\" or \"argue\"')
    ids = []
    for pid, cmap in counts_map.items():
        if cid_a not in cmap or cid_b not in cmap:
            continue
        a, b = cmap[cid_a], cmap[cid_b]
        if rate == "accept":
            ok = a.tot_correct > 0 and b.tot_correct > 0
        else:
            ok = a.tot_incorrect > 0 and b.tot_incorrect > 0
        if ok:
            ids.append(pid)
    return ids


def _paired_diffs_rate(
    case_ids: List[str],
    counts_map: Dict[str, Dict[str, Counts]],
    cid_a: str,
    cid_b: str,
    rate: str,
) -> np.ndarray:
    if rate == "accept":
        return np.array(
            [
                case_r_accept(counts_map[pid][cid_a])
                - case_r_accept(counts_map[pid][cid_b])
                for pid in case_ids
            ],
            dtype=float,
        )
    return np.array(
        [
            case_r_argue(counts_map[pid][cid_a])
            - case_r_argue(counts_map[pid][cid_b])
            for pid in case_ids
        ],
        dtype=float,
    )


def paired_bootstrap_mean_diff_case_rate_samples(
    case_ids: List[str],
    counts_map: Dict[str, Dict[str, Counts]],
    cid_a: str,
    cid_b: str,
    rate: str,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> Tuple[float, np.ndarray]:
    """
    delta = mean_i [ R(case_i, a) - R(case_i, b) ] with R = case R_accept or case R_argue.
    """
    if not case_ids:
        return float("nan"), np.array([])
    diffs = _paired_diffs_rate(case_ids, counts_map, cid_a, cid_b, rate)
    obs = float(np.mean(diffs))
    n = len(case_ids)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(np.mean(diffs[idx]))
    return obs, boot


def paired_bootstrap_mean_diff_case_rate(
    case_ids: List[str],
    counts_map: Dict[str, Dict[str, Counts]],
    cid_a: str,
    cid_b: str,
    rate: str,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
    alpha: float = ALPHA,
) -> Tuple[float, float, float]:
    obs, boot = paired_bootstrap_mean_diff_case_rate_samples(
        case_ids, counts_map, cid_a, cid_b, rate, rng, n_boot=n_boot
    )
    if boot.size == 0:
        return obs, float("nan"), float("nan")
    lo = np.percentile(boot, 100 * (alpha / 2))
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    return float(obs), float(lo), float(hi)


def paired_permutation_mean_diff_case_rate_pvalue(
    case_ids: List[str],
    counts_map: Dict[str, Dict[str, Counts]],
    cid_a: str,
    cid_b: str,
    rate: str,
    rng: np.random.Generator,
    n_perm: int = N_PERM,
) -> Tuple[float, float]:
    """
    Paired permutation via Rademacher sign-flips on per-case differences
    (equivalent to random swap of condition labels per case).
    """
    if not case_ids:
        return float("nan"), float("nan")
    diffs = _paired_diffs_rate(case_ids, counts_map, cid_a, cid_b, rate)
    obs = float(np.mean(diffs))
    n = len(case_ids)
    extreme = 0
    for _ in range(n_perm):
        signs = np.where(rng.random(n) < 0.5, -1.0, 1.0)
        d_perm = float(np.mean(signs * diffs))
        if abs(d_perm) >= abs(obs):
            extreme += 1
    p = (extreme + 1) / (n_perm + 1)
    return float(obs), float(p)


# ============================
# Analysis wrapper
# ============================
def paired_case_ids(
    counts_map: Dict[str, Dict[str, Counts]],
    cid_a: str,
    cid_b: str,
) -> List[str]:
    """Cases where both prompts exist and both have correct & error modes for dataset-level MedCOBE."""
    ids = []
    for pid, cmap in counts_map.items():
        if cid_a in cmap and cid_b in cmap:
            a = cmap[cid_a]
            b = cmap[cid_b]
            if (a.tot_correct > 0 and a.tot_incorrect > 0 and
                    b.tot_correct > 0 and b.tot_incorrect > 0):
                ids.append(pid)
    return ids


def bootstrap_deltas_vs_baseline(
    annotated_json_path: str,
    *,
    model_key: Optional[str] = None,
    baseline: str = "C1",
    cids: Optional[List[str]] = None,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Paired bootstrap sample arrays: MedCOBE(cid) - MedCOBE(baseline) at dataset level per draw.
    Returns (samples_per_cid, observed_delta_per_cid). Keys are subset of cids excluding baseline.
    """
    counts_map = build_case_prompt_counts(annotated_json_path, model_key=model_key)
    rng = np.random.default_rng(seed)
    want = cids if cids is not None else [f"C{i}" for i in range(2, 17)]
    out_samples: Dict[str, np.ndarray] = {}
    out_obs: Dict[str, float] = {}
    for cid in want:
        if cid == baseline:
            continue
        ids = paired_case_ids(counts_map, cid, baseline)
        if not ids:
            continue
        d_obs, boot = paired_bootstrap_delta_samples(
            ids, counts_map, cid, baseline, rng, n_boot=n_boot
        )
        out_samples[cid] = boot
        out_obs[cid] = d_obs
    return out_samples, out_obs


def analyze_dataset_level(
    annotated_json_path: str,
    model_key: Optional[str] = None,
) -> pd.DataFrame:
    print("Loading counts from JSON...")
    counts_map = build_case_prompt_counts(annotated_json_path, model_key=model_key)
    print(f"  Loaded {len(counts_map)} cases")

    rng = np.random.default_rng(SEED)

    rows = []
    baseline = "C1"

    # baseline의 dataset-level medcobe (C1이 있는 case들 전체)
    base_ids = paired_case_ids(counts_map, "C1", "C1")
    # (C1,C1)는 항상 참이지만 위 조건 때문에 C1에서 두 모드 다 있는 케이스만 남음
    c1_counts = Counts(0, 0, 0, 0)
    for pid in base_ids:
        c1_counts = add_counts(c1_counts, counts_map[pid]["C1"])
    c1_med = medcobe_from_dataset_counts(c1_counts)

    print(f"  C1 baseline: medcobe={c1_med:.4f}, n_cases={len(base_ids)}")
    # C1 row
    rows.append({
        "cid": "C1",
        "medcobe_dataset": c1_med,
        "n_cases_used": len(base_ids),
        "delta_vs_C1": np.nan,
        "CI_low": np.nan,
        "CI_high": np.nan,
        "perm_p": np.nan,
        "perm_p_holm": np.nan,
        "sig_boot": np.nan,
        "sig_perm": np.nan,
    })

    pvals = []
    tmp_results = []

    for i in range(2, 17):
        cid = f"C{i}"
        ids = paired_case_ids(counts_map, cid, baseline)
        print(f"  {cid}: n_cases={len(ids)}, bootstrap...", end=" ", flush=True)

        # dataset-level medcobe for cid using exactly the same paired ids (해석 일관성)
        ci_counts = Counts(0, 0, 0, 0)
        for pid in ids:
            ci_counts = add_counts(ci_counts, counts_map[pid][cid])
        med_k = medcobe_from_dataset_counts(ci_counts)

        # bootstrap CI (delta)
        delta, lo, hi = paired_bootstrap_delta(
            case_ids=ids,
            counts_map=counts_map,
            cid_a=cid,
            cid_b=baseline,
            rng=rng,
            n_boot=N_BOOT,
            alpha=ALPHA,
        )
        print("permutation...", end=" ", flush=True)

        # permutation p-value
        obs, p = paired_permutation_delta_pvalue(
            case_ids=ids,
            counts_map=counts_map,
            cid_a=cid,
            cid_b=baseline,
            rng=rng,
            n_perm=N_PERM,
        )

        print(f"done (delta={delta:.4f})")
        tmp_results.append({
            "cid": cid,
            "medcobe_dataset": med_k,
            "n_cases_used": len(ids),
            "delta_vs_C1": delta,
            "CI_low": lo,
            "CI_high": hi,
            "perm_p": p,
        })
        pvals.append(p)

    pvals = np.array(pvals, dtype=float)
    adj = holm_correction(pvals)

    for r, p_adj in zip(tmp_results, adj):
        r["perm_p_holm"] = p_adj
        r["sig_boot"] = (r["CI_low"] > 0) or (r["CI_high"] < 0)
        r["sig_perm"] = (p_adj < 0.05)
        rows.append(r)

    res = pd.DataFrame(rows)
    # 보기 좋게 정렬: delta 큰 순(단, C1은 맨 위)
    c1 = res[res["cid"] == "C1"]
    others = res[res["cid"] != "C1"].sort_values("delta_vs_C1", ascending=False)
    res = pd.concat([c1, others], ignore_index=True)
    return res


def analyze_pooled_rates_vs_C1(
    annotated_json_path: str,
    model_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Pooled R_accept and R_argue (figure-style: sum counts then ratio), same paired
    case swap / bootstrap as analyze_dataset_level.

    delta_vs_C1 = pooled(Ck) - pooled(C1) on the same paired case set;
    paired cases match paired_case_ids_for_rate (denominators > 0 on both arms).
    """
    print("Loading counts for pooled R_accept / R_argue...")
    counts_map = build_case_prompt_counts(annotated_json_path, model_key=model_key)
    rng = np.random.default_rng(SEED)
    baseline = "C1"

    rows = []
    for metric_label, rate_key in (("R_accept", "accept"), ("R_argue", "argue")):
        stat = _pooled_rate_fn(rate_key)
        base_ids = paired_case_ids_for_rate(counts_map, baseline, baseline, rate=rate_key)
        if base_ids:
            c0 = Counts(0, 0, 0, 0)
            for pid in base_ids:
                c0 = add_counts(c0, counts_map[pid][baseline])
            m0 = float(stat(c0))
        else:
            m0 = float("nan")
        rows.append({
            "metric": metric_label,
            "cid": baseline,
            "pooled_rate_cid": m0,
            "pooled_rate_C1": m0,
            "n_cases_used": len(base_ids),
            "delta_vs_C1": np.nan,
            "CI_low": np.nan,
            "CI_high": np.nan,
            "perm_p": np.nan,
            "perm_p_holm": np.nan,
            "sig_boot": np.nan,
            "sig_perm": np.nan,
        })

        pvals: List[float] = []
        tmp_results: List[dict] = []

        for i in range(2, 17):
            cid = f"C{i}"
            ids = paired_case_ids_for_rate(counts_map, cid, baseline, rate=rate_key)
            print(
                f"  [pooled] {metric_label} {cid}: n_cases={len(ids)}, bootstrap...",
                end=" ",
                flush=True,
            )

            if ids:
                ca = Counts(0, 0, 0, 0)
                cb = Counts(0, 0, 0, 0)
                for pid in ids:
                    ca = add_counts(ca, counts_map[pid][cid])
                    cb = add_counts(cb, counts_map[pid][baseline])
                mk = float(stat(ca))
                mb = float(stat(cb))

                delta, lo, hi = paired_bootstrap_delta_pooled_rate(
                    ids, counts_map, cid, baseline, rate_key, rng, n_boot=N_BOOT
                )
                print("permutation...", end=" ", flush=True)
                _, p = paired_permutation_delta_pooled_rate_pvalue(
                    ids, counts_map, cid, baseline, rate_key, rng, n_perm=N_PERM
                )
                print(f"done (delta={delta:.4f})")
                tmp_results.append({
                    "metric": metric_label,
                    "cid": cid,
                    "pooled_rate_cid": mk,
                    "pooled_rate_C1": mb,
                    "n_cases_used": len(ids),
                    "delta_vs_C1": delta,
                    "CI_low": lo,
                    "CI_high": hi,
                    "perm_p": p,
                })
                pvals.append(p)
            else:
                print("skipped (no paired cases)")
                tmp_results.append({
                    "metric": metric_label,
                    "cid": cid,
                    "pooled_rate_cid": np.nan,
                    "pooled_rate_C1": np.nan,
                    "n_cases_used": 0,
                    "delta_vs_C1": np.nan,
                    "CI_low": np.nan,
                    "CI_high": np.nan,
                    "perm_p": np.nan,
                })
                pvals.append(np.nan)

        p_arr = np.array(pvals, dtype=float)
        valid = np.isfinite(p_arr)
        adj = np.full_like(p_arr, np.nan, dtype=float)
        if valid.any():
            adj_vals = holm_correction(p_arr[valid])
            adj[np.where(valid)[0]] = adj_vals

        for r, p_adj in zip(tmp_results, adj):
            pp = r.get("perm_p", np.nan)
            if pp is None or (isinstance(pp, float) and math.isnan(pp)):
                r["perm_p_holm"] = np.nan
                r["sig_boot"] = np.nan
                r["sig_perm"] = np.nan
            else:
                r["perm_p_holm"] = float(p_adj)
                lo, hi = r.get("CI_low"), r.get("CI_high")
                r["sig_boot"] = (
                    isinstance(lo, (int, float))
                    and isinstance(hi, (int, float))
                    and math.isfinite(lo)
                    and math.isfinite(hi)
                    and ((lo > 0) or (hi < 0))
                )
                r["sig_perm"] = (
                    isinstance(p_adj, (int, float))
                    and math.isfinite(p_adj)
                    and p_adj < 0.05
                )

        rows.extend(tmp_results)

    res = pd.DataFrame(rows)
    parts = []
    for m in ("R_accept", "R_argue"):
        sub = res[res["metric"] == m]
        head = sub[sub["cid"] == baseline]
        tail = sub[sub["cid"] != baseline].sort_values("delta_vs_C1", ascending=False)
        parts.append(pd.concat([head, tail], ignore_index=True))
    return pd.concat(parts, ignore_index=True)


def analyze_case_level_rates(
    annotated_json_path: str,
    model_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Case-level mean R_accept and R_argue: paired bootstrap CI & permutation p-values vs C1.

    Statistic: mean_i [ r(case_i, Ck) - r(case_i, C1) ], where r is per-case rate
    (succ/tot within that case for the prompt).
    """
    print("Loading counts for case-level R_accept / R_argue...")
    counts_map = build_case_prompt_counts(annotated_json_path, model_key=model_key)
    rng = np.random.default_rng(SEED)
    baseline = "C1"

    rows = []
    for metric_label, rate_key in (("R_accept", "accept"), ("R_argue", "argue")):
        base_ids = paired_case_ids_for_rate(counts_map, baseline, baseline, rate=rate_key)
        if base_ids:
            if rate_key == "accept":
                m0 = float(
                    np.mean([case_r_accept(counts_map[pid][baseline]) for pid in base_ids])
                )
            else:
                m0 = float(
                    np.mean([case_r_argue(counts_map[pid][baseline]) for pid in base_ids])
                )
        else:
            m0 = float("nan")
        rows.append({
            "metric": metric_label,
            "cid": baseline,
            "mean_case_rate": m0,
            "mean_baseline_case_rate": m0,
            "n_cases_used": len(base_ids),
            "delta_vs_C1": np.nan,
            "CI_low": np.nan,
            "CI_high": np.nan,
            "perm_p": np.nan,
            "perm_p_holm": np.nan,
            "sig_boot": np.nan,
            "sig_perm": np.nan,
        })

        pvals: List[float] = []
        tmp_results: List[dict] = []

        for i in range(2, 17):
            cid = f"C{i}"
            ids = paired_case_ids_for_rate(counts_map, cid, baseline, rate=rate_key)
            print(
                f"  {metric_label} {cid}: n_cases={len(ids)}, bootstrap...",
                end=" ",
                flush=True,
            )

            if ids:
                if rate_key == "accept":
                    mk = float(
                        np.mean([case_r_accept(counts_map[pid][cid]) for pid in ids])
                    )
                    mb = float(
                        np.mean([case_r_accept(counts_map[pid][baseline]) for pid in ids])
                    )
                else:
                    mk = float(
                        np.mean([case_r_argue(counts_map[pid][cid]) for pid in ids])
                    )
                    mb = float(
                        np.mean([case_r_argue(counts_map[pid][baseline]) for pid in ids])
                    )

                delta, lo, hi = paired_bootstrap_mean_diff_case_rate(
                    ids, counts_map, cid, baseline, rate_key, rng, n_boot=N_BOOT
                )
                print("permutation...", end=" ", flush=True)
                _, p = paired_permutation_mean_diff_case_rate_pvalue(
                    ids, counts_map, cid, baseline, rate_key, rng, n_perm=N_PERM
                )
                print(f"done (delta={delta:.4f})")
                tmp_results.append({
                    "metric": metric_label,
                    "cid": cid,
                    "mean_case_rate": mk,
                    "mean_baseline_case_rate": mb,
                    "n_cases_used": len(ids),
                    "delta_vs_C1": delta,
                    "CI_low": lo,
                    "CI_high": hi,
                    "perm_p": p,
                })
                pvals.append(p)
            else:
                print("skipped (no paired cases)")
                tmp_results.append({
                    "metric": metric_label,
                    "cid": cid,
                    "mean_case_rate": np.nan,
                    "mean_baseline_case_rate": np.nan,
                    "n_cases_used": 0,
                    "delta_vs_C1": np.nan,
                    "CI_low": np.nan,
                    "CI_high": np.nan,
                    "perm_p": np.nan,
                })
                pvals.append(np.nan)

        p_arr = np.array(pvals, dtype=float)
        valid = np.isfinite(p_arr)
        adj = np.full_like(p_arr, np.nan, dtype=float)
        if valid.any():
            adj_vals = holm_correction(p_arr[valid])
            adj[np.where(valid)[0]] = adj_vals

        for r, p_adj in zip(tmp_results, adj):
            pp = r.get("perm_p", np.nan)
            if pp is None or (isinstance(pp, float) and math.isnan(pp)):
                r["perm_p_holm"] = np.nan
                r["sig_boot"] = np.nan
                r["sig_perm"] = np.nan
            else:
                r["perm_p_holm"] = float(p_adj)
                lo, hi = r.get("CI_low"), r.get("CI_high")
                r["sig_boot"] = (
                    isinstance(lo, (int, float))
                    and isinstance(hi, (int, float))
                    and math.isfinite(lo)
                    and math.isfinite(hi)
                    and ((lo > 0) or (hi < 0))
                )
                r["sig_perm"] = (
                    isinstance(p_adj, (int, float))
                    and math.isfinite(p_adj)
                    and p_adj < 0.05
                )

        rows.extend(tmp_results)

    res = pd.DataFrame(rows)
    # metric별로 baseline 먼저, 나머지는 delta 순
    parts = []
    for m in ("R_accept", "R_argue"):
        sub = res[res["metric"] == m]
        head = sub[sub["cid"] == baseline]
        tail = sub[sub["cid"] != baseline].sort_values("delta_vs_C1", ascending=False)
        parts.append(pd.concat([head, tail], ignore_index=True))
    return pd.concat(parts, ignore_index=True)


# ============================
# Run
# ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Paired bootstrap / permutation: MedCOBE; pooled R_accept/R_argue (figure-style); "
            "optional case-mean rate differences."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--rates-only",
        action="store_true",
        help=(
            "Only case-mean R_accept / R_argue (mean of per-case rates; not pooled). "
            "Skips MedCOBE and pooled analysis."
        ),
    )
    mode.add_argument(
        "--pooled-rates-only",
        action="store_true",
        help=(
            "Only pooled R_accept / R_argue (sum counts then ratio; matches typical figure bars). "
            "Skips MedCOBE. Combine with --case-mean-rates to also run case-mean."
        ),
    )
    parser.add_argument(
        "--case-mean-rates",
        action="store_true",
        help=(
            "Also run case-mean rate difference analysis (optional; distinct from pooled figure rates)."
        ),
    )
    args = parser.parse_args()

    model_map = load_model_map(EXPERIMENTS_YAML)
    short = resolve_short_model_name(MODEL_NAME, model_map)

    annotated_json_path = f"experiments/(main) {short}_combined_sample/evaluation/annotated_results.json"
    out_dir = Path(f"analysis/(main) paired bootstrap permutation_{short}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {short}...")
    if args.rates_only:
        res_rates = analyze_case_level_rates(annotated_json_path)
        print("\nCase-mean R_accept / R_argue (optional alternative to pooled):")
        print(res_rates)
        out_rates = out_dir / f"{short}_case_level_R_accept_R_argue_bootstrap_perm.csv"
        res_rates.to_csv(out_rates, index=False)
        print(f"\nSaved: {out_rates}")
    elif args.pooled_rates_only:
        res_pooled = analyze_pooled_rates_vs_C1(annotated_json_path)
        print("\nPooled R_accept / R_argue vs C1:")
        print(res_pooled)
        out_pooled = out_dir / f"{short}_pooled_R_accept_R_argue_bootstrap_perm.csv"
        res_pooled.to_csv(out_pooled, index=False)
        print(f"\nSaved: {out_pooled}")
        if args.case_mean_rates:
            res_rates = analyze_case_level_rates(annotated_json_path)
            print("\nCase-mean R_accept / R_argue:")
            print(res_rates)
            out_rates = out_dir / f"{short}_case_level_R_accept_R_argue_bootstrap_perm.csv"
            res_rates.to_csv(out_rates, index=False)
            print(f"\nSaved: {out_rates}")
    else:
        res = analyze_dataset_level(annotated_json_path)
        print("\nMedCOBE (dataset-level):")
        print(res)

        out_csv = out_dir / f"{short}_paired_bootstrap_permutation.csv"
        res.to_csv(out_csv, index=False)
        print(f"\nSaved: {out_csv}")

        res_pooled = analyze_pooled_rates_vs_C1(annotated_json_path)
        print("\nPooled R_accept / R_argue vs C1 (figure-style rates):")
        print(res_pooled)
        out_pooled = out_dir / f"{short}_pooled_R_accept_R_argue_bootstrap_perm.csv"
        res_pooled.to_csv(out_pooled, index=False)
        print(f"\nSaved: {out_pooled}")

        if args.case_mean_rates:
            res_rates = analyze_case_level_rates(annotated_json_path)
            print("\nCase-mean R_accept / R_argue:")
            print(res_rates)
            out_rates = out_dir / f"{short}_case_level_R_accept_R_argue_bootstrap_perm.csv"
            res_rates.to_csv(out_rates, index=False)
            print(f"\nSaved: {out_rates}")