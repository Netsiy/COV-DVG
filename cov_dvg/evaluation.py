"""Held-out test evaluation: baselines and paired statistics.

Baselines are compared at a *matched debate rate* (the same number of debates the
gate spends) so the comparison is about *which* items to debate, not how many.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .gate import ValueGate
from .utils import stable_seed


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def bootstrap_diff(a: np.ndarray, b: np.ndarray, iters: int, seed: int) -> Tuple[float, float, float]:
    """Paired mean difference a-b with a 95% percentile bootstrap CI."""
    rng = np.random.default_rng(seed)
    n = len(a)
    d = a.astype(float) - b.astype(float)
    diffs = np.empty(iters, dtype=float)
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        diffs[i] = float(np.mean(d[idx]))
    return float(np.mean(d)), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Two-sided exact McNemar p-value on discordant paired outcomes."""
    b01 = int(np.sum((a == 0) & (b == 1)))
    b10 = int(np.sum((a == 1) & (b == 0)))
    n = b01 + b10
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest
        return float(binomtest(min(b01, b10), n=n, p=0.5, alternative="two-sided").pvalue)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Matched-rate baselines
# ----------------------------------------------------------------------------
def matched_rate_decision(scores: np.ndarray, k: int, descending: bool = True) -> np.ndarray:
    n = len(scores)
    k = int(np.clip(k, 0, n))
    use = np.zeros(n, dtype=bool)
    if k == 0:
        return use
    order = np.argsort(scores)
    if descending:
        order = order[::-1]
    use[order[:k]] = True
    return use


def random_matched_accuracy(base, debate, k, seed, reps=2000) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(base)
    vals = []
    for _ in range(reps):
        use = np.zeros(n, dtype=bool)
        if k:
            use[rng.choice(n, size=k, replace=False)] = True
        vals.append(float(np.mean(np.where(use, debate, base))))
    return float(np.mean(vals)), float(np.std(vals))


# ----------------------------------------------------------------------------
# Per-benchmark evaluation
# ----------------------------------------------------------------------------
def evaluate_benchmark(
    scored: pd.DataFrame, gate: ValueGate, cfg: Config, threshold: Optional[float] = None
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    thr = gate.threshold if threshold is None else float(threshold)
    df = scored.copy().reset_index(drop=True)
    use = df["covdvg_utility"].to_numpy() > thr
    base = df["base_correct"].to_numpy(dtype=int)
    debate = df["debate_correct"].to_numpy(dtype=int)
    cov = np.where(use, debate, base).astype(int)
    n = len(df)
    k = int(use.sum())

    uncertainty_use = matched_rate_decision(df["feature_answer_entropy"].to_numpy(), k, True)
    lowconf_use = matched_rate_decision(df["feature_majority_confidence"].to_numpy(), k, False)
    uncertainty = np.where(uncertainty_use, debate, base).astype(int)
    lowconf = np.where(lowconf_use, debate, base).astype(int)
    rand_mean, rand_std = random_matched_accuracy(
        base, debate, k, stable_seed(df.iloc[0]["benchmark"], cfg.seed)
    )
    oracle = np.maximum(base, debate)

    gc = cfg.gate
    diff_base, lo_base, hi_base = bootstrap_diff(cov, base, gc.bootstrap_iters, cfg.seed + 11)
    diff_deb, lo_deb, hi_deb = bootstrap_diff(cov, debate, gc.bootstrap_iters, cfg.seed + 29)

    df["covdvg_use_debate"] = use.astype(int)
    df["covdvg_correct"] = cov
    df["uncertainty_use_debate"] = uncertainty_use.astype(int)
    df["uncertainty_correct"] = uncertainty
    df["lowconf_use_debate"] = lowconf_use.astype(int)
    df["lowconf_correct"] = lowconf

    primary_tokens = df["base_tokens"].to_numpy() + use.astype(int) * df["debate_extra_tokens"].to_numpy()
    always_tokens = df["always_debate_total_tokens"].to_numpy()

    metrics = {
        "benchmark": str(df.iloc[0]["benchmark"]),
        "n": n,
        "threshold": thr,
        "majority_accuracy": float(np.mean(base)),
        "always_debate_accuracy": float(np.mean(debate)),
        "covdvg_accuracy": float(np.mean(cov)),
        "uncertainty_matched_accuracy": float(np.mean(uncertainty)),
        "lowconf_matched_accuracy": float(np.mean(lowconf)),
        "random_matched_accuracy_mean": rand_mean,
        "random_matched_accuracy_std": rand_std,
        "oracle_accuracy": float(np.mean(oracle)),
        "debate_rate": float(np.mean(use)),
        "avg_base_tokens": float(np.mean(df["base_tokens"])),
        "avg_always_debate_tokens": float(np.mean(always_tokens)),
        "avg_covdvg_tokens": float(np.mean(primary_tokens)),
        "token_savings_vs_always_pct": float(100.0 * (1.0 - np.mean(primary_tokens) / max(1e-9, np.mean(always_tokens)))),
        "covdvg_minus_majority": diff_base,
        "covdvg_minus_majority_ci_low": lo_base,
        "covdvg_minus_majority_ci_high": hi_base,
        "covdvg_minus_always": diff_deb,
        "covdvg_minus_always_ci_low": lo_deb,
        "covdvg_minus_always_ci_high": hi_deb,
        "mcnemar_p_vs_majority": mcnemar_exact(base, cov),
        "mcnemar_p_vs_always": mcnemar_exact(debate, cov),
        "corrections_selected": int(np.sum(use & (df["delta"].to_numpy() == 1))),
        "subversions_selected": int(np.sum(use & (df["delta"].to_numpy() == -1))),
        "available_corrections": int(np.sum(df["delta"].to_numpy() == 1)),
        "available_subversions": int(np.sum(df["delta"].to_numpy() == -1)),
    }
    return metrics, df


def validation_policy_curve(gate: ValueGate, val_scored: pd.DataFrame, test_scored: pd.DataFrame) -> pd.DataFrame:
    """Test accuracy vs. debate rate, with thresholds set from validation quantiles."""
    rows: List[Dict[str, Any]] = []
    val_u = val_scored["covdvg_utility"].to_numpy()
    for target_rate in np.linspace(0, 1, 11):
        if target_rate <= 0:
            t = float("inf")
        elif target_rate >= 1:
            t = float("-inf")
        else:
            t = float(np.quantile(val_u, 1.0 - target_rate))
        for bench, g in test_scored.groupby("benchmark"):
            use = g["covdvg_utility"].to_numpy() > t
            correct = np.where(use, g["debate_correct"].to_numpy(), g["base_correct"].to_numpy())
            tokens = g["base_tokens"].to_numpy() + use.astype(int) * g["debate_extra_tokens"].to_numpy()
            rows.append({
                "benchmark": bench,
                "validation_target_rate": float(target_rate),
                "threshold": t,
                "test_realized_rate": float(np.mean(use)),
                "test_accuracy": float(np.mean(correct)),
                "avg_tokens": float(np.mean(tokens)),
            })
    return pd.DataFrame(rows)
