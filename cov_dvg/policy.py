"""Operating-point selection and policy metrics.

The gate produces a per-item ``covdvg_utility``. A single scalar threshold turns
that utility into a debate/no-debate decision. This module tunes that threshold
on validation and computes the policy-level metrics used both for in-terminal
monitoring during training and for the final evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .config import GateConfig


def tune_threshold(scored_val: pd.DataFrame, cfg: GateConfig, max_rate: Optional[float] = None) -> Dict[str, Any]:
    """Pick the utility threshold that maximises a token-penalised validation
    accuracy, subject to a maximum debate rate.

    ``max_rate`` defaults to ``cfg.max_debate_rate``. Pass a larger value (e.g.
    1.0) to let the token-penalised objective decide the rate freely — useful for
    per-benchmark tuning where debating almost everything can be optimal.

    Returns the chosen threshold and the validation metrics at that point.
    """
    if max_rate is None:
        max_rate = cfg.max_debate_rate
    s = scored_val
    utilities = np.sort(s["covdvg_utility"].unique())
    candidates = [float("inf"), float("-inf")]
    if len(utilities):
        candidates.extend(utilities.tolist())
        if len(utilities) > 1:
            candidates.extend(((utilities[:-1] + utilities[1:]) / 2.0).tolist())

    base = s["base_correct"].to_numpy()
    debate = s["debate_correct"].to_numpy()
    extra = s["debate_extra_tokens"].to_numpy()
    util = s["covdvg_utility"].to_numpy()

    best = None
    for t in candidates:
        use = util > t
        rate = float(np.mean(use))
        if rate > max_rate + 1e-12:
            continue
        chosen = np.where(use, debate, base)
        acc = float(np.mean(chosen))
        avg_extra = float(np.mean(use * extra))
        objective = acc - cfg.token_penalty_per_1k * (avg_extra / 1000.0)
        candidate = (objective, acc, -avg_extra, -rate, t)
        if best is None or candidate > best[0]:
            best = (candidate, {
                "threshold": float(t),
                "validation_accuracy": acc,
                "validation_debate_rate": rate,
                "validation_avg_extra_tokens": avg_extra,
                "validation_objective": objective,
            })
    if best is None:
        raise RuntimeError("Could not select a validation threshold.")
    return best[1]


def tune_thresholds_per_benchmark(
    val_scored: pd.DataFrame, cfg: GateConfig, global_threshold: float
) -> Dict[str, Any]:
    """Tune one threshold per benchmark, with a global fallback.

    A benchmark with at least ``cfg.per_benchmark_min_val`` validation episodes
    gets its own threshold (tuned with the relaxed
    ``cfg.per_benchmark_max_debate_rate`` cap); a benchmark with too few
    validation items falls back to ``global_threshold``. This lets the gate
    debate almost everything where debate helps broadly (low base accuracy) while
    staying selective where debate should be rare (high base accuracy).

    Returns ``{"thresholds": {benchmark: float}, "details": {...}}``.
    """
    thresholds: Dict[str, float] = {}
    details: Dict[str, Any] = {}
    for bench, g in val_scored.groupby("benchmark"):
        if len(g) < cfg.per_benchmark_min_val:
            thresholds[bench] = float(global_threshold)
            details[bench] = {"source": "global_fallback", "n_val": int(len(g)),
                              "threshold": float(global_threshold)}
            continue
        info = tune_threshold(g, cfg, max_rate=cfg.per_benchmark_max_debate_rate)
        thresholds[bench] = float(info["threshold"])
        details[bench] = {"source": "per_benchmark", "n_val": int(len(g)), **info}
    return {"thresholds": thresholds, "details": details, "global_threshold": float(global_threshold)}


def policy_metrics(scored: pd.DataFrame, threshold: float, cfg: GateConfig) -> Dict[str, float]:
    """Policy-level accuracy / cost summary at a fixed threshold.

    Used for periodic in-terminal validation monitoring: it tells us whether the
    gate's learned policy actually beats always-majority and how it compares to
    the always-debate and oracle references.
    """
    use = scored["covdvg_utility"].to_numpy() > threshold
    base = scored["base_correct"].to_numpy(dtype=int)
    debate = scored["debate_correct"].to_numpy(dtype=int)
    cov = np.where(use, debate, base)
    oracle = np.maximum(base, debate)

    base_tokens = scored["base_tokens"].to_numpy()
    extra = scored["debate_extra_tokens"].to_numpy()
    always_tokens = scored["always_debate_total_tokens"].to_numpy()
    gate_tokens = base_tokens + use.astype(int) * extra

    return {
        "gate_accuracy": float(np.mean(cov)),
        "majority_accuracy": float(np.mean(base)),
        "always_debate_accuracy": float(np.mean(debate)),
        "oracle_accuracy": float(np.mean(oracle)),
        "debate_rate": float(np.mean(use)),
        "avg_gate_tokens": float(np.mean(gate_tokens)),
        "token_savings_vs_always_pct": float(100.0 * (1.0 - np.mean(gate_tokens) / max(1e-9, np.mean(always_tokens)))),
        "gate_minus_majority": float(np.mean(cov) - np.mean(base)),
        "gate_minus_always": float(np.mean(cov) - np.mean(debate)),
    }
