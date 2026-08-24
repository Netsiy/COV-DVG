#!/usr/bin/env python
"""Stage 3 — evaluate the trained gate on the held-out test episodes.

Loads the trained gate and the cached test episodes, scores them, and reports
per-benchmark accuracy against matched-rate baselines (uncertainty, low-confidence,
random) plus the always-majority / always-debate / oracle references, with paired
bootstrap CIs and exact McNemar p-values. Also writes a validation-thresholded
policy curve (test accuracy vs. debate rate).

Example
-------
    python scripts/evaluate.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cov_dvg.config import Config
from cov_dvg.evaluation import evaluate_benchmark, validation_policy_curve
from cov_dvg.gate import ValueGate
from cov_dvg.policy import tune_thresholds_per_benchmark
from cov_dvg.utils import get_logger

logger = get_logger("covdvg.evaluate")


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the COV-DVG value gate.")
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--threshold-mode", type=str, default="per_benchmark",
                    choices=["per_benchmark", "global"],
                    help="per_benchmark tunes one threshold per benchmark on validation "
                         "(global fallback for tiny val sets); global uses the single "
                         "trained threshold everywhere.")
    args = ap.parse_args()

    cfg = Config()
    if args.output_dir:
        cfg.output_dir = args.output_dir
    cfg.__post_init__()

    val_path = cfg.episodes_dir / "all_val.csv"
    test_path = cfg.episodes_dir / "all_test.csv"
    for p in (val_path, test_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run scripts/generate_episodes.py first.")

    device = pick_device(args.device)
    gate = ValueGate.load(cfg.gate_dir, cfg.gate, device=device)
    logger.info("Loaded gate (threshold=%.4f, best_epoch=%s)",
                gate.threshold, gate.meta.get("best_epoch"))

    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)
    val_scored = gate.score_frame(val)
    test_scored = gate.score_frame(test)
    val_scored.to_csv(cfg.results_dir / "validation_scored.csv", index=False)

    # Threshold selection: per-benchmark (with global fallback) or single global.
    if args.threshold_mode == "per_benchmark":
        pb = tune_thresholds_per_benchmark(val_scored, cfg.gate, gate.threshold)
        thresholds = pb["thresholds"]
        with (cfg.results_dir / "thresholds.json").open("w", encoding="utf-8") as f:
            json.dump(pb, f, ensure_ascii=False, indent=2, default=str)
        logger.info("Per-benchmark thresholds: %s",
                    {k: round(v, 4) for k, v in thresholds.items()})
    else:
        thresholds = {}
        logger.info("Using global threshold everywhere: %.4f", gate.threshold)

    all_metrics, details = [], []
    for bench, g in test_scored.groupby("benchmark", sort=False):
        thr = thresholds.get(bench, gate.threshold)
        metrics, detail = evaluate_benchmark(g, gate, cfg, threshold=thr)
        all_metrics.append(metrics)
        details.append(detail)

    summary = pd.DataFrame(all_metrics)
    detail_df = pd.concat(details, ignore_index=True)
    summary.to_csv(cfg.results_dir / "results_summary.csv", index=False)
    detail_df.to_csv(cfg.results_dir / "test_details.csv", index=False)

    curve = validation_policy_curve(gate, val_scored, test_scored)
    curve.to_csv(cfg.results_dir / "policy_curve.csv", index=False)

    print("\n" + "=" * 100)
    print(f"PRIMARY TEST RESULTS  (threshold mode: {args.threshold_mode})")
    print("=" * 100)
    cols = [
        "benchmark", "n", "threshold", "majority_accuracy", "always_debate_accuracy",
        "covdvg_accuracy", "uncertainty_matched_accuracy", "random_matched_accuracy_mean",
        "oracle_accuracy", "debate_rate", "token_savings_vs_always_pct",
        "covdvg_minus_majority", "covdvg_minus_majority_ci_low", "covdvg_minus_majority_ci_high",
        "mcnemar_p_vs_majority",
        "covdvg_minus_always", "covdvg_minus_always_ci_low", "covdvg_minus_always_ci_high",
        "mcnemar_p_vs_always",
    ]
    print(summary[cols].to_string(index=False))

    print("\nGate metadata:")
    print(json.dumps(gate.meta, indent=2, default=str))

    print("\nInterpretation guardrails:")
    print("* GSM8K/MATH test rows come from their official test splits.")
    print("* MMLU-Pro test rows come from its official test split; its 70-row validation")
    print("  split trains/tunes the gate portion.")
    print("* GPQA Diamond has no separate public labeled test split; its reported test is a")
    print("  deterministic held-out partition of the 198 labeled Diamond items.")
    print("* Claim a benefit only when the paired CI/p-value and matched-rate baselines agree.")
    print(f"\nAll artifacts saved under: {cfg.results_dir}")


if __name__ == "__main__":
    main()
