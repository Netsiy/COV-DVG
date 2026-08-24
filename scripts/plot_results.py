#!/usr/bin/env python
"""Plot training history and policy curves.

Produces (whatever inputs are available):
    * training_curves.png  -- loss components, learning rate, train vs validation
      accuracy, and the gate-minus-majority / debate-rate trajectories.
    * policy_curves.png     -- test accuracy vs realized debate rate, and vs token
      cost, per benchmark, with the chosen operating point starred.

Runs standalone (only matplotlib + pandas). Paths default to the standard output
layout but can be overridden, e.g. when the CSVs were synced to ``results/``.

Examples
--------
    python scripts/plot_results.py
    python scripts/plot_results.py --history outputs/gate/training_history.csv \
        --policy-curve results/policy_curve.csv --summary results/results_summary.csv \
        --outdir results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _first_existing(*candidates: Path) -> Path | None:
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    return None


def plot_training(history_path: Path, out_path: Path) -> None:
    h = pd.read_csv(history_path)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("COV-DVG gate training", fontsize=14, fontweight="bold")

    # (0,0) Loss components.
    ax[0, 0].plot(h["epoch"], h["loss"], label="total", color="#1f77b4")
    ax[0, 0].plot(h["epoch"], h["ce"], label="outcome CE", color="#ff7f0e")
    ax[0, 0].plot(h["epoch"], h["cost"], label="cost (SmoothL1)", color="#2ca02c")
    ax[0, 0].set_title("Training loss"); ax[0, 0].set_xlabel("epoch"); ax[0, 0].set_ylabel("loss")
    ax[0, 0].legend(); ax[0, 0].grid(alpha=0.3)

    # (0,1) Learning rate (scheduler).
    ax[0, 1].plot(h["epoch"], h["lr"], color="#d62728")
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_title("Learning rate (ReduceLROnPlateau)")
    ax[0, 1].set_xlabel("epoch"); ax[0, 1].set_ylabel("lr (log)"); ax[0, 1].grid(alpha=0.3)

    # Validation rows (only eval epochs carry val_* columns).
    ev = h.dropna(subset=[c for c in h.columns if c.startswith("val_gate_accuracy")]) \
        if "val_gate_accuracy" in h.columns else pd.DataFrame()

    # (1,0) Accuracy: train (all epochs) + validation references (eval epochs).
    ax[1, 0].plot(h["epoch"], h["train_outcome_acc"], label="train outcome acc",
                  color="#7f7f7f", alpha=0.7)
    if len(ev):
        ax[1, 0].plot(ev["epoch"], ev["val_gate_accuracy"], "o-", label="val gate acc", color="#1f77b4")
        ax[1, 0].plot(ev["epoch"], ev["val_majority_accuracy"], "s--", label="val majority", color="#ff7f0e")
        if "val_always_debate_accuracy" in ev.columns:
            ax[1, 0].plot(ev["epoch"], ev["val_always_debate_accuracy"], "^--", label="val always-debate", color="#2ca02c")
        if "val_oracle_accuracy" in ev.columns:
            ax[1, 0].plot(ev["epoch"], ev["val_oracle_accuracy"], ":", label="val oracle", color="#9467bd")
    ax[1, 0].set_title("Accuracy: train vs validation policy")
    ax[1, 0].set_xlabel("epoch"); ax[1, 0].set_ylabel("accuracy"); ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.3)

    # (1,1) Gate value-add and debate rate.
    if len(ev):
        ax[1, 1].plot(ev["epoch"], ev["val_gate_minus_majority"], "o-", label="val gate − majority", color="#1f77b4")
        ax[1, 1].axhline(0.0, color="k", lw=0.8, alpha=0.5)
        ax[1, 1].plot(ev["epoch"], ev["val_debate_rate"], "s--", label="val debate rate", color="#8c564b")
        best = ev.loc[ev["val_val_objective"].idxmax()] if "val_val_objective" in ev.columns else None
        if best is not None:
            ax[1, 1].axvline(best["epoch"], color="green", ls=":", alpha=0.7,
                             label=f"best epoch {int(best['epoch'])}")
    ax[1, 1].set_title("Validation gate value-add & debate rate")
    ax[1, 1].set_xlabel("epoch"); ax[1, 1].set_ylabel("value"); ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_policy(policy_path: Path, summary_path: Path | None, out_path: Path) -> None:
    pc = pd.read_csv(policy_path)
    summary = pd.read_csv(summary_path) if summary_path and Path(summary_path).exists() else None
    benches = sorted(pc["benchmark"].unique())
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("COV-DVG policy curves (test)", fontsize=14, fontweight="bold")

    for i, bench in enumerate(benches):
        g = pc[pc["benchmark"] == bench].sort_values("test_realized_rate")
        c = colors[i % len(colors)]
        # Accuracy vs debate rate.
        ax[0].plot(g["test_realized_rate"], g["test_accuracy"], "-o", ms=3, color=c, label=bench)
        # Accuracy vs token cost.
        ax[1].plot(g["avg_tokens"], g["test_accuracy"], "-o", ms=3, color=c, label=bench)
        # Chosen operating point from the summary (starred).
        if summary is not None and bench in set(summary["benchmark"]):
            row = summary[summary["benchmark"] == bench].iloc[0]
            ax[0].plot(row["debate_rate"], row["covdvg_accuracy"], "*", ms=16,
                       color=c, markeredgecolor="k", zorder=5)
            if "avg_covdvg_tokens" in row:
                ax[1].plot(row["avg_covdvg_tokens"], row["covdvg_accuracy"], "*", ms=16,
                           color=c, markeredgecolor="k", zorder=5)

    ax[0].set_title("Accuracy vs debate rate  (★ = chosen operating point)")
    ax[0].set_xlabel("debate rate"); ax[0].set_ylabel("test accuracy")
    ax[0].legend(title="benchmark"); ax[0].grid(alpha=0.3)

    ax[1].set_title("Accuracy vs token cost")
    ax[1].set_xlabel("avg tokens per item"); ax[1].set_ylabel("test accuracy")
    ax[1].legend(title="benchmark"); ax[1].grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Plot COV-DVG training and policy curves.")
    ap.add_argument("--history", type=str, default=None)
    ap.add_argument("--policy-curve", type=str, default=None)
    ap.add_argument("--summary", type=str, default=None)
    ap.add_argument("--outdir", type=str, default=None)
    args = ap.parse_args()

    history = _first_existing(
        Path(args.history) if args.history else None,
        root / "outputs" / "gate" / "training_history.csv",
        root / "gate" / "training_history.csv",
    )
    policy = _first_existing(
        Path(args.policy_curve) if args.policy_curve else None,
        root / "outputs" / "results" / "policy_curve.csv",
        root / "results" / "policy_curve.csv",
    )
    summary = _first_existing(
        Path(args.summary) if args.summary else None,
        root / "outputs" / "results" / "results_summary.csv",
        root / "results" / "results_summary.csv",
    )
    outdir = Path(args.outdir) if args.outdir else (
        policy.parent if policy else root / "results"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    did = False
    if history:
        plot_training(history, outdir / "training_curves.png"); did = True
    else:
        print("No training_history.csv found — skipping training plot.")
    if policy:
        plot_policy(policy, summary, outdir / "policy_curves.png"); did = True
    else:
        print("No policy_curve.csv found — skipping policy plot.")
    if not did:
        raise SystemExit("Nothing to plot: provide --history and/or --policy-curve.")
    print(f"\nPlots saved under: {outdir}")


if __name__ == "__main__":
    main()
