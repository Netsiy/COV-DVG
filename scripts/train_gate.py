#!/usr/bin/env python
"""Stage 2 — train the neural value gate.

Loads the cached train/val episode tables produced by ``generate_episodes.py``
and trains the dual-head gate for ``--epochs`` epochs (default 120) with a live
progress bar, a per-epoch summary, and an in-terminal validation evaluation every
``--eval-every`` epochs (default 10). The best-validation checkpoint, the fitted
scaler, the tuned threshold and the training history are saved under
``outputs/gate``.

Examples
--------
    python scripts/train_gate.py
    python scripts/train_gate.py --epochs 120 --eval-every 10 --lr 1e-3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cov_dvg.config import Config
from cov_dvg.trainer import GateTrainer
from cov_dvg.utils import get_logger

logger = get_logger("covdvg.train")


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the COV-DVG value gate.")
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.20)
    ap.add_argument("--hidden-dims", nargs="+", type=int, default=[128, 64])
    ap.add_argument("--cost-loss-weight", type=float, default=0.30)
    ap.add_argument("--token-penalty", type=float, default=0.001)
    ap.add_argument("--max-debate-rate", type=float, default=0.50)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = Config()
    if args.output_dir:
        cfg.output_dir = args.output_dir
    g = cfg.gate
    g.epochs = args.epochs
    g.eval_every = args.eval_every
    g.batch_size = args.batch_size
    g.lr = args.lr
    g.weight_decay = args.weight_decay
    g.dropout = args.dropout
    g.hidden_dims = args.hidden_dims
    g.cost_loss_weight = args.cost_loss_weight
    g.token_penalty_per_1k = args.token_penalty
    g.max_debate_rate = args.max_debate_rate
    cfg.seed = args.seed
    cfg.__post_init__()

    train_path = cfg.episodes_dir / "all_train.csv"
    val_path = cfg.episodes_dir / "all_val.csv"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Missing episode tables. Expected {train_path} and {val_path}. "
            "Run scripts/generate_episodes.py first."
        )
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    if len(train) == 0 or len(val) == 0:
        raise RuntimeError("Train or validation episode table is empty.")

    device = pick_device(args.device)
    trainer = GateTrainer(cfg, device=device)
    gate = trainer.train(train, val)

    gate.save(cfg.gate_dir)
    trainer.save_history(cfg.gate_dir / "training_history.csv")
    logger.info("Saved gate to %s", cfg.gate_dir)
    print(f"\nGate saved to: {cfg.gate_dir}")
    print("Next: python scripts/evaluate.py")


if __name__ == "__main__":
    main()
