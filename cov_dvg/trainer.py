"""Training loop for the neural value gate.

Design goals requested for this reimplementation:
    * A live tqdm progress bar per epoch.
    * A one-line summary printed after every epoch.
    * In-terminal validation evaluation every ``eval_every`` epochs (default 10),
      so genuine learning is visible without a separate validation run.
    * An LR scheduler (ReduceLROnPlateau on the validation objective) whose state
      is reported at each evaluation, to monitor progress.
    * Best-checkpoint tracking on the validation objective.

The gate is small and the episode tables are modest, so training ~120 epochs is
fast; the point of the epoch loop is to *observe* whether the gate is learning a
policy that beats the always-majority and always-debate references.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import Config
from .gate import CLASS_TO_INDEX, FEATURE_COLS, OUTCOME_CLASSES, ValueGate
from .policy import policy_metrics, tune_threshold
from .utils import get_logger, seed_everything

logger = get_logger("covdvg.train")


def _class_weights(deltas: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights over the fixed 3-class outcome space."""
    counts = np.ones(len(OUTCOME_CLASSES), dtype=float)
    for d in deltas:
        counts[CLASS_TO_INDEX[int(d)]] += 1.0
    total = counts.sum()
    w = total / (len(OUTCOME_CLASSES) * counts)
    return w / w.mean()


def _outcome_indices(deltas: np.ndarray) -> np.ndarray:
    return np.asarray([CLASS_TO_INDEX[int(d)] for d in deltas], dtype=np.int64)


class GateTrainer:
    def __init__(self, cfg: Config, device: str = "cuda"):
        self.cfg = cfg
        self.gcfg = cfg.gate
        self.device = device
        self.gate = ValueGate(self.gcfg)
        self.history: List[Dict[str, Any]] = []

    # -- tensor preparation --------------------------------------------------
    def _prepare(self, train: pd.DataFrame):
        import torch

        Xtr = train[FEATURE_COLS].to_numpy(dtype=float)
        self.gate.scaler.fit(Xtr)
        Xtr = self.gate.scaler.transform(Xtr)
        ytr = _outcome_indices(train["delta"].to_numpy())
        ctr = np.log1p(train["debate_extra_tokens"].to_numpy(dtype=float))

        Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
        ytr_t = torch.tensor(ytr, dtype=torch.long)
        ctr_t = torch.tensor(ctr, dtype=torch.float32)
        return Xtr_t, ytr_t, ctr_t

    # -- one training epoch --------------------------------------------------
    def _train_epoch(self, loader, optimizer, ce_loss, cost_loss) -> Dict[str, float]:
        import torch

        self.gate.net.train()
        total, n = 0.0, 0
        ce_tot, cost_tot = 0.0, 0.0
        correct = 0
        pbar = tqdm(loader, desc="  batches", leave=False, dynamic_ncols=True)
        for xb, yb, cb in pbar:
            xb = xb.to(self.device); yb = yb.to(self.device); cb = cb.to(self.device)
            optimizer.zero_grad()
            logits, cost = self.gate.net(xb)
            l_ce = ce_loss(logits, yb)
            l_cost = cost_loss(cost, cb)
            loss = l_ce + self.gcfg.cost_loss_weight * l_cost
            loss.backward()
            if self.gcfg.grad_clip and self.gcfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.gate.net.parameters(), self.gcfg.grad_clip)
            optimizer.step()

            bs = xb.size(0)
            total += float(loss.item()) * bs
            ce_tot += float(l_ce.item()) * bs
            cost_tot += float(l_cost.item()) * bs
            correct += int((logits.argmax(dim=-1) == yb).sum().item())
            n += bs
            pbar.set_postfix(loss=f"{total/max(1,n):.4f}")
        return {
            "loss": total / max(1, n),
            "ce": ce_tot / max(1, n),
            "cost": cost_tot / max(1, n),
            "train_outcome_acc": correct / max(1, n),
        }

    # -- validation monitoring ----------------------------------------------
    def _evaluate(self, val: pd.DataFrame) -> Dict[str, Any]:
        """Score validation, tune a temporary threshold, return policy + outcome metrics."""
        scored = self.gate.score_frame(val)
        info = tune_threshold(scored, self.gcfg)
        pol = policy_metrics(scored, info["threshold"], self.gcfg)

        # Outcome-head quality (macro-F1 over the 3 classes) and cost MAE.
        pred = self.gate.predict(val)
        prob = np.stack([pred["p_subversion"], pred["p_no_change"], pred["p_correction"]], axis=1)
        y_pred = prob.argmax(axis=1)
        y_true = _outcome_indices(val["delta"].to_numpy())
        macro_f1 = _macro_f1(y_true, y_pred, num_classes=len(OUTCOME_CLASSES))
        cost_mae = float(np.mean(np.abs(pred["pred_debate_tokens"] - val["debate_extra_tokens"].to_numpy(dtype=float))))

        return {
            "threshold": info["threshold"],
            "val_objective": info["validation_objective"],
            "macro_f1": macro_f1,
            "cost_mae": cost_mae,
            **pol,
        }

    # -- full training -------------------------------------------------------
    def train(self, train: pd.DataFrame, val: pd.DataFrame) -> ValueGate:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        seed_everything(self.gcfg.seed)
        self.gate.build(device=self.device)

        Xtr_t, ytr_t, ctr_t = self._prepare(train)
        dataset = TensorDataset(Xtr_t, ytr_t, ctr_t)
        batch_size = min(self.gcfg.batch_size, max(1, len(dataset)))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        weights = torch.tensor(_class_weights(train["delta"].to_numpy()), dtype=torch.float32, device=self.device)
        ce_loss = torch.nn.CrossEntropyLoss(weight=weights)
        cost_loss = torch.nn.SmoothL1Loss()

        optimizer = torch.optim.AdamW(
            self.gate.net.parameters(), lr=self.gcfg.lr, weight_decay=self.gcfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=self.gcfg.scheduler_factor,
            patience=self.gcfg.scheduler_patience, min_lr=self.gcfg.scheduler_min_lr,
        )

        self._print_train_header(train, val, batch_size)

        best_obj = -np.inf
        best_state = copy.deepcopy(self.gate.net.state_dict())
        best_epoch = 0

        for epoch in range(1, self.gcfg.epochs + 1):
            tr = self._train_epoch(loader, optimizer, ce_loss, cost_loss)
            lr = optimizer.param_groups[0]["lr"]
            record: Dict[str, Any] = {"epoch": epoch, "lr": lr, **tr}

            # Per-epoch one-line summary.
            print(
                f"Epoch {epoch:3d}/{self.gcfg.epochs} | "
                f"loss {tr['loss']:.4f} (ce {tr['ce']:.4f}, cost {tr['cost']:.4f}) | "
                f"train_acc {tr['train_outcome_acc']:.3f} | lr {lr:.2e}"
            )

            # Periodic in-terminal validation evaluation.
            is_eval = (epoch % self.gcfg.eval_every == 0) or (epoch == self.gcfg.epochs)
            if is_eval:
                ev = self._evaluate(val)
                record.update({f"val_{k}": v for k, v in ev.items()})
                self._print_eval_block(epoch, ev)
                scheduler.step(ev["val_objective"])
                if ev["val_objective"] > best_obj + 1e-9:
                    best_obj = ev["val_objective"]
                    best_state = copy.deepcopy(self.gate.net.state_dict())
                    best_epoch = epoch

            self.history.append(record)

        # Restore the best-validation checkpoint and fix the final threshold on val.
        self.gate.net.load_state_dict(best_state)
        final = tune_threshold(self.gate.score_frame(val), self.gcfg)
        self.gate.threshold = final["threshold"]
        self.gate.meta = {
            "best_epoch": best_epoch,
            "best_val_objective": float(best_obj),
            "epochs": self.gcfg.epochs,
            "n_train": int(len(train)),
            "n_val": int(len(val)),
            "final_threshold_selection": final,
            "train_outcome_counts": {int(c): int((train["delta"] == c).sum()) for c in OUTCOME_CLASSES},
        }
        print("\n" + "=" * 78)
        print(f"Best validation objective {best_obj:.4f} at epoch {best_epoch}. "
              f"Final threshold = {self.gate.threshold:.4f}")
        print("=" * 78)
        return self.gate

    def save_history(self, path: Path) -> None:
        pd.DataFrame(self.history).to_csv(path, index=False)

    # -- pretty printing -----------------------------------------------------
    def _print_train_header(self, train, val, batch_size) -> None:
        counts = {c: int((train["delta"] == c).sum()) for c in OUTCOME_CLASSES}
        print("\n" + "=" * 78)
        print("TRAINING THE COV-DVG VALUE GATE")
        print("=" * 78)
        print(f"Device: {self.device} | epochs: {self.gcfg.epochs} | batch size: {batch_size} | "
              f"eval every: {self.gcfg.eval_every}")
        print(f"Train episodes: {len(train)} | Val episodes: {len(val)} | Features: {len(FEATURE_COLS)}")
        print(f"Train outcome counts  subversion(-1)={counts[-1]}  no_change(0)={counts[0]}  correction(+1)={counts[1]}")
        print(f"Hidden dims: {self.gcfg.hidden_dims} | dropout: {self.gcfg.dropout} | lr: {self.gcfg.lr}")
        print("=" * 78 + "\n")

    def _print_eval_block(self, epoch: int, ev: Dict[str, Any]) -> None:
        print("   " + "-" * 72)
        print(f"   [VALIDATION @ epoch {epoch}]  threshold={ev['threshold']:.4f}  "
              f"objective={ev['val_objective']:.4f}")
        print(f"     outcome macro-F1 : {ev['macro_f1']:.3f}     cost MAE (tokens): {ev['cost_mae']:.1f}")
        print(f"     gate accuracy    : {ev['gate_accuracy']:.3f}"
              f"   (majority {ev['majority_accuracy']:.3f}, "
              f"always-debate {ev['always_debate_accuracy']:.3f}, oracle {ev['oracle_accuracy']:.3f})")
        print(f"     gate - majority  : {ev['gate_minus_majority']:+.3f}"
              f"   gate - always: {ev['gate_minus_always']:+.3f}")
        print(f"     debate rate      : {ev['debate_rate']:.3f}"
              f"   token savings vs always: {ev['token_savings_vs_always_pct']:.1f}%")
        print("   " + "-" * 72)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        if tp + fp == 0 or tp + fn == 0:
            # Class absent from predictions or truth: define F1=0 unless it is
            # entirely absent from both (then it does not contribute).
            if tp + fp + fn == 0:
                continue
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else 0.0
