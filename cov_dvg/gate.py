"""The COV-DVG value gate: a trainable dual-head MLP.

Given the pre-debate features of a problem, the gate predicts:
    * outcome head  -> P(subversion), P(no_change), P(correction)   (3-way softmax)
    * cost head     -> predicted debate cost as log1p(extra tokens)  (regression)

At decision time the two heads combine into a utility:

    utility = P(correction) - P(subversion) - token_penalty_per_1k * (cost / 1000)

The gate debates a problem when ``utility > threshold``. The threshold is tuned
on validation subject to a maximum debate rate.

This module holds three things:
    * :class:`GateNet`   -- the raw ``nn.Module``.
    * :class:`FeatureScaler` -- a small standardiser saved alongside the weights.
    * :class:`ValueGate` -- a convenience wrapper (build / save / load / score).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import GateConfig
from .features import FEATURE_NAMES

# Outcome class order is fixed so probability columns are unambiguous.
OUTCOME_CLASSES = [-1, 0, 1]            # subversion, no_change, correction
CLASS_TO_INDEX = {c: i for i, c in enumerate(OUTCOME_CLASSES)}
FEATURE_COLS = [f"feature_{n}" for n in FEATURE_NAMES]


# ----------------------------------------------------------------------------
# Feature standardisation
# ----------------------------------------------------------------------------
class FeatureScaler:
    """Z-score standardiser fit on the training features only."""

    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self.mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-8] = 1.0
        self.std = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return (X - self.mean) / self.std

    def to_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureScaler":
        s = cls()
        s.mean = np.asarray(d["mean"], dtype=float)
        s.std = np.asarray(d["std"], dtype=float)
        return s


# ----------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------
def build_net(input_dim: int, hidden_dims: List[int], dropout: float):
    import torch.nn as nn

    class GateNet(nn.Module):
        def __init__(self):
            super().__init__()
            layers: List[nn.Module] = []
            prev = input_dim
            for h in hidden_dims:
                layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            self.trunk = nn.Sequential(*layers)
            self.outcome_head = nn.Linear(prev, len(OUTCOME_CLASSES))
            self.cost_head = nn.Linear(prev, 1)

        def forward(self, x):
            z = self.trunk(x)
            return self.outcome_head(z), self.cost_head(z).squeeze(-1)

    return GateNet()


# ----------------------------------------------------------------------------
# High-level wrapper
# ----------------------------------------------------------------------------
class ValueGate:
    """Wraps the network, the scaler, the tuned threshold and metadata."""

    def __init__(self, cfg: GateConfig):
        self.cfg = cfg
        self.scaler = FeatureScaler()
        self.net = None
        self.threshold: float = 0.0
        self.feature_cols: List[str] = list(FEATURE_COLS)
        self.meta: Dict[str, Any] = {}
        self._device = "cpu"

    # -- construction --------------------------------------------------------
    def build(self, device: str = "cpu") -> "ValueGate":
        self.net = build_net(len(self.feature_cols), self.cfg.hidden_dims, self.cfg.dropout)
        self._device = device
        self.net.to(device)
        return self

    def to(self, device: str) -> "ValueGate":
        self._device = device
        if self.net is not None:
            self.net.to(device)
        return self

    # -- inference -----------------------------------------------------------
    def _features_matrix(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Episode frame is missing feature columns: {missing[:5]} ...")
        return df[self.feature_cols].to_numpy(dtype=float)

    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Return outcome probabilities and predicted debate cost (in tokens)."""
        import torch

        X = self.scaler.transform(self._features_matrix(df))
        self.net.eval()
        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32, device=self._device)
            logits, cost = self.net(xb)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            cost = cost.cpu().numpy()
        pred_tokens = np.maximum(1.0, np.expm1(cost))
        return {
            "p_subversion": probs[:, CLASS_TO_INDEX[-1]],
            "p_no_change": probs[:, CLASS_TO_INDEX[0]],
            "p_correction": probs[:, CLASS_TO_INDEX[1]],
            "pred_debate_tokens": pred_tokens,
        }

    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        pred = self.predict(df)
        out = df.copy()
        for k, v in pred.items():
            out[k] = v
        out["covdvg_utility"] = (
            pred["p_correction"]
            - pred["p_subversion"]
            - self.cfg.token_penalty_per_1k * (pred["pred_debate_tokens"] / 1000.0)
        )
        return out

    # -- persistence ---------------------------------------------------------
    def save(self, directory: Path) -> None:
        import torch

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), directory / "gate_net.pt")
        payload = {
            "cfg": {
                "hidden_dims": self.cfg.hidden_dims,
                "dropout": self.cfg.dropout,
                "token_penalty_per_1k": self.cfg.token_penalty_per_1k,
                "max_debate_rate": self.cfg.max_debate_rate,
            },
            "scaler": self.scaler.to_dict(),
            "threshold": float(self.threshold),
            "feature_cols": self.feature_cols,
            "outcome_classes": OUTCOME_CLASSES,
            "meta": self.meta,
        }
        with (directory / "gate_meta.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: Path, cfg: GateConfig, device: str = "cpu") -> "ValueGate":
        import torch

        directory = Path(directory)
        with (directory / "gate_meta.json").open("r", encoding="utf-8") as f:
            payload = json.load(f)
        # Respect the architecture that was actually trained.
        cfg.hidden_dims = payload["cfg"]["hidden_dims"]
        cfg.dropout = payload["cfg"]["dropout"]
        cfg.token_penalty_per_1k = payload["cfg"]["token_penalty_per_1k"]
        cfg.max_debate_rate = payload["cfg"]["max_debate_rate"]

        gate = cls(cfg)
        gate.feature_cols = payload["feature_cols"]
        gate.scaler = FeatureScaler.from_dict(payload["scaler"])
        gate.threshold = float(payload["threshold"])
        gate.meta = payload.get("meta", {})
        gate.build(device=device)
        state = torch.load(directory / "gate_net.pt", map_location=device)
        gate.net.load_state_dict(state)
        gate.net.eval()
        return gate
