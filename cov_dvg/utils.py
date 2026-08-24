"""Small shared utilities: hashing, seeding, text cleaning and logging."""

from __future__ import annotations

import os
import hashlib
import logging
import random
from typing import Any, Optional

import numpy as np


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
def get_logger(name: str = "covdvg") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# ----------------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------------
def stable_hash(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def stable_seed(text: str, base_seed: int = 42) -> int:
    h = hashlib.sha256(f"{base_seed}|{text}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2**31 - 1)


def seed_everything(seed: int) -> None:
    """Seed python, numpy and (if available) torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ----------------------------------------------------------------------------
# Text helpers
# ----------------------------------------------------------------------------
def clean_text(x: Any) -> str:
    return "" if x is None else str(x).strip()


def get_hf_token() -> Optional[str]:
    """Read an HF token from the environment (or a Kaggle secret if present)."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN").strip()
        if token:
            os.environ["HF_TOKEN"] = token
            return token
    except Exception:
        pass
    return None
