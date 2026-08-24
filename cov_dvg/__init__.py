"""COV-DVG: Cost-Of-Verification Debate Value Gate.

A clean, modular reimplementation of the multi-agent debate value-gate project.

Pipeline stages
---------------
1. Episode generation  (scripts/generate_episodes.py)
   Real LLM agents + critic/adjudicator debate -> cached episode tables.
2. Gate training       (scripts/train_gate.py)
   A trainable PyTorch value gate, 120 epochs, tqdm progress + periodic
   in-terminal validation evaluation.
3. Evaluation          (scripts/evaluate.py)
   Held-out test accuracy with matched-rate baselines and paired statistics.

Scientific invariants preserved from the original design
--------------------------------------------------------
* Agents solve independently; the critic/adjudicator never see ground truth.
* The gate sees ONLY pre-debate observable features (no correctness leakage).
* Labels are observed correction / no-change / subversion outcomes.
* Threshold and gate parameters are fit on non-test episodes only.
"""

from .config import Config, DataConfig, LLMConfig, GateConfig

__all__ = ["Config", "DataConfig", "LLMConfig", "GateConfig"]
__version__ = "3.0.0"
