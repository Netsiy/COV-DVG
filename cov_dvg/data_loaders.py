"""Benchmark loading and official-split sampling.

Local datasets (read from ``config.data.data_dir``):
    * GSM8K     -> data/GSM8K/{train,test}.jsonl
    * MMLU-Pro  -> data/MMLU-Pro/{validation,test}-00000-of-00001.parquet
    * GPQA      -> data/GPQA/gpqa_diamond.csv

Remote dataset (Hugging Face hub):
    * MATH      -> DigitalLearningGmbH/MATH-lighteval  (default config)

Every benchmark is standardised to columns:
    sample_id, benchmark, task_type, split, question, answer
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .grading import extract_gsm8k_gold, extract_last_boxed
from .utils import clean_text, get_hf_token, get_logger, stable_hash, stable_seed

logger = get_logger("covdvg.data")


class BenchmarkAccessError(RuntimeError):
    pass


@dataclass
class BenchmarkBundle:
    name: str
    task_type: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


# Task-type mapping used by the grader / answer extractor.
TASK_TYPES = {
    "gsm8k": "gsm8k",
    "mmlu_pro": "mcq10",
    "math": "math",
    "gpqa": "mcq4",
}


class BenchmarkLoader:
    def __init__(self, config: Config):
        self.cfg = config
        self.data_dir = config.data.resolved_data_dir()
        self.token = get_hf_token()

    # -- sampling helpers ----------------------------------------------------
    def _seed(self, text: str) -> int:
        return stable_seed(text, self.cfg.data.seed)

    def _sample(self, df: pd.DataFrame, n: int, seed_text: str) -> pd.DataFrame:
        """Deterministic prefix sample (nested across budgets for cache reuse)."""
        shuffled = df.sample(frac=1.0, random_state=self._seed(seed_text)).reset_index(drop=True)
        if n <= 0 or len(shuffled) <= n:
            return shuffled.copy()
        return shuffled.iloc[:n].copy().reset_index(drop=True)

    def _fixed_train_val(
        self, df: pd.DataFrame, train_n: int, val_n: int, max_val_n: int, seed_text: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Permanently disjoint train / validation pools.

        The first ``max_val_n`` shuffled rows are reserved for validation forever;
        training is drawn from the remainder. A validation item can therefore
        never leak into training when the budget is increased later.
        """
        shuffled = df.sample(frac=1.0, random_state=self._seed(seed_text)).reset_index(drop=True)
        max_val_n = min(max_val_n, max(1, len(shuffled) - 1))
        val_pool = shuffled.iloc[:max_val_n].reset_index(drop=True)
        train_pool = shuffled.iloc[max_val_n:].reset_index(drop=True)
        tr = train_pool.iloc[:min(train_n, len(train_pool))].reset_index(drop=True)
        va = val_pool.iloc[:min(val_n, len(val_pool))].reset_index(drop=True)
        return tr, va

    @staticmethod
    def _standardize(df: pd.DataFrame, benchmark: str, task_type: str, split: str) -> pd.DataFrame:
        out = df.copy().reset_index(drop=True)
        out["benchmark"] = benchmark
        out["task_type"] = task_type
        out["split"] = split
        # sample_id excludes the split so cached generations remain reusable when
        # split membership or sample budgets change.
        out["sample_id"] = [f"{benchmark}:{stable_hash(q, 20)}" for q in out["question"].astype(str)]
        return out[["sample_id", "benchmark", "task_type", "split", "question", "answer"]]

    def _require(self, path: Path) -> Path:
        if not path.exists():
            raise BenchmarkAccessError(f"Expected local file not found: {path}")
        return path

    # -- individual benchmarks ----------------------------------------------
    def gsm8k(self) -> BenchmarkBundle:
        root = self.data_dir / "GSM8K"
        train_all = self._read_jsonl(self._require(root / "train.jsonl"))
        test_all = self._read_jsonl(self._require(root / "test.jsonl"))
        train_all = pd.DataFrame({
            "question": train_all["question"].astype(str),
            "answer": train_all["answer"].map(extract_gsm8k_gold),
        })
        test_all = pd.DataFrame({
            "question": test_all["question"].astype(str),
            "answer": test_all["answer"].map(extract_gsm8k_gold),
        })
        b = self.cfg.data.budget("gsm8k")
        tr, va = self._fixed_train_val(train_all, b["train"], b["val"], 60, "gsm8k-trainval")
        te = self._sample(test_all, b["test"], "gsm8k-test")
        return self._bundle("gsm8k", tr, va, te)

    def mmlu_pro(self) -> BenchmarkBundle:
        root = self.data_dir / "MMLU-Pro"
        val_raw = pd.read_parquet(self._require(root / "validation-00000-of-00001.parquet"))
        test_raw = pd.read_parquet(self._require(root / "test-00000-of-00001.parquet"))
        pool = self._convert_mmlu(val_raw)
        test_all = self._convert_mmlu(test_raw)
        b = self.cfg.data.budget("mmlu_pro")
        # The official validation split is only 70 rows: reserve at most 20.
        tr_target = min(b["train"], max(35, len(pool) - 20))
        va_target = min(b["val"], len(pool) - tr_target)
        tr, va = self._fixed_train_val(pool, tr_target, va_target, 20, "mmlu-pro-validation")
        te = self._sample(test_all, b["test"], "mmlu-pro-test")
        return self._bundle("mmlu_pro", tr, va, te)

    def math(self) -> BenchmarkBundle:
        train_ds = self._load_hf(split="train").to_pandas()
        test_ds = self._load_hf(split="test").to_pandas()
        train_all = pd.DataFrame({
            "question": train_ds["problem"].astype(str),
            "answer": train_ds["solution"].map(extract_last_boxed),
        })
        test_all = pd.DataFrame({
            "question": test_ds["problem"].astype(str),
            "answer": test_ds["solution"].map(extract_last_boxed),
        })
        b = self.cfg.data.budget("math")
        tr, va = self._fixed_train_val(train_all, b["train"], b["val"], 60, "math-trainval")
        te = self._sample(test_all, b["test"], "math-test")
        return self._bundle("math", tr, va, te)

    def gpqa(self) -> BenchmarkBundle:
        root = self.data_dir / "GPQA"
        raw = pd.read_csv(self._require(root / f"gpqa_{self.cfg.data.gpqa_split}.csv"))
        q_col = next(c for c in ["Question", "question"] if c in raw.columns)
        c_col = next(c for c in ["Correct Answer", "correct_answer"] if c in raw.columns)
        wrong_cols = [
            c for c in [
                "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3",
                "incorrect_answer_1", "incorrect_answer_2", "incorrect_answer_3",
            ] if c in raw.columns
        ][:3]

        rows = []
        for _, r in raw.iterrows():
            q0 = clean_text(r[q_col])
            correct = clean_text(r[c_col])
            options = [correct] + [clean_text(r[c]) for c in wrong_cols]
            rng = random.Random(self._seed(q0))
            rng.shuffle(options)
            gold = chr(65 + options.index(correct))
            question = q0 + "\nOptions:\n" + "\n".join(
                f"{chr(65+i)}) {opt}" for i, opt in enumerate(options)
            )
            rows.append({"question": question, "answer": gold})
        all_df = pd.DataFrame(rows)

        # GPQA Diamond has no public held-out test labels. Use one permanent
        # deterministic 100/30/68 development/validation/test partition, then take
        # nested prefixes so no item ever changes role between runs.
        shuffled = all_df.sample(
            frac=1.0, random_state=self._seed(f"gpqa-{self.cfg.data.gpqa_split}")
        ).reset_index(drop=True)
        # Permanent disjoint pools; a larger "main" split simply yields a larger
        # test pool. Prefixes are nested so an item never changes role.
        n = len(shuffled)
        train_pool = shuffled.iloc[:100].reset_index(drop=True)
        val_pool = shuffled.iloc[100:130].reset_index(drop=True)
        test_pool = shuffled.iloc[130:].reset_index(drop=True)
        b = self.cfg.data.budget("gpqa")
        tr = train_pool.iloc[:min(b["train"], len(train_pool))]
        va = val_pool.iloc[:min(b["val"], len(val_pool))]
        te = test_pool.iloc[:min(b["test"], len(test_pool))]
        return self._bundle("gpqa", tr, va, te)

    # -- shared plumbing -----------------------------------------------------
    @staticmethod
    def _read_jsonl(path: Path) -> pd.DataFrame:
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pd.DataFrame(records)

    @staticmethod
    def _convert_mmlu(df: pd.DataFrame) -> pd.DataFrame:
        qs, ans = [], []
        for _, r in df.iterrows():
            opts = r["options"]
            if isinstance(opts, np.ndarray):
                opts = opts.tolist()
            opts = list(opts)
            q = clean_text(r["question"]) + "\nOptions:\n" + "\n".join(
                f"{chr(65+i)}) {clean_text(o)}" for i, o in enumerate(opts)
            )
            qs.append(q)
            ans.append(clean_text(r["answer"]).upper())
        return pd.DataFrame({"question": qs, "answer": ans})

    def _load_hf(self, split: str):
        try:
            from datasets import load_dataset
        except Exception as e:
            raise BenchmarkAccessError(
                "Hugging Face `datasets` is required for the MATH benchmark. "
                "Install with: pip install datasets"
            ) from e
        kw = {"token": self.token} if self.token else {}
        return load_dataset(
            self.cfg.data.math_hf_id, self.cfg.data.math_hf_config, split=split, **kw
        )

    def _bundle(self, name: str, tr, va, te) -> BenchmarkBundle:
        tt = TASK_TYPES[name]
        return BenchmarkBundle(
            name, tt,
            self._standardize(tr, name, tt, "train"),
            self._standardize(va, name, tt, "val"),
            self._standardize(te, name, tt, "test"),
        )

    # -- driver --------------------------------------------------------------
    def load_all(self) -> Dict[str, BenchmarkBundle]:
        dispatch = {
            "gsm8k": self.gsm8k,
            "mmlu_pro": self.mmlu_pro,
            "math": self.math,
            "gpqa": self.gpqa,
        }
        bundles: Dict[str, BenchmarkBundle] = {}
        for name in self.cfg.data.benchmarks:
            fn = dispatch.get(name)
            if fn is None:
                logger.warning("Unknown benchmark '%s' — skipping.", name)
                continue
            logger.info("Loading benchmark: %s", name)
            try:
                b = fn()
                bundles[name] = b
                logger.info(
                    "  %s -> train=%d val=%d test=%d",
                    name, len(b.train), len(b.val), len(b.test),
                )
            except Exception as e:
                logger.exception("Failed to load %s: %s", name, e)
        if not bundles:
            raise RuntimeError("No benchmarks were loaded successfully.")
        return bundles
