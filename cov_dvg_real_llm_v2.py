# ============================================================
# COV-DVG: REAL-LLM MULTI-AGENT DEBATE EVALUATION (KAGGLE)
# ============================================================
# This script replaces the earlier simulated agents/debate with real LLM calls.
# Default backend: Qwen/Qwen3-4B-Instruct-2507 on a Kaggle GPU.
#
# IMPORTANT KAGGLE SETUP
# ----------------------
# 1) Turn on a GPU accelerator.
# 2) Keep a Kaggle Secret named HF_TOKEN (needed for gated GPQA).
# 3) Do NOT upgrade numpy/pandas/scikit-learn. If needed, install only:
#      !pip install -q "transformers>=4.51,<6" "accelerate>=1.3" \
#          "math-verify[antlr4_13_2]"
# 4) Run this file. Start with COVDVG_RUN_MODE=pilot. If the signal is promising,
#    rerun with COVDVG_RUN_MODE=paper; the generation cache is reused.
#
# SCIENTIFIC DESIGN
# -----------------
# * Five genuinely generated, independent role-conditioned solver responses.
# * A real critic -> adjudicator debate stage, with no access to ground truth.
# * The gate sees ONLY pre-debate observable features (no correctness leakage).
# * Outcome labels are observed correction/no-change/subversion on training items.
# * A learned token-cost model predicts debate cost from pre-debate features.
# * Gate + threshold are fit/tuned on non-test episodes.
# * Test baselines: majority, always-debate, uncertainty heuristic,
#   low-confidence heuristic, random matched-rate, COV-DVG, oracle.
# * Paired bootstrap confidence intervals and McNemar exact p-values are reported.
# * MATH uses math-verify when available, with a safe fallback.
# * All generations are cached to JSONL so interrupted Kaggle runs can resume.
# ============================================================

from __future__ import annotations

import os
import re
import gc
import json
import math
import time
import random
import hashlib
import logging
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from collections import Counter
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("covdvg_real")


# ============================================================
# BASIC HELPERS
# ============================================================

def stable_hash(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def stable_seed(text: str, base_seed: int = 42) -> int:
    h = hashlib.sha256(f"{base_seed}|{text}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2**31 - 1)


def clean_text(x: Any) -> str:
    return "" if x is None else str(x).strip()


def get_hf_token() -> Optional[str]:
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


HF_TOKEN = get_hf_token()


def strip_think_block(text: str) -> str:
    # Harmless for non-thinking Qwen3-4B-Instruct-2507; useful if MODEL_NAME is
    # overridden with a thinking model.
    return re.sub(r"(?is)<think>.*?</think>", "", clean_text(text)).strip()


def extract_last_boxed(text: str) -> str:
    text = clean_text(text)
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return text
    i = start + len(marker)
    depth = 1
    chars: List[str] = []
    while i < len(text) and depth:
        ch = text[i]
        if ch == "{":
            depth += 1
            chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth:
                chars.append(ch)
        else:
            chars.append(ch)
        i += 1
    return "".join(chars).strip() or text


def extract_gsm8k_gold(text: str) -> str:
    text = clean_text(text)
    if "####" in text:
        text = text.rsplit("####", 1)[1].strip()
    return text.replace(",", "")


def normalize_simple_answer(x: Any) -> str:
    s = clean_text(x)
    s = s.replace("−", "-").replace("–", "-")
    s = s.strip("` ")
    s = re.sub(r"^\$|\$$", "", s).strip()
    s = re.sub(r"^\\boxed\{(.*)\}$", r"\1", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_confidence(text: str) -> float:
    matches = re.findall(
        r"(?im)^\s*CONFIDENCE\s*:\s*([0-9]{1,3}(?:\.[0-9]+)?)\s*%?\s*$",
        clean_text(text),
    )
    if not matches:
        return 0.50
    try:
        v = float(matches[-1])
        if v > 1.0:
            v /= 100.0
        return float(np.clip(v, 0.0, 1.0))
    except Exception:
        return 0.50


def extract_final_line(text: str) -> str:
    body = strip_think_block(text)
    matches = re.findall(
        r"(?im)^\s*FINAL(?:\s+ANSWER)?\s*:\s*(.*?)\s*$",
        body,
    )
    if matches:
        return matches[-1].strip()
    # Fallback: last non-empty line.
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def extract_mcq_answer(text: str, max_letter: str = "J") -> str:
    s = extract_final_line(text).upper().strip()
    # Prefer a standalone option label.
    m = re.search(rf"\b([A-{max_letter}])\b", s)
    if m:
        return m.group(1)
    # Last fallback across full response.
    hits = re.findall(rf"(?im)(?:ANSWER|OPTION|CHOICE|FINAL)\s*[:=\-]?\s*\(?([A-{max_letter}])\)?", text.upper())
    return hits[-1] if hits else ""


def extract_gsm8k_answer(text: str) -> str:
    s = extract_final_line(text)
    s = normalize_simple_answer(s)
    # Allow integer, decimal, simple fraction; choose the last numeric expression.
    hits = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*(?:\.\d+)?)?", s)
    if not hits:
        hits = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*(?:\.\d+)?)?", text)
    return hits[-1].replace(",", "").replace(" ", "") if hits else s


def extract_math_answer(text: str) -> str:
    s = extract_final_line(text)
    boxed = extract_last_boxed(s)
    return normalize_simple_answer(boxed)


def extract_answer(text: str, task_type: str) -> str:
    if task_type == "mcq10":
        return extract_mcq_answer(text, "J")
    if task_type == "mcq4":
        return extract_mcq_answer(text, "D")
    if task_type == "gsm8k":
        return extract_gsm8k_answer(text)
    if task_type == "math":
        return extract_math_answer(text)
    return normalize_simple_answer(extract_final_line(text))


def _to_float_or_fraction(s: str) -> Optional[float]:
    s = normalize_simple_answer(s).replace(",", "")
    try:
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?/[-+]?\d+(?:\.\d+)?", s):
            a, b = s.split("/", 1)
            if float(b) == 0:
                return None
            return float(a) / float(b)
        return float(s)
    except Exception:
        return None


class AnswerGrader:
    """Benchmark-aware grader. Ground truth is used ONLY here, never in gate features/prompts."""

    def __init__(self):
        self.math_verify_available = False
        try:
            from math_verify import parse, verify  # noqa: F401
            self.math_verify_available = True
            logger.info("math-verify available for MATH grading")
        except Exception:
            logger.warning(
                "math-verify not installed; MATH grading will use a weaker fallback. "
                "For paper runs install: math-verify[antlr4_13_2]"
            )

    def _math_equal(self, pred: str, gold: str) -> bool:
        pred = normalize_simple_answer(pred)
        gold = normalize_simple_answer(gold)
        if not pred or not gold:
            return False

        if self.math_verify_available:
            try:
                from math_verify import parse, verify
                # Wrapping in $...$ makes bare LaTeX parseable while parse() also
                # retains expression extraction as a fallback.
                g = parse(f"${gold}$")
                p = parse(f"${pred}$")
                if g and p and bool(verify(g, p)):
                    return True
            except Exception:
                pass

        # Numeric fallback.
        pv, gv = _to_float_or_fraction(pred), _to_float_or_fraction(gold)
        if pv is not None and gv is not None:
            return math.isclose(pv, gv, rel_tol=1e-6, abs_tol=1e-8)
        return pred.replace(" ", "") == gold.replace(" ", "")

    def equal(self, pred: str, gold: str, task_type: str) -> bool:
        pred = normalize_simple_answer(pred)
        gold = normalize_simple_answer(gold)
        if task_type in {"mcq10", "mcq4"}:
            return pred.upper() == gold.upper()
        if task_type == "gsm8k":
            pv, gv = _to_float_or_fraction(pred), _to_float_or_fraction(gold)
            if pv is not None and gv is not None:
                return math.isclose(pv, gv, rel_tol=1e-9, abs_tol=1e-9)
            return pred == gold
        if task_type == "math":
            return self._math_equal(pred, gold)
        return pred == gold

    def equivalent(self, a: str, b: str, task_type: str) -> bool:
        # Used for semantic majority grouping. Neither argument is ground truth.
        return self.equal(a, b, task_type)


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    seed: int = 42
    run_mode: str = os.getenv("COVDVG_RUN_MODE", "pilot").strip().lower()
    model_name: str = os.getenv("COVDVG_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
    num_agents: int = int(os.getenv("COVDVG_NUM_AGENTS", "5"))
    output_dir: str = os.getenv("COVDVG_OUTPUT_DIR", "/kaggle/working/covdvg_real")
    generation_cache: str = os.getenv(
        "COVDVG_CACHE", "/kaggle/working/covdvg_real/generation_cache.jsonl"
    )
    max_input_tokens: int = 8192
    initial_max_new_tokens: int = 256
    critic_max_new_tokens: int = 256
    judge_max_new_tokens: int = 192
    temperature: float = 0.70
    top_p: float = 0.80
    top_k: int = 20
    repetition_penalty: float = 1.05
    # Maximum fraction of validation samples allowed to debate when choosing the
    # primary operating point. This prevents trivial "always debate" selection.
    max_debate_rate: float = float(os.getenv("COVDVG_MAX_DEBATE_RATE", "0.50"))
    # Small accuracy-equivalent penalty per 1k average extra tokens. Accuracy is
    # still reported separately; this term only breaks close validation choices.
    token_penalty_per_1k: float = float(os.getenv("COVDVG_TOKEN_PENALTY", "0.001"))
    bootstrap_iters: int = int(os.getenv("COVDVG_BOOTSTRAP_ITERS", "2000"))
    # Batch critic/adjudicator generations across questions. Initial agent generations
    # already use num_agents as a safe batch on 16 GB GPUs.
    debate_batch_size: int = int(os.getenv("COVDVG_DEBATE_BATCH_SIZE", "2"))

    def __post_init__(self):
        if self.run_mode not in {"smoke", "pilot", "paper"}:
            raise ValueError("COVDVG_RUN_MODE must be 'smoke', 'pilot', or 'paper'")
        if self.num_agents < 3:
            raise ValueError("Use at least 3 agents for a meaningful panel")
        if self.debate_batch_size < 1:
            raise ValueError("COVDVG_DEBATE_BATCH_SIZE must be >= 1")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.generation_cache).parent.mkdir(parents=True, exist_ok=True)

    @property
    def sample_plan(self) -> Dict[str, int]:
        if self.run_mode == "smoke":
            return {"train": 3, "val": 2, "test": 2}
        # Pilot is intended to detect direction/sign before spending a full GPU run.
        if self.run_mode == "pilot":
            return {"train": 50, "val": 20, "test": 60}
        # Paper mode is still intentionally capped for computational feasibility.
        return {"train": 120, "val": 40, "test": 200}


CONFIG = Config()


# ============================================================
# DATASETS AND OFFICIAL-SPLIT SAMPLING
# ============================================================

class BenchmarkAccessError(RuntimeError):
    pass


def load_dataset_hf(*args, **kwargs):
    try:
        from datasets import load_dataset
    except Exception as e:
        raise BenchmarkAccessError("Hugging Face datasets is unavailable") from e
    return load_dataset(*args, **kwargs)


@dataclass
class BenchmarkBundle:
    name: str
    task_type: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class BenchmarkLoader:
    def __init__(self, config: Config):
        self.cfg = config
        self.token = HF_TOKEN

    def _kw(self) -> Dict[str, Any]:
        return {"token": self.token} if self.token else {}

    def _sample(self, df: pd.DataFrame, n: int, seed_text: str) -> pd.DataFrame:
        """Stable nested sampling: smoke ⊂ pilot ⊂ paper for cache reuse."""
        shuffled = df.sample(frac=1.0, random_state=stable_seed(seed_text, self.cfg.seed)).reset_index(drop=True)
        if n <= 0 or len(shuffled) <= n:
            return shuffled.copy()
        return shuffled.iloc[:n].copy().reset_index(drop=True)

    def _fixed_train_val(
        self, df: pd.DataFrame, train_n: int, val_n: int, max_val_n: int, seed_text: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Permanent disjoint train/validation pools across run modes.

        The first max_val_n rows are permanently reserved for validation; training
        always comes from the remainder. Thus a smoke/pilot validation item can never
        become a paper training item.
        """
        shuffled = df.sample(frac=1.0, random_state=stable_seed(seed_text, self.cfg.seed)).reset_index(drop=True)
        max_val_n = min(max_val_n, max(1, len(shuffled) - 1))
        val_pool = shuffled.iloc[:max_val_n].copy().reset_index(drop=True)
        train_pool = shuffled.iloc[max_val_n:].copy().reset_index(drop=True)
        tr = train_pool.iloc[:min(train_n, len(train_pool))].copy().reset_index(drop=True)
        va = val_pool.iloc[:min(val_n, len(val_pool))].copy().reset_index(drop=True)
        return tr, va

    @staticmethod
    def _standardize(df: pd.DataFrame, benchmark: str, task_type: str, split: str) -> pd.DataFrame:
        out = df.copy().reset_index(drop=True)
        out["benchmark"] = benchmark
        out["task_type"] = task_type
        out["split"] = split
        # sample_id deliberately excludes split so cached generations are reusable
        # across smoke/pilot/paper while split membership remains stored separately.
        out["sample_id"] = [
            f"{benchmark}:{stable_hash(q, 20)}" for q in out["question"].astype(str)
        ]
        return out[["sample_id", "benchmark", "task_type", "split", "question", "answer"]]

    def gsm8k(self) -> BenchmarkBundle:
        train_ds = load_dataset_hf("openai/gsm8k", "main", split="train", **self._kw()).to_pandas()
        test_ds = load_dataset_hf("openai/gsm8k", "main", split="test", **self._kw()).to_pandas()
        train_all = pd.DataFrame({
            "question": train_ds["question"].astype(str),
            "answer": train_ds["answer"].map(extract_gsm8k_gold),
        })
        test_all = pd.DataFrame({
            "question": test_ds["question"].astype(str),
            "answer": test_ds["answer"].map(extract_gsm8k_gold),
        })
        p = self.cfg.sample_plan
        tr, va = self._fixed_train_val(train_all, p["train"], p["val"], 40, "gsm8k-trainval")
        te = self._sample(test_all, p["test"], "gsm8k-test")
        return BenchmarkBundle(
            "gsm8k", "gsm8k",
            self._standardize(tr, "gsm8k", "gsm8k", "train"),
            self._standardize(va, "gsm8k", "gsm8k", "val"),
            self._standardize(te, "gsm8k", "gsm8k", "test"),
        )

    def mmlu_pro(self) -> BenchmarkBundle:
        # MMLU-Pro has a dedicated validation split (70 examples) and a large test split.
        val_raw = load_dataset_hf("TIGER-Lab/MMLU-Pro", split="validation", **self._kw()).to_pandas()
        test_raw = load_dataset_hf("TIGER-Lab/MMLU-Pro", split="test", **self._kw()).to_pandas()

        def convert(df: pd.DataFrame) -> pd.DataFrame:
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

        pool = convert(val_raw)
        test_all = convert(test_raw)
        # The official validation split is only 70 rows: reserve 50/20 at most.
        p = self.cfg.sample_plan
        tr_target = min(p["train"], max(35, len(pool) - 20))
        va_target = min(p["val"], len(pool) - tr_target)
        tr, va = self._fixed_train_val(pool, tr_target, va_target, 20, "mmlu-pro-validation")
        te = self._sample(test_all, p["test"], "mmlu-pro-test")
        return BenchmarkBundle(
            "mmlu_pro", "mcq10",
            self._standardize(tr, "mmlu_pro", "mcq10", "train"),
            self._standardize(va, "mmlu_pro", "mcq10", "val"),
            self._standardize(te, "mmlu_pro", "mcq10", "test"),
        )

    def math(self) -> BenchmarkBundle:
        train_ds = load_dataset_hf(
            "DigitalLearningGmbH/MATH-lighteval", "default", split="train", **self._kw()
        ).to_pandas()
        test_ds = load_dataset_hf(
            "DigitalLearningGmbH/MATH-lighteval", "default", split="test", **self._kw()
        ).to_pandas()
        train_all = pd.DataFrame({
            "question": train_ds["problem"].astype(str),
            "answer": train_ds["solution"].map(extract_last_boxed),
        })
        test_all = pd.DataFrame({
            "question": test_ds["problem"].astype(str),
            "answer": test_ds["solution"].map(extract_last_boxed),
        })
        p = self.cfg.sample_plan
        tr, va = self._fixed_train_val(train_all, p["train"], p["val"], 40, "math-trainval")
        te = self._sample(test_all, p["test"], "math-test")
        return BenchmarkBundle(
            "math", "math",
            self._standardize(tr, "math", "math", "train"),
            self._standardize(va, "math", "math", "val"),
            self._standardize(te, "math", "math", "test"),
        )

    def gpqa(self) -> BenchmarkBundle:
        if not self.token:
            raise BenchmarkAccessError("GPQA requires HF_TOKEN and accepted gated access")
        raw = load_dataset_hf(
            "Idavidrein/gpqa", "gpqa_diamond", split="train", token=self.token
        ).to_pandas()
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
            rng = random.Random(stable_seed(q0, self.cfg.seed))
            rng.shuffle(options)
            gold = chr(65 + options.index(correct))
            question = q0 + "\nOptions:\n" + "\n".join(
                f"{chr(65+i)}) {opt}" for i, opt in enumerate(options)
            )
            rows.append({"question": question, "answer": gold})
        all_df = pd.DataFrame(rows)

        # GPQA Diamond has no public held-out test labels. Use one permanent
        # deterministic 100/30/68 development/validation/test partition. Run modes
        # take nested prefixes from those fixed pools, so no item changes role later.
        shuffled = all_df.sample(
            frac=1.0, random_state=stable_seed("gpqa-diamond", self.cfg.seed)
        ).reset_index(drop=True)
        train_pool = shuffled.iloc[:100].copy().reset_index(drop=True)
        val_pool = shuffled.iloc[100:130].copy().reset_index(drop=True)
        test_pool = shuffled.iloc[130:].copy().reset_index(drop=True)
        p = self.cfg.sample_plan
        if self.cfg.run_mode == "smoke":
            tr_n, va_n, te_n = 3, 2, 2
        elif self.cfg.run_mode == "pilot":
            tr_n, va_n, te_n = min(50, len(train_pool)), min(20, len(val_pool)), min(60, len(test_pool))
        else:
            tr_n, va_n, te_n = len(train_pool), len(val_pool), len(test_pool)
        tr = train_pool.iloc[:tr_n].copy()
        va = val_pool.iloc[:va_n].copy()
        te = test_pool.iloc[:te_n].copy()
        return BenchmarkBundle(
            "gpqa", "mcq4",
            self._standardize(tr, "gpqa", "mcq4", "train"),
            self._standardize(va, "gpqa", "mcq4", "val"),
            self._standardize(te, "gpqa", "mcq4", "test"),
        )

    def load_all(self) -> Dict[str, BenchmarkBundle]:
        bundles = {}
        for name, fn in [
            ("gsm8k", self.gsm8k),
            ("mmlu_pro", self.mmlu_pro),
            ("math", self.math),
            ("gpqa", self.gpqa),
        ]:
            print("\n" + "="*72)
            print(f"LOADING {name.upper()}")
            print("="*72)
            try:
                b = fn()
                bundles[name] = b
                print(f"{name}: train={len(b.train)}, val={len(b.val)}, test={len(b.test)}")
            except Exception as e:
                logger.exception("Failed to load %s", name)
                print(f"SKIPPING {name}: {e}")
        return bundles


# ============================================================
# GENERATION CACHE
# ============================================================

class GenerationCache:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        self.data[obj["key"]] = obj["value"]
                    except Exception:
                        continue
            logger.info("Loaded %d cached generations", len(self.data))

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self.data.get(key)

    def set(self, key: str, value: Dict[str, Any]) -> None:
        self.data[key] = value
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


# ============================================================
# REAL LOCAL LLM BACKEND
# ============================================================

@dataclass
class GenerationRequest:
    messages: List[Dict[str, str]]
    max_new_tokens: int
    temperature: float
    tag: str


class HFLocalLLM:
    def __init__(self, config: Config):
        self.cfg = config
        self.cache = GenerationCache(config.generation_cache)
        self.model = None
        self.tokenizer = None
        self.device = None
        self._load()

    def _load(self) -> None:
        try:
            import torch
            import transformers
            from packaging.version import Version
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except Exception as e:
            raise RuntimeError(
                "Missing LLM dependencies. Install only: transformers>=4.51,<6 and accelerate>=1.3"
            ) from e

        if Version(transformers.__version__) < Version("4.51.0"):
            raise RuntimeError(
                f"transformers {transformers.__version__} is too old for Qwen3. "
                "Install: !pip install -q 'transformers>=4.51,<6' 'accelerate>=1.3'"
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU not detected. In Kaggle: Settings -> Accelerator -> GPU, then restart the session."
            )

        self.device = "cuda"
        props = torch.cuda.get_device_properties(0)
        logger.info(
            "GPU: %s, %.1f GB", props.name, props.total_memory / (1024**3)
        )

        # 4B fp16/bf16 avoids bitsandbytes dependency conflicts and fits common
        # 16 GB Kaggle GPUs. Prefer bf16 only when hardware supports it.
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name,
            token=HF_TOKEN,
            trust_remote_code=False,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            token=HF_TOKEN,
            dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        self.model.eval()
        logger.info("Loaded real LLM: %s", self.cfg.model_name)

    def _render(self, messages: List[Dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _key(self, req: GenerationRequest) -> str:
        payload = {
            "model": self.cfg.model_name,
            "messages": req.messages,
            "max_new_tokens": req.max_new_tokens,
            "temperature": req.temperature,
            "top_p": self.cfg.top_p,
            "top_k": self.cfg.top_k,
            "repetition_penalty": self.cfg.repetition_penalty,
            "tag": req.tag,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def generate_many(self, requests: Sequence[GenerationRequest]) -> List[Dict[str, Any]]:
        import torch
        results: List[Optional[Dict[str, Any]]] = [None] * len(requests)
        misses: List[Tuple[int, str, GenerationRequest, str]] = []

        for i, req in enumerate(requests):
            key = self._key(req)
            cached = self.cache.get(key)
            if cached is not None:
                results[i] = cached
            else:
                misses.append((i, key, req, self._render(req.messages)))

        if misses:
            # Requests passed together by the caller share generation settings.
            # Process in small batches to improve throughput without exhausting a 16GB GPU.
            is_debate_stage = all((":critic" in x[2].tag or ":judge" in x[2].tag) for x in misses)
            target_batch = self.cfg.debate_batch_size if is_debate_stage else self.cfg.num_agents
            batch_size = min(len(misses), max(1, target_batch))
            for start in range(0, len(misses), batch_size):
                chunk = misses[start:start+batch_size]
                rendered = [x[3] for x in chunk]
                req0 = chunk[0][2]
                batch_seed = stable_seed("|".join(x[1] for x in chunk), self.cfg.seed)
                torch.manual_seed(batch_seed)
                torch.cuda.manual_seed_all(batch_seed)

                inputs = self.tokenizer(
                    rendered,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.cfg.max_input_tokens,
                )
                input_counts = inputs["attention_mask"].sum(dim=1).tolist()
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                padded_input_len = inputs["input_ids"].shape[1]

                with torch.inference_mode():
                    seqs = self.model.generate(
                        **inputs,
                        max_new_tokens=req0.max_new_tokens,
                        do_sample=True,
                        temperature=req0.temperature,
                        top_p=self.cfg.top_p,
                        top_k=self.cfg.top_k,
                        repetition_penalty=self.cfg.repetition_penalty,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                for j, (orig_idx, key, req, _) in enumerate(chunk):
                    new_ids = seqs[j, padded_input_len:]
                    text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                    # Count non-padding generated tokens. EOS is harmless in cost accounting.
                    out_tok = int((new_ids != self.tokenizer.pad_token_id).sum().item())
                    value = {
                        "text": text,
                        "input_tokens": int(input_counts[j]),
                        "output_tokens": out_tok,
                        "total_tokens": int(input_counts[j]) + out_tok,
                        "tag": req.tag,
                    }
                    self.cache.set(key, value)
                    results[orig_idx] = value

                del inputs, seqs
                gc.collect()
                torch.cuda.empty_cache()

        return [r for r in results if r is not None]

    def generate_one(self, req: GenerationRequest) -> Dict[str, Any]:
        return self.generate_many([req])[0]


# ============================================================
# REAL AGENTS + REAL DEBATE
# ============================================================

PROMPT_VERSION = "covdvg-real-v2-stable-splits-batched-2026-08-12"

AGENT_ROLES = [
    "Independent solver: derive the answer carefully from first principles.",
    "Skeptical verifier: look for traps, ambiguous wording, and common mistakes while solving independently.",
    "Formal analyst: prefer explicit equations, definitions, or logical implications and verify each key step.",
    "Alternative-method solver: seek a different route from the most obvious approach and cross-check the result.",
    "Examiner: solve the problem, then actively test your tentative answer against the alternatives before committing.",
]


def answer_format_instruction(task_type: str) -> str:
    if task_type in {"mcq10", "mcq4"}:
        return "Your FINAL line must contain exactly one option letter (for example: FINAL: C)."
    if task_type == "gsm8k":
        return "Your FINAL line must contain only the final numeric answer (for example: FINAL: 42)."
    if task_type == "math":
        return "Your FINAL line must contain the final mathematical expression, preferably in LaTeX (for example: FINAL: \\frac{1}{4})."
    return "End with a line of the form FINAL: <answer>."


class RealAgentPool:
    def __init__(self, backend: HFLocalLLM, config: Config):
        self.backend = backend
        self.cfg = config
        if config.num_agents > len(AGENT_ROLES):
            raise ValueError(f"This implementation defines {len(AGENT_ROLES)} independent roles")

    def make_requests(self, sample: Dict[str, Any]) -> List[GenerationRequest]:
        task = sample["task_type"]
        fmt = answer_format_instruction(task)
        requests = []
        for i, role in enumerate(AGENT_ROLES[:self.cfg.num_agents]):
            system = (
                "You are one member of an independent problem-solving panel. "
                "Do the problem yourself; you have not seen any other agent's answer. "
                "Accuracy matters more than agreement. Give a concise justification, then a mechanically gradable final answer. "
                "Report calibrated confidence from 0 to 100 on the last line.\n\n"
                f"Your role: {role}"
            )
            user = (
                f"Problem:\n{sample['question']}\n\n"
                f"{fmt}\n"
                "Use this exact ending format:\n"
                "FINAL: <answer>\n"
                "CONFIDENCE: <0-100>"
            )
            requests.append(GenerationRequest(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_new_tokens=self.cfg.initial_max_new_tokens,
                temperature=self.cfg.temperature,
                tag=f"{sample['sample_id']}:agent{i}",
            ))
        return requests

    def parse_responses(self, sample: Dict[str, Any], raw: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        task = sample["task_type"]
        responses = []
        for i, g in enumerate(raw):
            text = strip_think_block(g["text"])
            responses.append({
                "agent_id": i,
                "role": AGENT_ROLES[i],
                "text": text,
                "answer": extract_answer(text, task),
                "confidence": parse_confidence(text),
                "input_tokens": g["input_tokens"],
                "output_tokens": g["output_tokens"],
                "total_tokens": g["total_tokens"],
            })
        return responses

    def get_responses(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self.parse_responses(sample, self.backend.generate_many(self.make_requests(sample)))


class RealDebateEngine:
    def __init__(self, backend: HFLocalLLM, config: Config):
        self.backend = backend
        self.cfg = config

    @staticmethod
    def _candidate_block(responses: List[Dict[str, Any]], max_chars_each: int = 1800) -> str:
        blocks = []
        for r in responses:
            txt = clean_text(r["text"])
            if len(txt) > max_chars_each:
                txt = txt[-max_chars_each:]
            blocks.append(
                f"--- Agent {r['agent_id']+1} ({r['role']}) ---\n"
                f"Parsed answer: {r['answer']}\n"
                f"Self-confidence: {100*r['confidence']:.0f}/100\n"
                f"Response:\n{txt}"
            )
        return "\n\n".join(blocks)

    def make_critic_request(self, sample: Dict[str, Any], responses: List[Dict[str, Any]]) -> GenerationRequest:
        candidates = self._candidate_block(responses)
        critic_system = (
            "You are the adversarial critic in a multi-agent debate. You do NOT know the ground truth. "
            "Inspect the candidate solutions for concrete reasoning errors, unsupported assumptions, arithmetic mistakes, "
            "or option misreadings. Majority vote and stated confidence are evidence only, never proof. "
            "Re-solve disputed points as needed. Be concise and diagnostic."
        )
        critic_user = (
            f"Problem:\n{sample['question']}\n\nCandidate solutions:\n{candidates}\n\n"
            "Write a compact critique identifying which reasoning is reliable, which is flawed, and what the correct resolution should depend on. "
            "Do not claim access to an answer key."
        )
        return GenerationRequest(
            messages=[{"role": "system", "content": critic_system}, {"role": "user", "content": critic_user}],
            max_new_tokens=self.cfg.critic_max_new_tokens,
            temperature=0.55,
            tag=f"{sample['sample_id']}:critic",
        )

    def make_judge_request(
        self, sample: Dict[str, Any], responses: List[Dict[str, Any]], critique_text: str
    ) -> GenerationRequest:
        task = sample["task_type"]
        candidates = self._candidate_block(responses)
        judge_system = (
            "You are the final adjudicator after a structured multi-agent debate. You do NOT know the ground truth. "
            "Independently verify the problem, use the candidate arguments and critic only when they are correct, and do not choose an answer merely because it is the majority. "
            "Give a short justification, then a mechanically gradable final answer and calibrated confidence."
        )
        judge_user = (
            f"Problem:\n{sample['question']}\n\n"
            f"Candidate solutions:\n{candidates}\n\n"
            f"Adversarial critique:\n{critique_text[-2500:]}\n\n"
            f"{answer_format_instruction(task)}\n"
            "Use this exact ending format:\nFINAL: <answer>\nCONFIDENCE: <0-100>"
        )
        return GenerationRequest(
            messages=[{"role": "system", "content": judge_system}, {"role": "user", "content": judge_user}],
            max_new_tokens=self.cfg.judge_max_new_tokens,
            temperature=0.45,
            tag=f"{sample['sample_id']}:judge",
        )

    def parse_debate(
        self, critic: Dict[str, Any], judge: Dict[str, Any], task: str
    ) -> Dict[str, Any]:
        critique_text = strip_think_block(critic["text"])
        judge_text = strip_think_block(judge["text"])
        return {
            "critic_text": critique_text,
            "judge_text": judge_text,
            "answer": extract_answer(judge_text, task),
            "confidence": parse_confidence(judge_text),
            "critic_tokens": int(critic["total_tokens"]),
            "judge_tokens": int(judge["total_tokens"]),
            "extra_tokens": int(critic["total_tokens"] + judge["total_tokens"]),
        }

    def run(self, sample: Dict[str, Any], responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        task = sample["task_type"]
        candidates = self._candidate_block(responses)

        critic_system = (
            "You are the adversarial critic in a multi-agent debate. You do NOT know the ground truth. "
            "Inspect the candidate solutions for concrete reasoning errors, unsupported assumptions, arithmetic mistakes, "
            "or option misreadings. Majority vote and stated confidence are evidence only, never proof. "
            "Re-solve disputed points as needed. Be concise and diagnostic."
        )
        critic_user = (
            f"Problem:\n{sample['question']}\n\nCandidate solutions:\n{candidates}\n\n"
            "Write a compact critique identifying which reasoning is reliable, which is flawed, and what the correct resolution should depend on. "
            "Do not claim access to an answer key."
        )
        critic = self.backend.generate_one(GenerationRequest(
            messages=[{"role": "system", "content": critic_system}, {"role": "user", "content": critic_user}],
            max_new_tokens=self.cfg.critic_max_new_tokens,
            temperature=0.55,
            tag=f"{sample['sample_id']}:critic",
        ))
        critique_text = strip_think_block(critic["text"])

        judge_system = (
            "You are the final adjudicator after a structured multi-agent debate. You do NOT know the ground truth. "
            "Independently verify the problem, use the candidate arguments and critic only when they are correct, and do not choose an answer merely because it is the majority. "
            "Give a short justification, then a mechanically gradable final answer and calibrated confidence."
        )
        judge_user = (
            f"Problem:\n{sample['question']}\n\n"
            f"Candidate solutions:\n{candidates}\n\n"
            f"Adversarial critique:\n{critique_text[-2500:]}\n\n"
            f"{answer_format_instruction(task)}\n"
            "Use this exact ending format:\nFINAL: <answer>\nCONFIDENCE: <0-100>"
        )
        judge = self.backend.generate_one(GenerationRequest(
            messages=[{"role": "system", "content": judge_system}, {"role": "user", "content": judge_user}],
            max_new_tokens=self.cfg.judge_max_new_tokens,
            temperature=0.45,
            tag=f"{sample['sample_id']}:judge",
        ))
        judge_text = strip_think_block(judge["text"])

        return {
            "critic_text": critique_text,
            "judge_text": judge_text,
            "answer": extract_answer(judge_text, task),
            "confidence": parse_confidence(judge_text),
            "critic_tokens": int(critic["total_tokens"]),
            "judge_tokens": int(judge["total_tokens"]),
            "extra_tokens": int(critic["total_tokens"] + judge["total_tokens"]),
        }


# ============================================================
# PRE-DEBATE FEATURES + SEMANTIC MAJORITY
# ============================================================

def semantic_majority(
    responses: List[Dict[str, Any]], task_type: str, grader: AnswerGrader
) -> Tuple[str, List[int], List[List[int]]]:
    groups: List[List[int]] = []
    for i, r in enumerate(responses):
        ans = r["answer"]
        placed = False
        for g in groups:
            if grader.equivalent(ans, responses[g[0]]["answer"], task_type):
                g.append(i)
                placed = True
                break
        if not placed:
            groups.append([i])

    # Largest equivalence class; tie-break with mean confidence, then first appearance.
    groups = sorted(
        groups,
        key=lambda g: (len(g), float(np.mean([responses[i]["confidence"] for i in g])), -g[0]),
        reverse=True,
    )
    winner = groups[0]
    representative = max(winner, key=lambda i: responses[i]["confidence"])
    return responses[representative]["answer"], winner, groups


def pairwise_text_similarity(texts: List[str]) -> float:
    if len(texts) < 2:
        return 1.0
    vals = []
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            a, b = texts[i][-1200:], texts[j][-1200:]
            vals.append(SequenceMatcher(None, a, b).ratio())
    return float(np.mean(vals)) if vals else 1.0


class FeatureExtractor:
    def extract(
        self,
        sample: Dict[str, Any],
        responses: List[Dict[str, Any]],
        majority_indices: List[int],
        groups: List[List[int]],
    ) -> Dict[str, float]:
        n = len(responses)
        sizes = sorted([len(g) for g in groups], reverse=True)
        pmax = sizes[0] / n
        margin = (sizes[0] - sizes[1]) / n if len(sizes) > 1 else 1.0
        probs = np.asarray([s/n for s in sizes], dtype=float)
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        entropy_norm = entropy / math.log(len(probs)) if len(probs) > 1 else 0.0

        conf = np.asarray([r["confidence"] for r in responses], dtype=float)
        out_tokens = np.asarray([r["output_tokens"] for r in responses], dtype=float)
        majority_set = set(majority_indices)
        maj_conf = [responses[i]["confidence"] for i in majority_indices]
        min_conf = [responses[i]["confidence"] for i in range(n) if i not in majority_set]
        support = np.asarray([1.0 if i in majority_set else 0.0 for i in range(n)], dtype=float)
        corr = 0.0
        if np.std(conf) > 1e-8 and np.std(support) > 1e-8:
            corr = float(np.corrcoef(conf, support)[0, 1])

        texts = [r["text"] for r in responses]
        q = sample["question"]
        return {
            "pmax": pmax,
            "vote_margin": margin,
            "answer_diversity": len(groups) / n,
            "answer_entropy": entropy_norm,
            "is_consensus": float(len(groups) == 1),
            "mean_confidence": float(np.mean(conf)),
            "std_confidence": float(np.std(conf)),
            "majority_confidence": float(np.mean(maj_conf)) if maj_conf else 0.5,
            "minority_confidence": float(np.mean(min_conf)) if min_conf else 0.0,
            "confidence_gap": (float(np.mean(maj_conf)) if maj_conf else 0.5) - (float(np.mean(min_conf)) if min_conf else 0.0),
            "confidence_vote_corr": corr,
            "response_tokens_mean": float(np.mean(out_tokens)),
            "response_tokens_std": float(np.std(out_tokens)),
            "base_output_tokens": float(np.sum(out_tokens)),
            "reasoning_similarity": pairwise_text_similarity(texts),
            "question_chars": float(len(q)),
            "question_words": float(len(q.split())),
            "num_numbers": float(len(re.findall(r"\d+(?:\.\d+)?", q))),
            "is_mcq": float(sample["task_type"].startswith("mcq")),
            "is_math_expression": float(sample["task_type"] == "math"),
            "is_gsm8k": float(sample["task_type"] == "gsm8k"),
        }


# ============================================================
# EPISODE GENERATION
# ============================================================

class EpisodeBuilder:
    def __init__(self, agents: RealAgentPool, debate: RealDebateEngine, grader: AnswerGrader):
        self.agents = agents
        self.debate = debate
        self.grader = grader
        self.features = FeatureExtractor()

    def build(self, row: pd.Series) -> Dict[str, Any]:
        sample = row.to_dict()
        responses = self.agents.get_responses(sample)
        majority_answer, winner, groups = semantic_majority(responses, sample["task_type"], self.grader)
        feats = self.features.extract(sample, responses, winner, groups)
        base_correct = self.grader.equal(majority_answer, sample["answer"], sample["task_type"])

        debate = self.debate.run(sample, responses)
        debate_correct = self.grader.equal(debate["answer"], sample["answer"], sample["task_type"])
        delta = int(debate_correct) - int(base_correct)
        outcome = {1: "correction", 0: "no_change", -1: "subversion"}[delta]

        base_tokens = int(sum(r["total_tokens"] for r in responses))
        signature_payload = {
            "prompt_version": PROMPT_VERSION,
            "model": self.agents.cfg.model_name,
            "num_agents": self.agents.cfg.num_agents,
            "temperature": self.agents.cfg.temperature,
            "initial_max_new_tokens": self.agents.cfg.initial_max_new_tokens,
            "critic_max_new_tokens": self.agents.cfg.critic_max_new_tokens,
            "judge_max_new_tokens": self.agents.cfg.judge_max_new_tokens,
        }
        generation_signature = stable_hash(json.dumps(signature_payload, sort_keys=True), 24)

        result: Dict[str, Any] = {
            "generation_signature": generation_signature,
            "sample_id": sample["sample_id"],
            "benchmark": sample["benchmark"],
            "split": sample["split"],
            "task_type": sample["task_type"],
            "question": sample["question"],
            "gold": sample["answer"],
            "base_answer": majority_answer,
            "base_correct": int(base_correct),
            "debate_answer": debate["answer"],
            "debate_correct": int(debate_correct),
            "delta": delta,
            "outcome": outcome,
            "base_tokens": base_tokens,
            "debate_extra_tokens": int(debate["extra_tokens"]),
            "always_debate_total_tokens": base_tokens + int(debate["extra_tokens"]),
            "judge_confidence": float(debate["confidence"]),
            "agent_answers": json.dumps([r["answer"] for r in responses], ensure_ascii=False),
            "agent_confidences": json.dumps([round(float(r["confidence"]), 4) for r in responses]),
            "critic_text": debate["critic_text"],
            "judge_text": debate["judge_text"],
        }
        for k, v in feats.items():
            result[f"feature_{k}"] = float(v)
        return result

    def build_df(self, df: pd.DataFrame, desc: str) -> pd.DataFrame:
        """Build a split in three batched stages: agents -> critics -> judges.

        This preserves exactly the same causal information flow as build(), but it
        batches critic and adjudicator calls across questions for materially better
        T4 utilization.
        """
        if len(df) == 0:
            return pd.DataFrame()
        samples = [r.to_dict() for _, r in df.iterrows()]
        n_agents = self.agents.cfg.num_agents

        # Stage 1: all independent solver generations.
        agent_reqs = []
        for sample in samples:
            agent_reqs.extend(self.agents.make_requests(sample))
        raw_agents = self.agents.backend.generate_many(agent_reqs)
        responses_by_sample = []
        for i, sample in enumerate(samples):
            chunk = raw_agents[i*n_agents:(i+1)*n_agents]
            if len(chunk) != n_agents:
                raise RuntimeError(f"Missing agent generations for {sample['sample_id']}")
            responses_by_sample.append(self.agents.parse_responses(sample, chunk))

        # Pre-debate quantities are computed before any critic/judge output exists.
        pre = []
        for sample, responses in zip(samples, responses_by_sample):
            majority_answer, winner, groups = semantic_majority(responses, sample["task_type"], self.grader)
            feats = self.features.extract(sample, responses, winner, groups)
            base_correct = self.grader.equal(majority_answer, sample["answer"], sample["task_type"])
            pre.append((majority_answer, feats, base_correct))

        # Stage 2: critics batched across independent questions.
        critic_reqs = [
            self.debate.make_critic_request(sample, responses)
            for sample, responses in zip(samples, responses_by_sample)
        ]
        raw_critics = self.debate.backend.generate_many(critic_reqs)
        critiques = [strip_think_block(x["text"]) for x in raw_critics]

        # Stage 3: adjudicators batched across questions, each seeing only its own panel+critic.
        judge_reqs = [
            self.debate.make_judge_request(sample, responses, critique)
            for sample, responses, critique in zip(samples, responses_by_sample, critiques)
        ]
        raw_judges = self.debate.backend.generate_many(judge_reqs)

        signature_payload = {
            "prompt_version": PROMPT_VERSION,
            "model": self.agents.cfg.model_name,
            "num_agents": self.agents.cfg.num_agents,
            "temperature": self.agents.cfg.temperature,
            "initial_max_new_tokens": self.agents.cfg.initial_max_new_tokens,
            "critic_max_new_tokens": self.agents.cfg.critic_max_new_tokens,
            "judge_max_new_tokens": self.agents.cfg.judge_max_new_tokens,
        }
        generation_signature = stable_hash(json.dumps(signature_payload, sort_keys=True), 24)

        rows = []
        for sample, responses, pre_i, critic, judge in tqdm(
            zip(samples, responses_by_sample, pre, raw_critics, raw_judges),
            total=len(samples), desc=f"{desc}:assemble"
        ):
            majority_answer, feats, base_correct = pre_i
            debate = self.debate.parse_debate(critic, judge, sample["task_type"])
            debate_correct = self.grader.equal(debate["answer"], sample["answer"], sample["task_type"])
            delta = int(debate_correct) - int(base_correct)
            outcome = {1: "correction", 0: "no_change", -1: "subversion"}[delta]
            base_tokens = int(sum(r["total_tokens"] for r in responses))
            result = {
                "generation_signature": generation_signature,
                "sample_id": sample["sample_id"],
                "benchmark": sample["benchmark"],
                "split": sample["split"],
                "task_type": sample["task_type"],
                "question": sample["question"],
                "gold": sample["answer"],
                "base_answer": majority_answer,
                "base_correct": int(base_correct),
                "debate_answer": debate["answer"],
                "debate_correct": int(debate_correct),
                "delta": delta,
                "outcome": outcome,
                "base_tokens": base_tokens,
                "debate_extra_tokens": int(debate["extra_tokens"]),
                "always_debate_total_tokens": base_tokens + int(debate["extra_tokens"]),
                "judge_confidence": float(debate["confidence"]),
                "agent_answers": json.dumps([r["answer"] for r in responses], ensure_ascii=False),
                "agent_confidences": json.dumps([round(float(r["confidence"]), 4) for r in responses]),
                "critic_text": debate["critic_text"],
                "judge_text": debate["judge_text"],
            }
            for k, v in feats.items():
                result[f"feature_{k}"] = float(v)
            rows.append(result)
        return pd.DataFrame(rows)


# ============================================================
# COV-DVG GATE: OUTCOME MODEL + COST MODEL
# ============================================================

class COVDVGRealGate:
    def __init__(self, config: Config):
        self.cfg = config
        self.feature_cols: List[str] = []
        self.scaler = StandardScaler()
        self.outcome_model: Any = None
        self.cost_model = Ridge(alpha=5.0)
        self.threshold = 0.0

    def fit(self, train: pd.DataFrame) -> Dict[str, Any]:
        self.feature_cols = sorted(c for c in train.columns if c.startswith("feature_"))
        X = train[self.feature_cols].to_numpy(dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = train["delta"].to_numpy(dtype=int)
        self.scaler.fit(X)
        Xs = self.scaler.transform(X)

        base = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            C=0.5,
            solver="lbfgs",
            random_state=self.cfg.seed,
        )
        counts = Counter(y.tolist())
        min_count = min(counts.values())
        if len(counts) == 1:
            # A pilot can legitimately observe no corrections/subversions. Do not
            # crash; a constant outcome model correctly represents "no learned signal".
            only_class = int(next(iter(counts)))
            self.outcome_model = DummyClassifier(strategy="constant", constant=only_class)
            model_kind = f"constant_class_{only_class}"
        elif min_count >= 3:
            cv = min(3, min_count)
            self.outcome_model = CalibratedClassifierCV(base, method="sigmoid", cv=cv)
            model_kind = f"calibrated_multinomial_cv{cv}"
        else:
            self.outcome_model = base
            model_kind = "multinomial_logistic"
        self.outcome_model.fit(Xs, y)

        self.cost_model.fit(Xs, train["debate_extra_tokens"].to_numpy(dtype=float))
        pred_cost = self.cost_model.predict(Xs)
        cost_mae = float(np.mean(np.abs(pred_cost - train["debate_extra_tokens"].to_numpy(dtype=float))))

        return {
            "model_kind": model_kind,
            "classes": [int(c) for c in self.outcome_model.classes_],
            "outcome_counts": {str(k): int(v) for k, v in sorted(counts.items())},
            "n_train": int(len(train)),
            "n_features": int(len(self.feature_cols)),
            "train_cost_mae_tokens": cost_mae,
        }

    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_cols].to_numpy(dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        Xs = self.scaler.transform(X)
        probs = self.outcome_model.predict_proba(Xs)
        classes = [int(c) for c in self.outcome_model.classes_]
        pmap = {c: probs[:, i] for i, c in enumerate(classes)}
        p_sub = pmap.get(-1, np.zeros(len(df)))
        p_same = pmap.get(0, np.zeros(len(df)))
        p_corr = pmap.get(1, np.zeros(len(df)))
        denom = p_sub + p_same + p_corr
        denom = np.where(denom <= 0, 1.0, denom)
        p_sub, p_same, p_corr = p_sub/denom, p_same/denom, p_corr/denom
        pred_cost = np.maximum(1.0, self.cost_model.predict(Xs))

        out = df.copy()
        out["p_subversion"] = p_sub
        out["p_no_change"] = p_same
        out["p_correction"] = p_corr
        out["pred_debate_tokens"] = pred_cost
        # Expected accuracy delta minus a small token-cost term.
        out["covdvg_utility"] = (
            p_corr - p_sub - self.cfg.token_penalty_per_1k * (pred_cost / 1000.0)
        )
        return out

    def tune_threshold(self, val: pd.DataFrame) -> Dict[str, Any]:
        s = self.score_frame(val)
        utilities = np.sort(s["covdvg_utility"].unique())
        candidates = [float("inf"), float("-inf")]
        if len(utilities):
            candidates.extend(utilities.tolist())
            if len(utilities) > 1:
                candidates.extend(((utilities[:-1] + utilities[1:]) / 2.0).tolist())

        best = None
        for t in candidates:
            use = s["covdvg_utility"].to_numpy() > t
            rate = float(np.mean(use))
            if rate > self.cfg.max_debate_rate + 1e-12:
                continue
            chosen = np.where(use, s["debate_correct"].to_numpy(), s["base_correct"].to_numpy())
            acc = float(np.mean(chosen))
            extra = float(np.mean(use * s["debate_extra_tokens"].to_numpy()))
            objective = acc - self.cfg.token_penalty_per_1k * (extra / 1000.0)
            candidate = (objective, acc, -extra, -rate, t)
            if best is None or candidate > best[0]:
                best = (candidate, {
                    "threshold": float(t),
                    "validation_accuracy": acc,
                    "validation_debate_rate": rate,
                    "validation_avg_extra_tokens": extra,
                    "validation_objective": objective,
                })
        if best is None:
            raise RuntimeError("Could not select a validation threshold")
        self.threshold = best[1]["threshold"]
        return best[1]


# ============================================================
# STATISTICS + BASELINES
# ============================================================

def bootstrap_diff(a: np.ndarray, b: np.ndarray, iters: int, seed: int) -> Tuple[float, float, float]:
    # Paired difference a-b.
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(iters, dtype=float)
    d = a.astype(float) - b.astype(float)
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        diffs[i] = float(np.mean(d[idx]))
    return float(np.mean(d)), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    # Two-sided exact binomial test on discordant paired outcomes.
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


def random_matched_accuracy(base: np.ndarray, debate: np.ndarray, k: int, seed: int, reps: int = 2000) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(base)
    vals = []
    for _ in range(reps):
        use = np.zeros(n, dtype=bool)
        if k:
            use[rng.choice(n, size=k, replace=False)] = True
        vals.append(float(np.mean(np.where(use, debate, base))))
    return float(np.mean(vals)), float(np.std(vals))


def evaluate_benchmark(scored: pd.DataFrame, gate: COVDVGRealGate, cfg: Config) -> Tuple[Dict[str, Any], pd.DataFrame]:
    df = scored.copy().reset_index(drop=True)
    use = df["covdvg_utility"].to_numpy() > gate.threshold
    base = df["base_correct"].to_numpy(dtype=int)
    debate = df["debate_correct"].to_numpy(dtype=int)
    cov = np.where(use, debate, base).astype(int)
    n = len(df)
    k = int(use.sum())

    # Matched-rate heuristics: same number of debates as COV-DVG.
    uncertainty_use = matched_rate_decision(df["feature_answer_entropy"].to_numpy(), k, True)
    lowconf_use = matched_rate_decision(df["feature_majority_confidence"].to_numpy(), k, False)
    uncertainty = np.where(uncertainty_use, debate, base).astype(int)
    lowconf = np.where(lowconf_use, debate, base).astype(int)
    rand_mean, rand_std = random_matched_accuracy(base, debate, k, stable_seed(df.iloc[0]["benchmark"], cfg.seed))
    oracle = np.maximum(base, debate)

    diff_base, lo_base, hi_base = bootstrap_diff(cov, base, cfg.bootstrap_iters, cfg.seed + 11)
    diff_deb, lo_deb, hi_deb = bootstrap_diff(cov, debate, cfg.bootstrap_iters, cfg.seed + 29)

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
        "token_savings_vs_always_pct": float(100.0 * (1.0 - np.mean(primary_tokens) / np.mean(always_tokens))),
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


def validation_policy_curve(gate: COVDVGRealGate, val_scored: pd.DataFrame, test_scored: pd.DataFrame) -> pd.DataFrame:
    # Thresholds are determined exclusively from validation utility quantiles.
    rows = []
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


# ============================================================
# RUNNER
# ============================================================

def save_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    print("="*80)
    print("COV-DVG REAL-LLM EXPERIMENT")
    print("="*80)
    print(f"Run mode: {CONFIG.run_mode}")
    print(f"Model: {CONFIG.model_name}")
    print(f"Agents: {CONFIG.num_agents}")
    print(f"Output: {CONFIG.output_dir}")
    print("Ground truth is NEVER passed to agents, critic, adjudicator, or gate features.")

    outdir = Path(CONFIG.output_dir)
    save_json(outdir / "config.json", asdict(CONFIG))

    loader = BenchmarkLoader(CONFIG)
    bundles = loader.load_all()
    if not bundles:
        raise RuntimeError("No benchmarks loaded")

    backend = HFLocalLLM(CONFIG)
    grader = AnswerGrader()
    agents = RealAgentPool(backend, CONFIG)
    debate = RealDebateEngine(backend, CONFIG)
    builder = EpisodeBuilder(agents, debate, grader)

    episode_frames = {"train": [], "val": [], "test": []}
    for bench, bundle in bundles.items():
        for split in ["train", "val", "test"]:
            df = getattr(bundle, split)
            path = outdir / f"episodes_{bench}_{split}.csv"
            # Episode files are reusable only if complete for the current sample IDs.
            reuse = False
            if path.exists():
                try:
                    old = pd.read_csv(path)
                    signature_payload = {
                        "prompt_version": PROMPT_VERSION,
                        "model": CONFIG.model_name,
                        "num_agents": CONFIG.num_agents,
                        "temperature": CONFIG.temperature,
                        "initial_max_new_tokens": CONFIG.initial_max_new_tokens,
                        "critic_max_new_tokens": CONFIG.critic_max_new_tokens,
                        "judge_max_new_tokens": CONFIG.judge_max_new_tokens,
                    }
                    expected_signature = stable_hash(json.dumps(signature_payload, sort_keys=True), 24)
                    signature_ok = (
                        "generation_signature" in old.columns
                        and len(old) > 0
                        and set(old["generation_signature"].astype(str)) == {expected_signature}
                    )
                    if signature_ok and set(old.get("sample_id", [])) == set(df["sample_id"]):
                        episodes = old
                        reuse = True
                        print(f"Reusing completed episode file: {path.name}")
                except Exception:
                    reuse = False
            if not reuse:
                episodes = builder.build_df(df, f"{bench}:{split}")
                episodes.to_csv(path, index=False)
            episode_frames[split].append(episodes)

    train = pd.concat(episode_frames["train"], ignore_index=True)
    val = pd.concat(episode_frames["val"], ignore_index=True)
    test = pd.concat(episode_frames["test"], ignore_index=True)

    print("\nObserved TRAIN debate outcomes:")
    print(pd.crosstab(train["benchmark"], train["outcome"], margins=True))

    gate = COVDVGRealGate(CONFIG)
    fit_info = gate.fit(train)
    val_scored = gate.score_frame(val)
    tune_info = gate.tune_threshold(val)
    # Re-score only for saved consistent columns (threshold does not alter utility).
    val_scored = gate.score_frame(val)
    test_scored = gate.score_frame(test)

    save_json(outdir / "gate_fit.json", fit_info)
    save_json(outdir / "threshold_selection.json", tune_info)
    val_scored.to_csv(outdir / "validation_scored.csv", index=False)

    all_metrics = []
    details = []
    for bench, g in test_scored.groupby("benchmark", sort=False):
        metrics, detail = evaluate_benchmark(g, gate, CONFIG)
        all_metrics.append(metrics)
        details.append(detail)

    summary = pd.DataFrame(all_metrics)
    detail_df = pd.concat(details, ignore_index=True)
    summary.to_csv(outdir / "results_summary.csv", index=False)
    detail_df.to_csv(outdir / "test_details.csv", index=False)

    curve = validation_policy_curve(gate, val_scored, test_scored)
    curve.to_csv(outdir / "policy_curve.csv", index=False)

    # Gate coefficient inspection for the underlying logistic estimator when accessible.
    try:
        est = gate.outcome_model
        if hasattr(est, "estimator"):
            # CalibratedClassifierCV doesn't expose a single fitted estimator coefficient reliably.
            coef_df = pd.DataFrame({"feature": gate.feature_cols})
        else:
            coef_df = pd.DataFrame({"feature": gate.feature_cols})
            for i, cls in enumerate(est.classes_):
                coef_df[f"coef_class_{int(cls)}"] = est.coef_[i]
        coef_df.to_csv(outdir / "gate_coefficients.csv", index=False)
    except Exception:
        pass

    print("\n" + "="*100)
    print("PRIMARY TEST RESULTS")
    print("="*100)
    display_cols = [
        "benchmark", "n", "majority_accuracy", "always_debate_accuracy",
        "covdvg_accuracy", "uncertainty_matched_accuracy", "random_matched_accuracy_mean",
        "debate_rate", "token_savings_vs_always_pct", "covdvg_minus_majority",
        "covdvg_minus_majority_ci_low", "covdvg_minus_majority_ci_high",
    ]
    print(summary[display_cols].to_string(index=False))

    print("\nGate fit:")
    print(json.dumps(fit_info, indent=2))
    print("\nValidation-selected operating point:")
    print(json.dumps(tune_info, indent=2))

    print("\nInterpretation guardrails:")
    print("* GSM8K/MATH test rows come from their official test splits.")
    print("* MMLU-Pro test rows come from its official test split; its 70-row validation split trains/tunes the gate portion.")
    print("* GPQA Diamond has no separate public labeled test split here; its reported test is a deterministic held-out partition of the 198 labeled Diamond items.")
    print("* Do not claim a benefit unless the paired CI/p-value and matched-rate baselines support it; pilot mode is directional only.")
    print(f"\nAll artifacts saved under: {outdir}")
    if CONFIG.run_mode == "smoke":
        print("Smoke test complete. Stable nested IDs mean its generations are reusable in pilot/paper.")
    elif CONFIG.run_mode == "pilot":
        print("If pilot results show a consistent positive signal, set COVDVG_RUN_MODE=paper and rerun; cached generations are reused where sample IDs overlap.")


if __name__ == "__main__":
    main()
