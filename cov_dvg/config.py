"""Central configuration for the COV-DVG pipeline.

The configuration is grouped into three dataclasses (data, LLM, gate) composed
into a single :class:`Config`. Every field has a sensible default; the scripts
expose the most important ones as command-line flags, and a handful may also be
set through environment variables for convenience on a headless server.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional


# Local project root (the directory that contains the ``data`` folder).
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


# ----------------------------------------------------------------------------
# Data / benchmark configuration
# ----------------------------------------------------------------------------
@dataclass
class DataConfig:
    """Where the datasets live and how many items to sample per split.

    Three benchmarks are read from local files under ``data_dir`` (GSM8K,
    MMLU-Pro, GPQA). MATH is pulled from the Hugging Face hub because the local
    project does not ship it.
    """

    data_dir: str = str(PROJECT_ROOT / "data")
    # Which benchmarks to include in the run.
    benchmarks: List[str] = field(
        default_factory=lambda: ["gsm8k", "mmlu_pro", "math", "gpqa"]
    )

    # Hugging Face dataset id for MATH (only remote dataset used).
    math_hf_id: str = "DigitalLearningGmbH/MATH-lighteval"
    math_hf_config: str = "default"

    # GPQA subset: "diamond" (198 expert-validated) or "main" (~448, larger for
    # more statistical power / training signal).
    gpqa_split: str = "diamond"

    # Per-benchmark sampling budget (defaults). Episode generation is the
    # expensive step, so these are modest and fully configurable.
    n_train: int = 120
    n_val: int = 40
    n_test: int = 200

    # Optional per-benchmark overrides, e.g. {"math": {"train": 300, "val": 100,
    # "test": 300}}. Lets debate-valuable tasks get more data without inflating
    # the others (MMLU-Pro's val pool is only 70 rows, GSM8K is already easy).
    sample_overrides: Dict[str, Dict[str, int]] = field(default_factory=dict)

    seed: int = 42

    def budget(self, benchmark: str) -> Dict[str, int]:
        o = self.sample_overrides.get(benchmark, {})
        return {
            "train": int(o.get("train", self.n_train)),
            "val": int(o.get("val", self.n_val)),
            "test": int(o.get("test", self.n_test)),
        }

    def resolved_data_dir(self) -> Path:
        return Path(self.data_dir).expanduser().resolve()


# ----------------------------------------------------------------------------
# LLM / episode-generation configuration
# ----------------------------------------------------------------------------
@dataclass
class LLMConfig:
    """Backend model and decoding settings for the debate episodes."""

    model_name: str = _env("COVDVG_MODEL", "Qwen/Qwen3-8B")
    num_agents: int = int(_env("COVDVG_NUM_AGENTS", "5"))

    # Inference backend: "hf" (transformers .generate) or "vllm" (continuous
    # batching + paged attention; ~5-10x faster for this workload).
    backend: str = _env("COVDVG_BACKEND", "hf")
    # vLLM-only knobs.
    gpu_memory_utilization: float = float(_env("COVDVG_GPU_MEM_UTIL", "0.90"))
    vllm_max_model_len: int = int(_env("COVDVG_MAX_MODEL_LEN", "16384"))
    # vLLM quantisation string (e.g. "bitsandbytes", "awq", "gptq"); empty = none.
    vllm_quantization: str = _env("COVDVG_VLLM_QUANT", "")

    # Qwen3-8B is a hybrid thinking model. Non-thinking mode (default) produces
    # plain reasoning + a parseable FINAL/CONFIDENCE tail (fast). Thinking mode
    # emits a long <think> block first (better on GPQA/MATH) and needs a much
    # larger token budget; the <think> span is stripped before answer parsing.
    enable_thinking: bool = False

    # Optional weight quantisation to fit larger models (14B/32B) on a 24 GB GPU.
    # Requires bitsandbytes. At most one may be True.
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    # Optional code-execution verifier stage (MATH/GSM8K). A programmer writes
    # Python/SymPy to recompute the answer; the sandboxed result is fed to the
    # adjudicator as evidence. Off by default so it never disturbs the baseline.
    enable_tools: bool = False
    tool_max_new_tokens: int = 768
    tool_timeout_s: float = 5.0

    max_input_tokens: int = 8192

    # Per-task generation budgets. The original 256-token cap truncated reasoning
    # on the hard benchmarks, producing unparseable answers and empty confidences.
    # Keys are task_type values (see data_loaders.TASK_TYPES).
    agent_max_new_tokens: Dict[str, int] = field(default_factory=lambda: {
        "gsm8k": 320, "math": 768, "mcq10": 1024, "mcq4": 1024,
    })
    judge_max_new_tokens_by_task: Dict[str, int] = field(default_factory=lambda: {
        "gsm8k": 320, "math": 768, "mcq10": 768, "mcq4": 768,
    })
    critic_max_new_tokens: int = 512

    # Much larger budgets used when enable_thinking is True. The <think> block
    # alone can exceed 2.5k tokens on GPQA; the probe showed ~25% of agents still
    # truncating at 3072, so these are set higher to let them finish + emit FINAL.
    agent_max_new_tokens_thinking: Dict[str, int] = field(default_factory=lambda: {
        "gsm8k": 1536, "math": 4096, "mcq10": 4096, "mcq4": 4096,
    })
    judge_max_new_tokens_thinking: Dict[str, int] = field(default_factory=lambda: {
        "gsm8k": 1536, "math": 3584, "mcq10": 3072, "mcq4": 3072,
    })
    critic_max_new_tokens_thinking: int = 2560

    # Fallbacks when a task_type is not present in the dicts above.
    initial_max_new_tokens: int = 512
    judge_max_new_tokens: int = 512

    temperature: float = 0.70
    top_p: float = 0.80
    top_k: int = 20
    repetition_penalty: float = 1.05

    # Independent agent generations batch by ``agent_batch_size`` (0 -> num_agents);
    # debate calls (critic / judge) batch across questions using debate_batch_size.
    # Lower these if an 8B model + long generations OOMs on a 24 GB card.
    agent_batch_size: int = int(_env("COVDVG_AGENT_BATCH_SIZE", "0"))
    debate_batch_size: int = int(_env("COVDVG_DEBATE_BATCH_SIZE", "2"))

    def agent_tokens(self, task_type: str) -> int:
        d = self.agent_max_new_tokens_thinking if self.enable_thinking else self.agent_max_new_tokens
        return int(d.get(task_type, self.initial_max_new_tokens))

    def judge_tokens(self, task_type: str) -> int:
        d = self.judge_max_new_tokens_thinking if self.enable_thinking else self.judge_max_new_tokens_by_task
        return int(d.get(task_type, self.judge_max_new_tokens))

    def critic_tokens(self) -> int:
        return self.critic_max_new_tokens_thinking if self.enable_thinking else self.critic_max_new_tokens


# ----------------------------------------------------------------------------
# Gate model / training configuration
# ----------------------------------------------------------------------------
@dataclass
class GateConfig:
    """Neural value-gate architecture, optimisation and policy settings."""

    # Architecture.
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.20

    # Optimisation.
    epochs: int = 120
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0

    # In-terminal validation evaluation cadence (epochs).
    eval_every: int = 10

    # Scheduler (ReduceLROnPlateau on the validation objective).
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2          # in eval steps, not raw epochs
    scheduler_min_lr: float = 1e-5

    # Loss weighting between the outcome head and the cost head.
    cost_loss_weight: float = 0.30

    # Policy: cost model predicts log1p(extra tokens); utility trades expected
    # accuracy gain against a small token penalty.
    token_penalty_per_1k: float = float(_env("COVDVG_TOKEN_PENALTY", "0.001"))
    # Cap on the fraction of validation items allowed to debate when selecting
    # the GLOBAL operating threshold (prevents a trivial "always debate" choice).
    max_debate_rate: float = float(_env("COVDVG_MAX_DEBATE_RATE", "0.50"))
    # For PER-BENCHMARK thresholds the token-penalised objective is trusted to set
    # the rate, so the cap is relaxed (1.0 = no cap): on benchmarks where debate
    # helps almost everywhere, debating nearly everything can be optimal.
    per_benchmark_max_debate_rate: float = float(_env("COVDVG_PB_MAX_DEBATE_RATE", "1.0"))
    # A benchmark needs at least this many validation episodes to earn its own
    # threshold; otherwise it falls back to the global threshold.
    per_benchmark_min_val: int = int(_env("COVDVG_PB_MIN_VAL", "15"))

    # Statistics for the final evaluation.
    bootstrap_iters: int = int(_env("COVDVG_BOOTSTRAP_ITERS", "2000"))

    seed: int = 42


# ----------------------------------------------------------------------------
# Top-level composed configuration
# ----------------------------------------------------------------------------
@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    gate: GateConfig = field(default_factory=GateConfig)

    output_dir: str = _env("COVDVG_OUTPUT_DIR", str(PROJECT_ROOT / "outputs"))
    seed: int = 42

    def __post_init__(self) -> None:
        # Keep the seed consistent across sub-configs.
        self.data.seed = self.seed
        self.gate.seed = self.seed
        if self.llm.num_agents < 3:
            raise ValueError("Use at least 3 agents for a meaningful panel.")
        if self.llm.debate_batch_size < 1:
            raise ValueError("llm.debate_batch_size must be >= 1.")
        if self.gate.eval_every < 1:
            raise ValueError("gate.eval_every must be >= 1.")

    # Convenient derived paths -------------------------------------------------
    @property
    def out(self) -> Path:
        p = Path(self.output_dir).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def episodes_dir(self) -> Path:
        p = self.out / "episodes"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def generation_cache(self) -> Path:
        p = self.out / "generation_cache.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def gate_dir(self) -> Path:
        p = self.out / "gate"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def results_dir(self) -> Path:
        p = self.out / "results"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
