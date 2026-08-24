#!/usr/bin/env python
"""Stage 1 — generate debate episodes with the real LLM.

Runs the independent agent panel and the critic/adjudicator debate over every
sampled problem, labels the outcome (correction / no_change / subversion), and
writes per-benchmark-per-split episode tables plus combined split tables.

This is the only GPU-heavy stage. All generations are cached to
``outputs/generation_cache.jsonl`` so an interrupted run resumes for free.

Examples
--------
    python scripts/generate_episodes.py
    python scripts/generate_episodes.py --benchmarks gsm8k mmlu_pro gpqa \
        --n-train 120 --n-val 40 --n-test 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Make the package importable when run as a script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cov_dvg.agents import RealAgentPool
from cov_dvg.config import Config
from cov_dvg.data_loaders import BenchmarkLoader
from cov_dvg.debate import RealDebateEngine
from cov_dvg.episodes import EpisodeBuilder, generation_signature
from cov_dvg.grading import AnswerGrader
from cov_dvg.llm_backend import make_backend
from cov_dvg.utils import get_logger, seed_everything

logger = get_logger("covdvg.generate")


def _parse_overrides(items) -> dict:
    """Parse --sample-override 'math:train=300,val=100,test=300' entries."""
    out = {}
    for item in items or []:
        bench, _, spec = item.partition(":")
        bench = bench.strip()
        d = {}
        for kv in spec.split(","):
            kv = kv.strip()
            if not kv:
                continue
            k, _, v = kv.partition("=")
            d[k.strip()] = int(v)
        if bench and d:
            out[bench] = d
    return out


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    if args.data_dir:
        cfg.data.data_dir = args.data_dir
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.benchmarks:
        cfg.data.benchmarks = args.benchmarks
    if args.model:
        cfg.llm.model_name = args.model
    if args.num_agents:
        cfg.llm.num_agents = args.num_agents
    cfg.data.n_train = args.n_train
    cfg.data.n_val = args.n_val
    cfg.data.n_test = args.n_test
    cfg.data.gpqa_split = args.gpqa_split
    cfg.data.sample_overrides = _parse_overrides(args.sample_override)
    cfg.llm.enable_thinking = args.enable_thinking
    cfg.llm.load_in_4bit = args.load_4bit
    cfg.llm.load_in_8bit = args.load_8bit
    cfg.llm.enable_tools = args.enable_tools
    cfg.llm.backend = args.backend
    if args.vllm_quantization:
        cfg.llm.vllm_quantization = args.vllm_quantization
    if args.max_model_len:
        cfg.llm.vllm_max_model_len = args.max_model_len
    cfg.seed = args.seed
    cfg.__post_init__()
    return cfg


def _reuse_if_complete(path: Path, df: pd.DataFrame, sig: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        old = pd.read_csv(path)
    except Exception:
        return None
    ok = (
        "generation_signature" in old.columns
        and len(old) > 0
        and set(old["generation_signature"].astype(str)) == {sig}
        and set(old.get("sample_id", [])) == set(df["sample_id"])
    )
    return old if ok else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate COV-DVG debate episodes.")
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--benchmarks", nargs="+", default=None,
                    help="Subset of: gsm8k mmlu_pro math gpqa")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--num-agents", type=int, default=None)
    ap.add_argument("--n-train", type=int, default=120)
    ap.add_argument("--n-val", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--gpqa-split", type=str, default="diamond", choices=["diamond", "main"],
                    help="GPQA subset: diamond (198) or main (~448, more power).")
    ap.add_argument("--sample-override", action="append", default=[],
                    help="Per-benchmark budget, e.g. 'math:train=300,val=100,test=300'. Repeatable.")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="Run the model in thinking mode (big token budgets; better GPQA/MATH).")
    ap.add_argument("--load-4bit", action="store_true", help="4-bit quantisation (needs bitsandbytes).")
    ap.add_argument("--load-8bit", action="store_true", help="8-bit quantisation (needs bitsandbytes).")
    ap.add_argument("--enable-tools", action="store_true",
                    help="Enable the code-execution verifier stage for MATH/GSM8K.")
    ap.add_argument("--backend", type=str, default="hf", choices=["hf", "vllm"],
                    help="Inference backend. vllm is much faster for large sweeps.")
    ap.add_argument("--vllm-quantization", type=str, default="",
                    help="vLLM quant string, e.g. bitsandbytes / awq / gptq.")
    ap.add_argument("--max-model-len", type=int, default=0,
                    help="vLLM max_model_len (0 = config default 16384).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = build_config(args)
    seed_everything(cfg.seed)

    logger.info("Output dir: %s", cfg.out)
    logger.info("Benchmarks: %s", cfg.data.benchmarks)
    logger.info("Sample plan: train=%d val=%d test=%d",
                cfg.data.n_train, cfg.data.n_val, cfg.data.n_test)
    with (cfg.out / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)

    loader = BenchmarkLoader(cfg)
    bundles = loader.load_all()

    backend = make_backend(cfg)
    grader = AnswerGrader()
    agents = RealAgentPool(backend, cfg)
    debate = RealDebateEngine(backend, cfg)
    builder = EpisodeBuilder(agents, debate, grader, cfg)
    sig = generation_signature(cfg)

    frames = {"train": [], "val": [], "test": []}
    for bench, bundle in bundles.items():
        for split in ["train", "val", "test"]:
            df = getattr(bundle, split)
            path = cfg.episodes_dir / f"episodes_{bench}_{split}.csv"
            reused = _reuse_if_complete(path, df, sig)
            if reused is not None:
                logger.info("Reusing cached episodes: %s", path.name)
                episodes = reused
            else:
                logger.info("Building episodes: %s (%d items)", path.name, len(df))
                episodes = builder.build_df(df, f"{bench}:{split}")
                episodes.to_csv(path, index=False)
            frames[split].append(episodes)

    for split in ["train", "val", "test"]:
        combined = pd.concat(frames[split], ignore_index=True) if frames[split] else pd.DataFrame()
        combined.to_csv(cfg.episodes_dir / f"all_{split}.csv", index=False)
        logger.info("Wrote combined %s split: %d episodes", split, len(combined))

    train = pd.concat(frames["train"], ignore_index=True)
    print("\nObserved TRAIN debate outcomes by benchmark:")
    print(pd.crosstab(train["benchmark"], train["outcome"], margins=True))
    print(f"\nEpisodes written under: {cfg.episodes_dir}")
    print("Next: python scripts/train_gate.py")


if __name__ == "__main__":
    main()
