"""Episode construction: run the panel + debate and label the outcome.

An *episode* is one problem instance turned into a training/eval row containing:
    * pre-debate features (feature_*),
    * the observed base (majority) answer and its correctness,
    * the observed post-debate answer and its correctness,
    * the outcome label delta in {-1 subversion, 0 no-change, +1 correction},
    * token costs.

Generation is done in three batched stages (agents -> critics -> judges) for GPU
throughput, but the causal information flow is identical to solving each item
independently: pre-debate quantities are computed before any critic/judge output
exists, and each judge sees only its own panel and critic.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd
from tqdm.auto import tqdm

from .agents import PROMPT_VERSION, RealAgentPool
from .config import Config
from .debate import RealDebateEngine
from .features import FeatureExtractor, semantic_majority
from .grading import AnswerGrader
from .tools import CodeExecutionTool
from .utils import stable_hash


OUTCOME_NAME = {1: "correction", 0: "no_change", -1: "subversion"}


def generation_signature(cfg: Config) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "model": cfg.llm.model_name,
        "enable_thinking": cfg.llm.enable_thinking,
        "quant": "4bit" if cfg.llm.load_in_4bit else ("8bit" if cfg.llm.load_in_8bit else "none"),
        "enable_tools": cfg.llm.enable_tools,
        "tool_max_new_tokens": cfg.llm.tool_max_new_tokens if cfg.llm.enable_tools else 0,
        "num_agents": cfg.llm.num_agents,
        "temperature": cfg.llm.temperature,
        # Effective per-task budgets (mode-aware).
        "agent_tokens": {t: cfg.llm.agent_tokens(t) for t in ["gsm8k", "math", "mcq10", "mcq4"]},
        "critic_tokens": cfg.llm.critic_tokens(),
        "judge_tokens": {t: cfg.llm.judge_tokens(t) for t in ["gsm8k", "math", "mcq10", "mcq4"]},
    }
    return stable_hash(json.dumps(payload, sort_keys=True), 24)


class EpisodeBuilder:
    def __init__(self, agents: RealAgentPool, debate: RealDebateEngine, grader: AnswerGrader, cfg: Config):
        self.agents = agents
        self.debate = debate
        self.grader = grader
        self.cfg = cfg
        self.features = FeatureExtractor()
        self.tool = CodeExecutionTool(agents.backend, cfg)

    def _assemble_row(self, sample, responses, majority_answer, feats, base_correct, debate, sig,
                      tool: Dict[str, Any] = None) -> Dict[str, Any]:
        debate_correct = self.grader.equal(debate["answer"], sample["answer"], sample["task_type"])
        delta = int(debate_correct) - int(base_correct)
        base_tokens = int(sum(r["total_tokens"] for r in responses))
        # Tool token cost is folded into the debate cost so comparisons stay fair.
        tool_tokens = int(tool["tool_tokens"]) if tool else 0
        extra_tokens = int(debate["extra_tokens"]) + tool_tokens
        row: Dict[str, Any] = {
            "generation_signature": sig,
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
            "outcome": OUTCOME_NAME[delta],
            "base_tokens": base_tokens,
            "debate_extra_tokens": int(extra_tokens),
            "always_debate_total_tokens": base_tokens + int(extra_tokens),
            "judge_confidence": float(debate["confidence"]),
            "agent_answers": json.dumps([r["answer"] for r in responses], ensure_ascii=False),
            "agent_confidences": json.dumps([round(float(r["confidence"]), 4) for r in responses]),
            "critic_text": debate["critic_text"],
            "judge_text": debate["judge_text"],
        }
        if tool is not None:
            tool_answer = tool.get("tool_answer", "")
            row["tool_answer"] = tool_answer
            row["tool_status"] = tool.get("tool_status", "")
            row["tool_tokens"] = tool_tokens
            # Analysis-only: whether the tool's computed answer matched gold. This
            # uses ground truth ONLY for reporting, never inside the pipeline.
            row["tool_correct"] = int(self.grader.equal(tool_answer, sample["answer"], sample["task_type"])) if tool_answer else 0
        for k, v in feats.items():
            row[f"feature_{k}"] = float(v)
        return row

    def build_df(self, df: pd.DataFrame, desc: str) -> pd.DataFrame:
        if len(df) == 0:
            return pd.DataFrame()
        samples = [r.to_dict() for _, r in df.iterrows()]
        n_agents = self.cfg.llm.num_agents
        sig = generation_signature(self.cfg)

        # Stage 1: independent solver generations for every question.
        agent_reqs: List = []
        for sample in samples:
            agent_reqs.extend(self.agents.make_requests(sample))
        raw_agents = self.agents.backend.generate_many(agent_reqs)

        responses_by_sample = []
        for i, sample in enumerate(samples):
            chunk = raw_agents[i * n_agents:(i + 1) * n_agents]
            if len(chunk) != n_agents:
                raise RuntimeError(f"Missing agent generations for {sample['sample_id']}")
            responses_by_sample.append(self.agents.parse_responses(sample, chunk))

        # Pre-debate quantities: computed before any critic/judge output exists.
        pre = []
        for sample, responses in zip(samples, responses_by_sample):
            majority_answer, winner, groups = semantic_majority(responses, sample["task_type"], self.grader)
            feats = self.features.extract(sample, responses, winner, groups)
            base_correct = self.grader.equal(majority_answer, sample["answer"], sample["task_type"])
            pre.append((majority_answer, feats, base_correct))

        # Stage 2: critics batched across independent questions.
        from .grading import strip_think_block
        critic_reqs = [
            self.debate.make_critic_request(sample, responses)
            for sample, responses in zip(samples, responses_by_sample)
        ]
        raw_critics = self.debate.backend.generate_many(critic_reqs)
        critiques = [strip_think_block(x["text"]) for x in raw_critics]

        # Optional tool stage (code-execution verifier) between critic and judge,
        # for computational tasks only. Batched across eligible questions.
        tool_results: Dict[int, Dict[str, Any]] = {}
        if self.cfg.llm.enable_tools:
            tool_idx = [i for i, s in enumerate(samples) if self.tool.applies(s["task_type"])]
            if tool_idx:
                tool_reqs = [self.tool.make_request(samples[i], responses_by_sample[i]) for i in tool_idx]
                raw_tools = self.tool.backend.generate_many(tool_reqs)
                for j, i in enumerate(tool_idx):
                    tool_results[i] = self.tool.run(samples[i], raw_tools[j])

        # Stage 3: adjudicators batched across questions (each seeing its own
        # panel, critic, and — when present — tool evidence).
        judge_reqs = [
            self.debate.make_judge_request(
                sample, responses, critique,
                tool_evidence=self.tool.evidence_block(tool_results.get(i)),
            )
            for i, (sample, responses, critique) in enumerate(zip(samples, responses_by_sample, critiques))
        ]
        raw_judges = self.debate.backend.generate_many(judge_reqs)

        rows = []
        for i, (sample, responses, pre_i, critic, judge) in enumerate(tqdm(
            zip(samples, responses_by_sample, pre, raw_critics, raw_judges),
            total=len(samples), desc=f"{desc}:assemble", leave=False,
        )):
            majority_answer, feats, base_correct = pre_i
            debate = self.debate.parse_debate(critic, judge, sample["task_type"])
            rows.append(self._assemble_row(
                sample, responses, majority_answer, feats, base_correct, debate, sig,
                tool=tool_results.get(i),
            ))
        return pd.DataFrame(rows)
