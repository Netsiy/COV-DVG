"""The independent solver panel: role-conditioned agents.

Each agent solves the problem alone. No agent sees another's answer, and no
agent sees ground truth.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .config import Config
from .grading import extract_answer, parse_confidence, strip_think_block
from .llm_backend import GenerationRequest, HFLocalLLM


PROMPT_VERSION = "covdvg-clean-v4-qwen3-8b-2026-08-20"

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
        return ("Your FINAL line must contain the final mathematical expression, preferably in "
                "LaTeX (for example: FINAL: \\frac{1}{4}).")
    return "End with a line of the form FINAL: <answer>."


class RealAgentPool:
    def __init__(self, backend: HFLocalLLM, config: Config):
        self.backend = backend
        self.cfg = config
        self.llm = config.llm
        if self.llm.num_agents > len(AGENT_ROLES):
            raise ValueError(f"This implementation defines {len(AGENT_ROLES)} independent roles.")

    def make_requests(self, sample: Dict[str, Any]) -> List[GenerationRequest]:
        task = sample["task_type"]
        fmt = answer_format_instruction(task)
        requests = []
        for i, role in enumerate(AGENT_ROLES[:self.llm.num_agents]):
            system = (
                "You are one member of an independent problem-solving panel. "
                "Do the problem yourself; you have not seen any other agent's answer. "
                "Accuracy matters more than agreement. Reason concisely and efficiently, then a "
                "mechanically gradable final answer. "
                "Report calibrated confidence from 0 to 100 on the last line.\n\n"
                f"Your role: {role}"
            )
            user = (
                f"Problem:\n{sample['question']}\n\n"
                f"{fmt}\n"
                "Keep your reasoning brief so you have room to finish. You MUST end with these "
                "two lines, even if you must be brief:\n"
                "FINAL: <answer>\n"
                "CONFIDENCE: <0-100>"
            )
            requests.append(GenerationRequest(
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                max_new_tokens=self.llm.agent_tokens(task),
                temperature=self.llm.temperature,
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
