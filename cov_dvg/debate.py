"""The debate stage: an adversarial critic followed by a final adjudicator.

Neither the critic nor the adjudicator is given ground truth. The adjudicator
independently re-verifies and produces the post-debate answer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .agents import answer_format_instruction
from .config import Config
from .grading import extract_answer, parse_confidence, strip_think_block
from .llm_backend import GenerationRequest, HFLocalLLM
from .utils import clean_text


CRITIC_SYSTEM = (
    "You are the adversarial critic in a multi-agent debate. You do NOT know the ground truth. "
    "Inspect the candidate solutions for concrete reasoning errors, unsupported assumptions, "
    "arithmetic mistakes, or option misreadings. Majority vote and stated confidence are evidence "
    "only, never proof. Re-solve disputed points as needed. Be concise and diagnostic."
)

JUDGE_SYSTEM = (
    "You are the final adjudicator after a structured multi-agent debate. You do NOT know the "
    "ground truth. Independently verify the problem, use the candidate arguments and critic only "
    "when they are correct, and do not choose an answer merely because it is the majority. "
    "Give a short justification, then a mechanically gradable final answer and calibrated confidence."
)


class RealDebateEngine:
    def __init__(self, backend: HFLocalLLM, config: Config):
        self.backend = backend
        self.cfg = config
        self.llm = config.llm

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
        user = (
            f"Problem:\n{sample['question']}\n\nCandidate solutions:\n{candidates}\n\n"
            "Write a compact critique identifying which reasoning is reliable, which is flawed, "
            "and what the correct resolution should depend on. Do not claim access to an answer key."
        )
        return GenerationRequest(
            messages=[{"role": "system", "content": CRITIC_SYSTEM},
                      {"role": "user", "content": user}],
            max_new_tokens=self.llm.critic_tokens(),
            temperature=0.55,
            tag=f"{sample['sample_id']}:critic",
        )

    def make_judge_request(
        self, sample: Dict[str, Any], responses: List[Dict[str, Any]], critique_text: str,
        tool_evidence: Optional[str] = None,
    ) -> GenerationRequest:
        task = sample["task_type"]
        candidates = self._candidate_block(responses)
        # Tool block is inserted only when present, so a None keeps the prompt (and
        # thus the generation cache key) identical to a tool-free run.
        tool_section = f"{tool_evidence[-2000:]}\n\n" if tool_evidence else ""
        user = (
            f"Problem:\n{sample['question']}\n\n"
            f"Candidate solutions:\n{candidates}\n\n"
            f"Adversarial critique:\n{critique_text[-2500:]}\n\n"
            f"{tool_section}"
            f"{answer_format_instruction(task)}\n"
            "Keep your justification brief so you have room to finish. You MUST end with these "
            "two lines, even if you must be brief:\nFINAL: <answer>\nCONFIDENCE: <0-100>"
        )
        return GenerationRequest(
            messages=[{"role": "system", "content": JUDGE_SYSTEM},
                      {"role": "user", "content": user}],
            max_new_tokens=self.llm.judge_tokens(task),
            temperature=0.45,
            tag=f"{sample['sample_id']}:judge",
        )

    def parse_debate(self, critic: Dict[str, Any], judge: Dict[str, Any], task: str) -> Dict[str, Any]:
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
        """Single-item convenience path (critic then judge)."""
        critic = self.backend.generate_one(self.make_critic_request(sample, responses))
        critique_text = strip_think_block(critic["text"])
        judge = self.backend.generate_one(self.make_judge_request(sample, responses, critique_text))
        return self.parse_debate(critic, judge, sample["task_type"])
