"""Pre-debate feature extraction and semantic-majority voting.

Every feature here is computable from the independent agent panel BEFORE any
debate output exists. Ground truth is never used. These are exactly the signals
the value gate is allowed to condition on.
"""

from __future__ import annotations

import math
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

import numpy as np

from .grading import AnswerGrader


# The ordered list of feature names the gate consumes. Kept explicit so the gate
# input dimension and column order are stable and inspectable.
FEATURE_NAMES: List[str] = [
    "pmax",
    "vote_margin",
    "answer_diversity",
    "answer_entropy",
    "is_consensus",
    "mean_confidence",
    "std_confidence",
    "majority_confidence",
    "minority_confidence",
    "confidence_gap",
    "confidence_vote_corr",
    "response_tokens_mean",
    "response_tokens_std",
    "base_output_tokens",
    "reasoning_similarity",
    "question_chars",
    "question_words",
    "num_numbers",
    "is_mcq",
    "is_math_expression",
    "is_gsm8k",
    "is_mmlu_pro",
    "is_gpqa",
    "parse_fail_rate",
]


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
        for j in range(i + 1, len(texts)):
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
        probs = np.asarray([s / n for s in sizes], dtype=float)
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
        maj_c = float(np.mean(maj_conf)) if maj_conf else 0.5
        min_c = float(np.mean(min_conf)) if min_conf else 0.0
        # Fraction of agents whose answer failed to parse (empty). A strong
        # pre-debate signal that the base vote is unreliable.
        parse_fail_rate = float(np.mean([1.0 if str(r["answer"]).strip() == "" else 0.0 for r in responses]))
        bench = sample.get("benchmark", "")
        return {
            "pmax": pmax,
            "vote_margin": margin,
            "answer_diversity": len(groups) / n,
            "answer_entropy": entropy_norm,
            "is_consensus": float(len(groups) == 1),
            "mean_confidence": float(np.mean(conf)),
            "std_confidence": float(np.std(conf)),
            "majority_confidence": maj_c,
            "minority_confidence": min_c,
            "confidence_gap": maj_c - min_c,
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
            "is_mmlu_pro": float(bench == "mmlu_pro"),
            "is_gpqa": float(bench == "gpqa"),
            "parse_fail_rate": parse_fail_rate,
        }
