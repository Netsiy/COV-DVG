"""Answer parsing, normalisation and benchmark-aware grading.

Ground truth is consumed ONLY inside :class:`AnswerGrader`. It is never exposed
to agents, the critic, the adjudicator, or the gate's feature extractor.
"""

from __future__ import annotations

import math
import re
from typing import Any, List, Optional

import numpy as np

from .utils import clean_text, get_logger

logger = get_logger("covdvg.grading")


# ----------------------------------------------------------------------------
# Low-level text extraction
# ----------------------------------------------------------------------------
def strip_think_block(text: str) -> str:
    """Remove <think>...</think> spans (harmless for non-thinking models)."""
    return re.sub(r"(?is)<think>.*?</think>", "", clean_text(text)).strip()


def extract_last_boxed(text: str) -> str:
    """Return the content of the last ``\\boxed{...}`` group, or the text."""
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
    """Pull the numeric gold answer that follows the GSM8K '####' marker."""
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
    matches = re.findall(r"(?im)^\s*FINAL(?:\s+ANSWER)?\s*:\s*(.*?)\s*$", body)
    if matches:
        return matches[-1].strip()
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def extract_mcq_answer(text: str, max_letter: str = "J") -> str:
    """Extract a single option letter, robust to truncated / unformatted output.

    Tries, in order: an explicit FINAL line, answer-phrase patterns, a boxed
    letter, then a parenthesised letter near the end. Returns "" only if nothing
    answer-like is present (a genuine parse failure), never a stray capital from
    prose.
    """
    body = strip_think_block(text)
    up = body.upper()

    # 1) Explicit FINAL / FINAL ANSWER line -> a standalone letter within it.
    finals = re.findall(r"(?im)^\s*FINAL(?:\s+ANSWER)?\s*:\s*(.*?)\s*$", body)
    if finals:
        m = re.search(rf"\b([A-{max_letter}])\b", finals[-1].upper())
        if m:
            return m.group(1)

    # 2) Answer-phrase patterns anywhere (last occurrence wins).
    hits = re.findall(
        rf"(?im)(?:ANSWER|OPTION|CHOICE|FINAL)\s*(?:IS|:|=|-)?\s*\(?([A-{max_letter}])\)?\b",
        up,
    )
    if hits:
        return hits[-1]

    # 3) A boxed letter, e.g. \boxed{C}.
    boxed = normalize_simple_answer(extract_last_boxed(body)).upper()
    m = re.fullmatch(rf"\(?([A-{max_letter}])\)?", boxed)
    if m:
        return m.group(1)

    # 4) A parenthesised option letter near the end of the response.
    tail = up[-400:]
    cands = re.findall(rf"\(([A-{max_letter}])\)", tail)
    if cands:
        return cands[-1]

    return ""


def extract_gsm8k_answer(text: str) -> str:
    s = normalize_simple_answer(extract_final_line(text))
    num = r"[-+]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*(?:\.\d+)?)?"
    hits = re.findall(num, s) or re.findall(num, text)
    return hits[-1].replace(",", "").replace(" ", "") if hits else s


def extract_math_answer(text: str) -> str:
    return normalize_simple_answer(extract_last_boxed(extract_final_line(text)))


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


# ----------------------------------------------------------------------------
# Grader
# ----------------------------------------------------------------------------
class AnswerGrader:
    """Benchmark-aware equality. Ground truth enters the pipeline only here."""

    def __init__(self) -> None:
        self.math_verify_available = False
        try:
            from math_verify import parse, verify  # noqa: F401
            self.math_verify_available = True
            logger.info("math-verify available for MATH grading.")
        except Exception:
            logger.warning(
                "math-verify not installed; MATH grading uses a weaker fallback. "
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
                g = parse(f"${gold}$")
                p = parse(f"${pred}$")
                if g and p and bool(verify(g, p)):
                    return True
            except Exception:
                pass
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
        """Semantic-majority grouping. Neither argument is ground truth."""
        return self.equal(a, b, task_type)
