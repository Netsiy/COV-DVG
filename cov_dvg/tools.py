"""Code-execution verifier tool for the debate (optional, off by default).

For computational tasks (MATH, GSM8K) the panel's errors are often procedural —
arithmetic slips, algebra mistakes — that self-debate misses. This stage asks a
"programmer" to write a short, self-contained Python/SymPy program that recomputes
the answer, executes it in a sandbox, and hands the printed result to the
adjudicator as *evidence to verify* (never as ground truth — the code can be wrong
or crash). It raises the debate ceiling on exactly the benchmark where debate has
real headroom.

The tool NEVER sees the gold answer. Execution is sandboxed: a subprocess with a
wall-clock timeout and (on Linux) CPU/address-space rlimits, restricted to a small
set of math libraries.
"""

from __future__ import annotations

import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence

from .config import Config
from .grading import extract_answer, normalize_simple_answer, strip_think_block
from .llm_backend import GenerationRequest
from .utils import clean_text, get_logger

logger = get_logger("covdvg.tools")

# Tools only apply where executable code can actually produce the answer.
TOOL_TASK_TYPES = {"math", "gsm8k"}

try:
    import resource  # Linux/Unix only
except Exception:  # pragma: no cover - Windows dev machine
    resource = None


_PREAMBLE = (
    "import math\n"
    "from fractions import Fraction\n"
    "import itertools, cmath, statistics\n"
    "try:\n"
    "    import sympy\n"
    "    from sympy import *\n"
    "except Exception:\n"
    "    pass\n"
)


def extract_code(text: str) -> str:
    """Pull the last ```python ...``` (or ```) fenced block; fallback to text."""
    body = strip_think_block(text)
    blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", body, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return blocks[-1].strip()
    return body.strip()


def parse_tool_output(stdout: str, task_type: str) -> str:
    """Extract the tool's answer: an explicit 'ANSWER:' line, else last line."""
    s = clean_text(stdout)
    if not s:
        return ""
    m = re.findall(r"(?im)^\s*ANSWER\s*:\s*(.+?)\s*$", s)
    cand = m[-1] if m else [ln for ln in s.splitlines() if ln.strip()][-1]
    # Reuse the benchmark answer extractor for consistent normalisation.
    return extract_answer(f"FINAL: {cand}", task_type) or normalize_simple_answer(cand)


def safe_exec_python(code: str, timeout: float = 5.0, mem_mb: int = 1536) -> Dict[str, str]:
    """Execute ``code`` in a sandboxed subprocess; return {stdout, stderr, status}."""
    full = _PREAMBLE + "\n" + code

    def _limits():  # pragma: no cover - runs only in the child on Linux
        if resource is not None:
            cpu = int(timeout) + 1
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
            b = mem_mb * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (b, b))
            except Exception:
                pass

    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", full],
            capture_output=True, text=True, timeout=timeout,
            preexec_fn=_limits if resource is not None else None,
        )
        return {
            "stdout": (proc.stdout or "").strip()[-2000:],
            "stderr": (proc.stderr or "").strip()[-500:],
            "status": "ok" if proc.returncode == 0 else f"exit{proc.returncode}",
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "timeout", "status": "timeout"}
    except Exception as e:  # pragma: no cover
        return {"stdout": "", "stderr": str(e)[:300], "status": "error"}


class CodeExecutionTool:
    def __init__(self, backend, config: Config):
        self.backend = backend
        self.cfg = config
        self.llm = config.llm

    @staticmethod
    def applies(task_type: str) -> bool:
        return task_type in TOOL_TASK_TYPES

    def make_request(self, sample: Dict[str, Any], responses: List[Dict[str, Any]]) -> GenerationRequest:
        cand = ", ".join(sorted({clean_text(r["answer"]) for r in responses if clean_text(r["answer"])}))
        system = (
            "You are a careful Python programmer verifying a math answer. You do NOT know the "
            "answer key. Write ONE short, self-contained program that computes the final answer "
            "from scratch and prints it. Use only: math, sympy, fractions, itertools, statistics, "
            "cmath. No file, network, or system access. Keep it correct and minimal."
        )
        user = (
            f"Problem:\n{sample['question']}\n\n"
            f"Candidate answers from solvers (may be wrong): {cand or 'none'}\n\n"
            "Write a Python program that computes the answer and, as its LAST action, prints exactly:\n"
            "print(f'ANSWER: {result}')\n"
            "Return only one ```python``` code block."
        )
        return GenerationRequest(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_new_tokens=self.llm.tool_max_new_tokens,
            temperature=0.20,
            tag=f"{sample['sample_id']}:tool",
        )

    def run(self, sample: Dict[str, Any], raw_gen: Dict[str, Any]) -> Dict[str, Any]:
        code = extract_code(raw_gen["text"])
        exec_res = safe_exec_python(code, timeout=self.llm.tool_timeout_s)
        tool_answer = parse_tool_output(exec_res["stdout"], sample["task_type"])
        return {
            "tool_code": code,
            "tool_stdout": exec_res["stdout"],
            "tool_stderr": exec_res["stderr"],
            "tool_status": exec_res["status"],
            "tool_answer": tool_answer,
            "tool_tokens": int(raw_gen["total_tokens"]),
        }

    @staticmethod
    def evidence_block(tool: Optional[Dict[str, Any]]) -> Optional[str]:
        if not tool:
            return None
        code = clean_text(tool.get("tool_code", ""))[:1200]
        out = clean_text(tool.get("tool_stdout", ""))[:600]
        ans = clean_text(tool.get("tool_answer", ""))
        status = tool.get("tool_status", "")
        if not ans and status != "ok":
            # Failed execution is weak evidence; still disclose it briefly.
            return (
                "Tool verification (a Python program was executed; it does NOT know the answer key). "
                f"Execution status: {status}. No reliable computed answer was produced; rely on your "
                "own verification."
            )
        return (
            "Tool verification (a Python program was executed; it does NOT know the answer key). "
            "Treat it as evidence to check, not as ground truth (the code may be wrong).\n"
            f"Code:\n{code}\n"
            f"Executed output:\n{out}\n"
            f"Parsed tool answer: {ans}"
        )
