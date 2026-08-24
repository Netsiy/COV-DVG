"""vLLM inference backend.

Drop-in replacement for :class:`HFLocalLLM` (same ``generate_many`` / ``generate_one``
interface, same JSONL cache) that uses vLLM's continuous batching and paged
attention. All un-cached prompts in a call are submitted to vLLM at once, which
is where the throughput win comes from — vLLM schedules them optimally rather
than the fixed mini-batches the HF backend uses.

Only imported when ``config.llm.backend == "vllm"``, so vLLM is an optional dep.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import Config
from .llm_backend import GenerationCache, GenerationRequest
from .utils import get_hf_token, get_logger, stable_seed

logger = get_logger("covdvg.vllm")


class VLLMBackend:
    def __init__(self, config: Config):
        self.cfg = config
        self.llm_cfg = config.llm
        self.cache = GenerationCache(config.generation_cache)
        self.token = get_hf_token()
        self.llm = None
        self.tokenizer = None
        self._load()

    def _load(self) -> None:
        try:
            from vllm import LLM
        except Exception as e:
            raise RuntimeError(
                "vLLM is not installed. Install it (ideally in a dedicated env): "
                "pip install vllm  — or run with --backend hf."
            ) from e

        kwargs: Dict[str, Any] = dict(
            model=self.llm_cfg.model_name,
            dtype="bfloat16",
            gpu_memory_utilization=self.llm_cfg.gpu_memory_utilization,
            max_model_len=self.llm_cfg.vllm_max_model_len,
            trust_remote_code=False,
            seed=self.cfg.seed,
        )
        # Quantisation: explicit string wins; else map load_in_4bit -> bitsandbytes.
        quant = self.llm_cfg.vllm_quantization.strip()
        if not quant and self.llm_cfg.load_in_4bit:
            quant = "bitsandbytes"
        if quant:
            kwargs["quantization"] = quant
            if quant == "bitsandbytes":
                kwargs["load_format"] = "bitsandbytes"
            logger.info("vLLM quantisation: %s", quant)

        logger.info("Loading vLLM model: %s (thinking=%s)",
                    self.llm_cfg.model_name, self.llm_cfg.enable_thinking)
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        logger.info("vLLM ready.")

    # -- rendering / keys ----------------------------------------------------
    def _render(self, messages: List[Dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.llm_cfg.enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def _key(self, req: GenerationRequest) -> str:
        payload = {
            "backend": "vllm",  # keep vLLM outputs separate from HF-cached ones
            "model": self.llm_cfg.model_name,
            "messages": req.messages,
            "max_new_tokens": req.max_new_tokens,
            "temperature": req.temperature,
            "top_p": self.llm_cfg.top_p,
            "top_k": self.llm_cfg.top_k,
            "repetition_penalty": self.llm_cfg.repetition_penalty,
            "enable_thinking": self.llm_cfg.enable_thinking,
            "quant": self.llm_cfg.vllm_quantization or ("bitsandbytes" if self.llm_cfg.load_in_4bit else "none"),
            "tag": req.tag,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    # -- generation ----------------------------------------------------------
    def generate_many(self, requests: Sequence[GenerationRequest]) -> List[Dict[str, Any]]:
        from vllm import SamplingParams

        results: List[Optional[Dict[str, Any]]] = [None] * len(requests)
        misses: List[Tuple[int, str, GenerationRequest]] = []
        for i, req in enumerate(requests):
            key = self._key(req)
            cached = self.cache.get(key)
            if cached is not None:
                results[i] = cached
            else:
                misses.append((i, key, req))

        if misses:
            prompts = [self._render(req.messages) for (_, _, req) in misses]
            sampling = [
                SamplingParams(
                    temperature=req.temperature,
                    top_p=self.llm_cfg.top_p,
                    top_k=self.llm_cfg.top_k,
                    repetition_penalty=self.llm_cfg.repetition_penalty,
                    max_tokens=req.max_new_tokens,
                    seed=stable_seed(key, self.cfg.seed),
                )
                for (_, key, req) in misses
            ]
            # Submit everything at once; vLLM batches internally. Order preserved.
            outputs = self.llm.generate(prompts, sampling)
            for (orig_idx, key, req), out in zip(misses, outputs):
                gen = out.outputs[0]
                text = gen.text.strip()
                in_tok = len(out.prompt_token_ids)
                out_tok = len(gen.token_ids)
                value = {
                    "text": text,
                    "input_tokens": int(in_tok),
                    "output_tokens": int(out_tok),
                    "total_tokens": int(in_tok + out_tok),
                    "tag": req.tag,
                }
                self.cache.set(key, value)
                results[orig_idx] = value

        return [r for r in results if r is not None]

    def generate_one(self, req: GenerationRequest) -> Dict[str, Any]:
        return self.generate_many([req])[0]
