"""Local Hugging Face LLM backend with a JSONL generation cache.

All generations are cached to disk keyed by the full request payload, so an
interrupted episode-generation run resumes without recomputing anything.
"""

from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import Config
from .utils import get_hf_token, get_logger, stable_seed

logger = get_logger("covdvg.llm")


@dataclass
class GenerationRequest:
    messages: List[Dict[str, str]]
    max_new_tokens: int
    temperature: float
    tag: str


class GenerationCache:
    """Append-only JSONL cache mapping request-hash -> generation result."""

    def __init__(self, path: Path):
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
            logger.info("Loaded %d cached generations from %s", len(self.data), self.path.name)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self.data.get(key)

    def set(self, key: str, value: Dict[str, Any]) -> None:
        self.data[key] = value
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


class HFLocalLLM:
    """Batched causal-LM generation on a single CUDA GPU."""

    def __init__(self, config: Config):
        self.cfg = config
        self.llm = config.llm
        self.cache = GenerationCache(config.generation_cache)
        self.token = get_hf_token()
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
                "Missing LLM dependencies. Install: transformers>=4.51,<6 and accelerate>=1.3"
            ) from e

        if Version(transformers.__version__) < Version("4.51.0"):
            raise RuntimeError(
                f"transformers {transformers.__version__} is too old for Qwen3. "
                "Install: transformers>=4.51,<6"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU not detected. This backend requires a GPU.")

        self.device = "cuda"
        props = torch.cuda.get_device_properties(0)
        logger.info("GPU: %s, %.1f GB", props.name, props.total_memory / (1024**3))

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm.model_name, token=self.token, trust_remote_code=False
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        load_kwargs: Dict[str, Any] = dict(
            token=self.token,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        # Optional 4-bit / 8-bit quantisation to fit 14B/32B on a 24 GB card.
        if self.llm.load_in_4bit or self.llm.load_in_8bit:
            if self.llm.load_in_4bit and self.llm.load_in_8bit:
                raise RuntimeError("Set at most one of load_in_4bit / load_in_8bit.")
            try:
                from transformers import BitsAndBytesConfig
            except Exception as e:
                raise RuntimeError("bitsandbytes is required for quantisation: pip install bitsandbytes") from e
            if self.llm.load_in_4bit:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                )
                logger.info("Loading in 4-bit (nf4).")
            else:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                logger.info("Loading in 8-bit.")
        else:
            load_kwargs["dtype"] = dtype

        self.model = AutoModelForCausalLM.from_pretrained(self.llm.model_name, **load_kwargs)
        self.model.eval()
        logger.info(
            "Loaded model: %s (%s, thinking=%s)",
            self.llm.model_name,
            "4bit" if self.llm.load_in_4bit else ("8bit" if self.llm.load_in_8bit else dtype),
            self.llm.enable_thinking,
        )

    def _render(self, messages: List[Dict[str, str]]) -> str:
        # ``enable_thinking`` is a Qwen3-family kwarg; fall back gracefully for
        # tokenizers whose chat template does not accept it.
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.llm.enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def _key(self, req: GenerationRequest) -> str:
        payload = {
            "model": self.llm.model_name,
            "messages": req.messages,
            "max_new_tokens": req.max_new_tokens,
            "temperature": req.temperature,
            "top_p": self.llm.top_p,
            "top_k": self.llm.top_k,
            "repetition_penalty": self.llm.repetition_penalty,
            "enable_thinking": self.llm.enable_thinking,
            "quant": "4bit" if self.llm.load_in_4bit else ("8bit" if self.llm.load_in_8bit else "none"),
            "tag": req.tag,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

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
            is_debate = all((":critic" in x[2].tag or ":judge" in x[2].tag) for x in misses)
            agent_bs = self.llm.agent_batch_size or self.llm.num_agents
            target = self.llm.debate_batch_size if is_debate else agent_bs
            batch_size = min(len(misses), max(1, target))
            for start in range(0, len(misses), batch_size):
                chunk = misses[start:start + batch_size]
                rendered = [x[3] for x in chunk]
                req0 = chunk[0][2]
                batch_seed = stable_seed("|".join(x[1] for x in chunk), self.cfg.seed)
                torch.manual_seed(batch_seed)
                torch.cuda.manual_seed_all(batch_seed)

                inputs = self.tokenizer(
                    rendered, return_tensors="pt", padding=True,
                    truncation=True, max_length=self.llm.max_input_tokens,
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
                        top_p=self.llm.top_p,
                        top_k=self.llm.top_k,
                        repetition_penalty=self.llm.repetition_penalty,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                for j, (orig_idx, key, req, _) in enumerate(chunk):
                    new_ids = seqs[j, padded_input_len:]
                    text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
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


def make_backend(config: Config):
    """Construct the configured inference backend (``hf`` or ``vllm``)."""
    backend = config.llm.backend.strip().lower()
    if backend == "vllm":
        from .vllm_backend import VLLMBackend
        return VLLMBackend(config)
    if backend == "hf":
        return HFLocalLLM(config)
    raise ValueError(f"Unknown backend '{backend}' (use 'hf' or 'vllm').")
