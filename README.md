# COV-DVG — Cost-Of-Verification Debate Value Gate

A clean, modular reimplementation of the multi-agent debate value-gate project.

Five role-conditioned LLM agents solve each problem **independently**, producing a
semantic-majority *base* answer. A **critic → adjudicator** debate then produces a
*debate* answer. The observed change is the label:

| delta | outcome     | meaning                                   |
|:-----:|:------------|:------------------------------------------|
| `+1`  | correction  | debate fixed a wrong base answer          |
| `0`   | no_change   | correctness unchanged                     |
| `-1`  | subversion  | debate broke a correct base answer        |

A trainable **value gate** looks only at *pre-debate* features (no ground-truth
leakage) and predicts, per problem, whether debating is worth its token cost. It
debates when `utility = P(correction) − P(subversion) − λ·cost > threshold`.

## Pipeline

The pipeline is split into three stages so the fast neural training can be
re-run without repeating the expensive LLM generation.

```
data/ ──► [1] generate_episodes.py ──► outputs/episodes/*.csv
                                           │
                                           ├─► [2] train_gate.py  ──► outputs/gate/
                                           │        (120 epochs, live progress,
                                           │         val eval every 10 epochs)
                                           │
                                           └─► [3] evaluate.py    ──► outputs/results/
```

### 1. Generate episodes (GPU / LLM — run on the server)

```bash
python scripts/generate_episodes.py --benchmarks gsm8k mmlu_pro math gpqa \
    --n-train 120 --n-val 40 --n-test 200
```

* Default model is `Qwen/Qwen3-8B`, run in **non-thinking** mode (`llm.enable_thinking=False`).
  Override with `--model`. Fits a 24 GB GPU in bf16 (~16 GB weights).
* Generation budgets are **per task** (`llm.agent_max_new_tokens`): GSM8K 320,
  MATH 768, MMLU-Pro/GPQA 1024; judge up to 768, critic 512. The hard benchmarks
  need the room — a 256-token cap truncates reasoning before the model can emit
  `FINAL:`, which was the main cause of weak GPQA/MMLU-Pro results.
* If generation OOMs, lower `--num-agents` or set `COVDVG_AGENT_BATCH_SIZE=2`
  (and `COVDVG_DEBATE_BATCH_SIZE=1`).

**Pushing accuracy on the hard benchmarks** (debate headroom is small once the
base model is strong — the lever is model capability, not debate):
* `--enable-thinking` runs Qwen3-8B in thinking mode (large token budgets,
  markedly higher GPQA/MATH accuracy). Use a separate `--output-dir`.
* `--load-4bit` / `--load-8bit` fit a bigger `--model` (e.g. `Qwen/Qwen3-14B`,
  `Qwen/Qwen3-32B`) on a 24 GB card (needs `bitsandbytes`).
* `--gpqa-split main` uses the larger GPQA set (~448) for statistical power.
* `--sample-override 'math:train=300,val=100,test=300'` gives debate-valuable
  tasks more data without inflating the others.

**Faster inference with vLLM** (`--backend vllm`): continuous batching + paged
attention, ~5-10x faster than the HF backend — needed for thinking mode, large
models, or big sweeps. Install vLLM in a **dedicated env** (it pins specific
torch builds and can clash with an existing install):

```bash
python scripts/generate_episodes.py --backend vllm --model Qwen/Qwen3-14B \
    --vllm-quantization bitsandbytes --benchmarks gsm8k mmlu_pro math gpqa \
    --gpqa-split main --output-dir outputs/14b_vllm
```

vLLM outputs are cached under a separate key from HF outputs, so the two never
collide. `--max-model-len` bounds the KV cache (default 16384).

**Code-execution verifier** (`--enable-tools`): for MATH/GSM8K, a programmer
writes a self-contained Python/SymPy program that recomputes the answer; it runs
in a sandbox (subprocess, timeout, Linux rlimits, math libs only) and its printed
result is given to the adjudicator as evidence to check. It targets the one place
debate has real headroom — procedural errors — and never sees the gold answer.
To A/B it against a baseline while reusing the (identical) agent/critic
generations, copy the cache into a new output dir first:

```bash
mkdir -p outputs/14b_tools
cp outputs/14b_vllm/generation_cache.jsonl outputs/14b_tools/generation_cache.jsonl
python scripts/generate_episodes.py --backend vllm --model Qwen/Qwen3-14B-AWQ \
    --benchmarks gsm8k mmlu_pro math gpqa --gpqa-split main \
    --sample-override 'math:train=300,val=100,test=300' \
    --enable-tools --output-dir outputs/14b_tools
```

Only the tool call and the MATH/GSM8K judges re-run (the MCQ judges get no tool
block, so they cache-hit); the agents and critics are reused for free.
* GSM8K, MMLU-Pro and GPQA are read from local files under `data/`.
* MATH is pulled from the Hugging Face hub (`DigitalLearningGmbH/MATH-lighteval`);
  set `HF_TOKEN` in the environment if needed.
* All generations are cached to `outputs/generation_cache.jsonl` — interrupted
  runs resume for free. Drop `math` from `--benchmarks` to skip the remote set.

### 2. Train the value gate (fast — CPU or GPU)

```bash
python scripts/train_gate.py --epochs 120 --eval-every 10 --lr 1e-3
```

You get a live tqdm progress bar, a one-line summary after **every** epoch, and a
full **validation evaluation block every 10 epochs** showing gate accuracy vs.
the majority / always-debate / oracle references, the debate rate, token savings,
outcome macro-F1 and cost MAE — so you can see whether the gate is genuinely
learning. An LR scheduler (`ReduceLROnPlateau` on the validation objective)
reduces the learning rate when progress stalls; the best-validation checkpoint is
restored at the end. History is written to `outputs/gate/training_history.csv`.

### 3. Evaluate on held-out test

```bash
python scripts/evaluate.py
```

Reports per-benchmark test accuracy with matched-rate baselines, paired bootstrap
CIs and exact McNemar p-values; writes `outputs/results/`.

## Data layout expected

```
data/
├── GSM8K/{train,test}.jsonl
├── MMLU-Pro/{validation,test}-00000-of-00001.parquet
└── GPQA/gpqa_diamond.csv
```

## Package layout

```
cov_dvg/
├── config.py        # DataConfig / LLMConfig / GateConfig composed in Config
├── utils.py         # hashing, seeding, logging, text helpers
├── grading.py       # answer extraction + benchmark-aware grader
├── data_loaders.py  # local + HF loaders, standardisation, fixed splits
├── llm_backend.py   # HFLocalLLM + JSONL generation cache
├── agents.py        # independent role-conditioned solver panel
├── debate.py        # adversarial critic + final adjudicator
├── features.py      # pre-debate feature extraction + semantic majority
├── episodes.py      # batched episode construction + labelling
├── gate.py          # trainable dual-head MLP value gate (model)
├── policy.py        # threshold tuning + policy monitoring metrics
├── trainer.py       # 120-epoch training loop (progress + periodic eval)
└── evaluation.py    # baselines + paired statistics
scripts/
├── generate_episodes.py
├── train_gate.py
└── evaluate.py
```

## Scientific invariants

* Agents solve independently; critic and adjudicator never see ground truth.
* The gate conditions **only** on pre-debate observable features.
* Threshold and gate parameters are fit on non-test episodes only.
* Ground truth is consumed exclusively inside `grading.AnswerGrader`.

## Install

```bash
pip install -r requirements.txt
```
