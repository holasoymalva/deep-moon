# deep-moon — Working Plan

Distill a small language model specialized in writing code, evaluate it rigorously, and publish it on HuggingFace.

**The real goal:** learn the full lifecycle (distillation → training → evaluation → quantization → publication). The model is the vehicle; the learning is the product.

---

## 0. The honest premise (read this first)

One thing to be clear about from day one, because it shapes every other decision:

**You are not going to beat `Qwen2.5-Coder-1.5B-Instruct` at general-purpose code.** That model is already a distillation, made by Alibaba, with thousands of GPU-hours and proprietary data, starting from the same base model you'd be starting from. Replicating that on a laptop is not going to happen.

**Where you *can* win: a narrow domain.** A 1.5B model specialized in *one* thing can beat a generalist of its size — and sometimes models far larger — at that one thing. That is the central lesson of small language models, and it's a result that is genuinely reachable here.

That's why this plan pins down a **narrow domain** in Phase 1. The alternative (a generic "code model") produces something worse than the starting point, plus a confusing conclusion about why.

---

## 1. Hardware and architecture decisions

Your machine: **MacBook Pro M5 Max, 64 GB unified memory, 1.6 TB free.** That's considerably more than this project needs, which opens up options.

| Decision | Choice | Why |
|---|---|---|
| Training framework | **MLX** (`mlx-lm` 0.30+) | Native Apple Silicon, unified memory with no host↔device copies. Fully local, no rented GPUs. |
| Student | **`Qwen2.5-Coder-1.5B`** (base, not instruct) | Apache 2.0, pre-trained on code, family spans 0.5B/1.5B/3B/7B so you can compare scales. |
| Teacher | **`Qwen3-Coder-30B-A3B-Instruct`** | Apache 2.0. MoE: 30B total but **3.3B active** → fast local generation. ~17 GB at 4-bit. |
| Local runtime | MLX to train, **GGUF + Ollama** to serve | Ollama is already installed on your machine. |

**On the teacher:** starting from an MoE with 3.3B active parameters is precisely why data generation is viable on a laptop. A dense 32B would generate the same dataset several times slower.

**Licensing — verified:** student and teacher are both **Apache 2.0**. That explicitly permits using the teacher's outputs for training and publishing the resulting model, including commercially. It only requires attribution. There is no anti-distillation clause of the kind other model families carry. This is not a minor point — it's what makes the project publishable at all.

---

## 2. The two kinds of distillation (the core concept)

This is the thing you're here to learn, so it's worth stating plainly.

**Sequence-level distillation (off-policy / SFT on teacher outputs)**
The teacher generates solutions; you train the student to imitate that text. Simple, and it works with *any* teacher — even one behind an API, with no access to weights. It's SFT on synthetic data.
*Limitation:* the student only sees the chosen token, not *how confident* the teacher was or what the alternatives were. Most of the signal is thrown away.

**Logit-level distillation (classic KD)**
The student learns the teacher's *full distribution* over the vocabulary at each position, minimizing KL divergence. Far more signal per token → converges on less data.
*Hard requirement:* teacher and student must **share a tokenizer**. Otherwise the logits don't align position-by-position and the loss is meaningless.

**The detail to verify rather than assume:** Qwen2.5 and Qwen3 both declare `vocab_size = 151936`, but Qwen3 added its own control tokens. A matching number **does not prove** the token→id mapping is identical. Phase 0 includes a script that checks this empirically before we bet the design on it.

Hence two tracks:
- **Track A (sequence)** — always works, tokenizer-independent. Done first.
- **Track B (logits)** — only if the check passes. If it fails, the fallback is *on-policy distillation*, which in recent literature (2026) consistently beats plain SFT and does **not** require a shared tokenizer: the student generates, the teacher corrects its own trajectories. This attacks *exposure bias*, which is the real failure mode of off-policy SFT.

---

## 3. Phases

### Phase 0 — Environment and fundamentals
**Goal:** everything installed, a model running locally, inference understood.

- [ ] Install `uv` and `git-lfs` (neither is present yet). The system Python 3.9 is too old → `uv` will manage an isolated 3.12+.
- [ ] `uv add mlx-lm huggingface_hub datasets evalplus`
- [ ] HF account + write token (`hf auth login`)
- [ ] Download student and teacher; run inference on both; measure tok/s
- [ ] **Script `scripts/check_tokenizers.py`** → compares teacher and student vocabularies and decides whether Track B is viable. *This result gates Phase 5.*

**You learn:** weight formats (safetensors vs GGUF — MLX **cannot** load GGUF, it needs safetensors), chat templates, quantization, tokenization.

---

### Phase 1 — Pick the domain + build evaluation **before** training
**Goal:** know how to measure before you have anything to measure.

This phase comes before training on purpose. It's the most common methodological mistake: train first, evaluate later, and end up with no baseline to compare against. Without a baseline, "my model scores 61%" means nothing at all.

- [ ] **Fix the narrow domain.** Candidates: pandas/polars and data wrangling; FastAPI + Pydantic; pytest testing; SQL. Criteria: verifiable by execution, with enough public seed data.
- [ ] Set up `evalplus` → **HumanEval+ / MBPP+** (hardened-test versions; the originals have tests loose enough to overstate pass@1)
- [ ] Build your own **domain held-out set** (~50-100 problems with executable tests). Public benchmarks don't measure your niche.
- [ ] **Measure three baselines:**
  - Base student (`Qwen2.5-Coder-1.5B`) → the floor
  - Teacher (`Qwen3-Coder-30B-A3B`) → the ceiling
  - `Qwen2.5-Coder-1.5B-Instruct` → **the honest rival**, the one to beat in-domain

**You learn:** pass@k and why temperature changes everything; sandboxing generated code; data contamination. Note: HumanEval is saturated at the frontier (96-98%), but at 1.5B scale it still discriminates fine. For contamination-free measurement there's **LiveCodeBench**, which only uses problems published after the training cutoff.

---

### Phase 2 — Teacher data generation
**Goal:** a distillation dataset whose quality is verified by execution.

- [ ] Gather domain seed prompts (public datasets + synthetic generation)
- [ ] Teacher generates solution + tests for each
- [ ] **Rejection sampling: run the tests and discard whatever fails.** This filter is the difference between distilling the teacher's competence and also distilling its hallucinations.
- [ ] Deduplicate; **decontaminate against the held-out set and against HumanEval/MBPP**
- [ ] Target: ~10-20k verified examples

**You learn:** synthetic data, rejection sampling, why dataset quality dominates volume.

---

### Phase 3 — Distillation v1 (Track A: sequence)
- [ ] LoRA via `mlx_lm.lora --train`
- [ ] Sweep hyperparameters: rank, learning rate, epochs
- [ ] Watch validation loss → catch overfitting
- [ ] `mlx_lm.fuse` to merge the adapter into the base weights

**You learn:** LoRA vs full fine-tune, QLoRA, loss curves. With 64 GB, a **full fine-tune of 1.5B also fits** (~24 GB with Adam states) → comparing LoRA vs full is a free and very instructive experiment.

---

### Phase 4 — Evaluation v1 and analysis
- [ ] Run the full suite against all three baselines
- [ ] **Inspect failures by hand.** Aggregates hide the actual failure modes.
- [ ] Document: did it improve in-domain? how much did it degrade outside?

**You learn:** *catastrophic forgetting* — the model will almost certainly get worse outside the domain. That's an expected result, not a bug, and it belongs in the model card.

---

### Phase 5 — Distillation v2 (Track B or on-policy)
Depending on what `check_tokenizers.py` returned in Phase 0:
- **If tokenizers match** → KD with KL divergence over teacher logits
- **If not** → on-policy distillation: student generates, teacher scores/corrects
- [ ] Compare v1 vs v2 under the same evaluation

**You learn:** why forward KL is *mode-covering* and pushes the student to spread probability mass over low-probability regions of the teacher → hallucination. That's the theoretical reason on-policy wins.

---

### Phase 6 — Quantization and local execution
- [ ] Quantize (4-bit / 8-bit) and measure the **quality loss**, not just the size
- [ ] Export to GGUF → **caveat:** `mlx_lm.fuse --export-gguf` only supports Llama/Mistral-style architectures. For Qwen you need llama.cpp's `convert_hf_to_gguf.py`.
- [ ] Ollama `Modelfile`; measure tok/s and memory
- [ ] Quality/size/speed curve per quantization level

**You learn:** quantization families, real deployment trade-offs.

---

### Phase 7 — Publishing to HuggingFace
- [ ] Upload weights (safetensors + GGUF + LoRA adapters separately)
- [ ] **Honest model card:** training data, recipe, evals *with* baselines, limitations, out-of-domain degradation, Apache 2.0 attribution to Qwen
- [ ] Publish **the distillation dataset** too (Apache 2.0)
- [ ] Optional: submit to leaderboards

**You learn:** the HF ecosystem, model cards, provenance and licensing.

---

## 4. Repo structure

```
deep-moon/
├── README.md
├── PLAN.md
├── pyproject.toml
├── scripts/
│   ├── check_tokenizers.py    # Phase 0 — decides Track A vs B
│   ├── generate_data.py       # Phase 2
│   ├── verify_data.py         # Phase 2 — rejection sampling
│   ├── decontaminate.py       # Phase 2
│   ├── train_lora.py          # Phase 3
│   ├── distill_logits.py      # Phase 5
│   └── evaluate.py            # Phase 1/4
├── data/                      # gitignored, ships with a datasheet
├── evals/
│   └── domain_heldout/
├── models/                    # gitignored
└── notes/                     # per-phase lab notebook
```

---

## 5. Success criteria

**Learning** (what actually matters):
- Being able to explain the difference between sequence-level and logit-level distillation, and when each applies
- Understanding why pass@1 at temperature 0.8 is not the same as at 0.0
- Having seen catastrophic forgetting in your own metrics

**Model:**
- ✅ Floor: beat the **base student** in-domain → proves the distillation worked
- 🎯 Target: beat **`Qwen2.5-Coder-1.5B-Instruct`** in-domain → proves specialization pays
- 🌙 Stretch: approach the **teacher** in-domain at ~1/20 the size

A negative result, well measured and well documented, is a success for this project. A model that "seems good" with no baselines is not.

---

## 6. Sources

- [Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) · [Qwen2.5-Coder-1.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B) · [Qwen2.5-Coder Technical Report](https://arxiv.org/pdf/2409.12186)
- [mlx-lm LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md) · [MLX on HuggingFace](https://huggingface.co/docs/hub/en/mlx)
- [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html) · [BigCode Evaluation Harness](https://github.com/bigcode-project/bigcode-evaluation-harness)
- [awesome-on-policy-distillation](https://github.com/chrisliu298/awesome-on-policy-distillation) · [Uni-OPD (2026)](https://arxiv.org/pdf/2605.03677) · [Supervision Fidelity Decay in OPD (2026)](https://arxiv.org/pdf/2605.30833)
