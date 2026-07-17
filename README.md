<div align="center">

# 🌑 deep-moon

**A small language model that writes code, distilled on a laptop — with every step shown.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Status: Phase 1](https://img.shields.io/badge/Status-Phase_1_of_7-orange.svg)](PLAN.md)
[![Results: none yet](https://img.shields.io/badge/Results-none_yet-lightgrey.svg)](#results)
[![Built with MLX](https://img.shields.io/badge/Built_with-MLX-black.svg)](https://github.com/ml-explore/mlx)

*No GPU cluster. No proprietary data. No unreproducible numbers.*

[The thesis](#the-thesis) · [How it works](#how-it-works) · [Results](#results) · [Reproduce it](#reproduce-it) · [Plan](PLAN.md)

</div>

---

## The thesis

> **Small models don't fail because they're small. They fail because we ask them to be everything.**

The last few years have been an argument about scale. The more interesting argument is about *scope*.

A 1.5B-parameter model asked to write Rust, explain Kubernetes, and refactor legacy COBOL will lose to a 30B model at all three. The same 1.5B model, aimed at **one narrow domain** and taught by a teacher that has already mastered it, is a different proposition entirely — small enough to run on the machine you already own, fast enough to sit inside your editor's keystroke loop, and good enough at the one thing you actually asked for.

That's not a hunch. It's the mechanism behind every distilled model that punched above its weight. `deep-moon` is an attempt to run that mechanism end to end, in the open, on hardware anyone can buy.

**What makes this different is not the model. It's that every step is visible** — including the parts that don't work.

---

## What this actually is

An **open, reproducible pipeline** that takes a 30B teacher and a 1.5B student and produces a specialized code model, running entirely on a single Apple Silicon laptop.

| | |
|---|---|
| 🎓 **Teacher** | [`Qwen3-Coder-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) — 30B total, **3.3B active** (MoE), Apache 2.0 |
| 🎒 **Student** | [`Qwen2.5-Coder-1.5B`](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B) — Apache 2.0 |
| 🖥️ **Hardware** | One MacBook Pro (M5 Max, 64 GB). That's the whole budget. |
| ⚙️ **Stack** | [MLX](https://github.com/ml-explore/mlx) to train · [EvalPlus](https://github.com/evalplus/evalplus) to measure · [Ollama](https://ollama.com) to serve |
| 📜 **License** | Apache 2.0, top to bottom — weights, data, and code |

The MoE teacher is the trick that makes this fit on a laptop: 30B of knowledge, but only 3.3B activated per token, so it generates a full training set overnight instead of over a week.

---

## How it works

```
   seed prompts
        │
        ▼
  ┌───────────┐   generates solutions + tests
  │  TEACHER  │   Qwen3-Coder-30B-A3B
  └─────┬─────┘
        │
        ▼
  ┌───────────────────────────────┐
  │  REJECTION SAMPLING           │   ← run the tests. keep what passes.
  │  execute → verify → discard   │      this is the whole ballgame.
  └─────┬─────────────────────────┘
        │
        ▼
  ~10-20k verified examples
        │
        ▼
  ┌───────────┐   LoRA / full FT / logit-KD
  │  STUDENT  │   Qwen2.5-Coder-1.5B
  └─────┬─────┘
        │
        ▼
  ┌───────────────────────────────┐
  │  EVALUATION vs 3 BASELINES    │   ← built BEFORE training. on purpose.
  └─────┬─────────────────────────┘
        │
        ▼
  quantize → GGUF → Ollama → 🤗
```

### The three ideas doing the real work

**1. Verify by execution, not by vibes.**
Teachers hallucinate. A teacher's confident wrong answer is still a wrong answer, and naive distillation copies it faithfully. So every generated example gets its tests *run*. What fails gets deleted. This single filter is the difference between distilling competence and distilling confidence.

**2. Build the evaluation before the model.**
The most common way these projects fail isn't a bad learning rate — it's finishing a training run with nothing to compare against. `deep-moon` builds the eval harness in Phase 1, before a single gradient step, and measures **three** baselines: the base student (the floor), the teacher (the ceiling), and the *instruct* model of identical size — the honest rival. Any claim that skips that third number isn't a claim, it's marketing.

**3. Know which distillation you're doing.**
Most "distillation" is SFT on teacher outputs: the student sees the token the teacher picked, and nothing else — no confidence, no alternatives. Real logit-level KD transfers the teacher's *entire distribution* and needs far less data. It also demands aligned tokenizers. Qwen2.5 and Qwen3 both *report* `vocab_size = 151643`, but that number is a red herring — their real vocabularies are 151,665 / 151,669. We didn't assume; [Phase 0 measured it](scripts/check_tokenizers.py): every student token maps to the same id in the teacher, and the only difference is 4 control tokens on the teacher's tail. **Verdict: logit-level KD is viable.** The high-signal path is open.

---

## Results

**There are none yet.** This project is at Phase 1 of 7 — the eval harness is being built *before* any training, on purpose. This table is a contract about what will be reported — not a placeholder for numbers we hope to see.

| Model | Size | In-domain | HumanEval+ | MBPP+ |
|---|---|---|---|---|
| Teacher — `Qwen3-Coder-30B-A3B` | 30B (3.3B active) | *pending* | *pending* | *pending* |
| **`deep-moon-1.5b`** | **1.5B** | *pending* | *pending* | *pending* |
| Rival — `Qwen2.5-Coder-1.5B-Instruct` | 1.5B | *pending* | *pending* | *pending* |
| Floor — `Qwen2.5-Coder-1.5B` (base) | 1.5B | *pending* | *pending* | *pending* |

### What we're honestly aiming for

- ✅ **Floor** — beat the base student in-domain. Proves the distillation did *something*.
- 🎯 **Target** — beat `Qwen2.5-Coder-1.5B-Instruct` in-domain. Proves specialization pays for itself.
- 🌙 **Stretch** — approach the teacher in-domain at ~1/20 the size.

### What we're *not* claiming

**This will not beat `Qwen2.5-Coder-1.5B-Instruct` at general-purpose code, and we're not going to try.** That model is itself a distillation, built by Alibaba with thousands of GPU-hours and proprietary data, from the same base we start from. A laptop does not close that gap, and any project telling you otherwise is either fooling you or itself.

It will also **get worse at everything outside its domain.** That's catastrophic forgetting; it's expected, it will be measured, and it will be printed in the model card next to the wins. A number without its cost isn't a result.

---

## Reproduce it

> ⚠️ Early days. `check_tokenizers.py` is real and runnable today; the later scripts are the target interface, not yet shipped. Follow [PLAN.md](PLAN.md) for current status.

```bash
# Requires: Apple Silicon, ~32 GB+ unified memory, uv, git-lfs
git clone https://github.com/<you>/deep-moon && cd deep-moon
uv sync

# Phase 0 — does logit-level KD even work with this pair?
uv run scripts/check_tokenizers.py

# Phase 1 — measure the world before you change it
uv run scripts/evaluate.py --baselines

# Phase 2 — teacher generates, execution decides
uv run scripts/generate_data.py --n 20000
uv run scripts/verify_data.py --execute --decontaminate

# Phase 3 — distill
uv run scripts/train_lora.py

# Phase 4 — did it actually work?
uv run scripts/evaluate.py --model models/deep-moon-1.5b --compare-baselines
```

---

## Why "deep-moon"

A moon is small, and it only exists in relation to something much larger. It doesn't try to be the planet. It just needs to be very good at the one orbit it has.

---

## Project status

**Phase 1 of 7.** The full roadmap, with the reasoning behind each decision, is in **[PLAN.md](PLAN.md)**.

- [x] Plan and architecture decisions
- [x] License audit — Apache 2.0 confirmed clean for distillation *and* redistribution
- [x] Phase 0 — environment (uv/MLX), tokenizer gate → **logit-level KD viable**; both models running locally (student 150 tok/s, teacher 67 tok/s)
- [ ] Phase 1 — domain fixed (**pandas/polars**); eval harness + three baselines next
- [ ] Phase 2 — teacher data generation + rejection sampling
- [ ] Phase 3 — distillation v1 (sequence-level)
- [ ] Phase 4 — evaluation v1
- [ ] Phase 5 — distillation v2 (logit-level or on-policy)
- [ ] Phase 6 — quantization, GGUF, Ollama
- [ ] Phase 7 — publish to HuggingFace

---

## Credits

Standing on: [Qwen](https://huggingface.co/Qwen) (teacher and student, Apache 2.0) · [Apple MLX](https://github.com/ml-explore/mlx) · [EvalPlus](https://github.com/evalplus/evalplus) · [BigCode](https://github.com/bigcode-project/bigcode-evaluation-harness) · [Ollama](https://ollama.com)

Everything here — weights, dataset, and code — ships **Apache 2.0**. Take it, fork it, beat it. If you beat it, please tell us how.

<div align="center">

**Built in the open. Measured honestly. Published either way.**

</div>
