# Phase 0 — Environment & fundamentals

## Environment

| Tool | Version |
|---|---|
| Python (uv-managed) | 3.12.13 |
| MLX | 0.32.0 (runs on GPU, unified memory) |
| mlx-lm | 0.31.3 |
| transformers | 5.14.1 |
| huggingface_hub | 1.24.0 |
| datasets | 5.0.0 |
| uv | 0.11.29 |
| git-lfs | 3.7.1 |
| Ollama | 0.32.0 |

Hardware: MacBook Pro M5 Max, 64 GB unified memory, ~1.6 TB free.

`transformers` prints "PyTorch was not found" — expected and harmless. We only
use it for tokenizers/config; all tensor work goes through MLX.

## Tokenizer compatibility — the Phase 5 gate

Ran `scripts/check_tokenizers.py`. **Verdict: ALIGNABLE → Track B (logit-level KD) is viable.**

This is the key Phase 0 decision, and the result is favorable. Details:

- Teacher `Qwen3-Coder-30B-A3B-Instruct`: 151,669 vocab entries
- Student `Qwen2.5-Coder-1.5B`: 151,665 vocab entries
- **All 151,665 student tokens exist in the teacher with identical ids** (0 remapped).
- **id → token agrees across the entire shared range [0, 151665)**.
- **All 7 code probes tokenize identically** (0 mismatches), including
  indentation, f-strings, operators, SQL, and multi-byte UTF-8 (`café`, `你好`, `🌑`).
- Only difference: the teacher appends **4 control tokens** to the tail —
  `<think>`, `</think>`, `<tool_response>`, `</tool_response>` — plus a
  different `eos` (`<|im_end|>` vs `<|endoftext|>`).

### Why this matters

The reported `vocab_size` is 151,643 for *both*, but that number is a red
herring — the real vocabularies are 151,665 / 151,669. Trusting the reported
size would have been the exact trap the check exists to catch. What actually
governs logit-level KD is whether the logit vectors align on the shared
vocabulary, and here they align perfectly on the first 151,665 ids.

### Consequence for Phase 5

Logit-level KD is on the table (higher signal-per-token than sequence-level
SFT). The only reconciliation needed: **slice the teacher's logits to the first
151,665 entries** (or pad the student's output head by 4) so the two vectors
line up. The 4 dropped tokens are `<think>`/`<tool_response>` control tokens the
student never emits while writing code, so ignoring them is safe.

The on-policy fallback (Track A) is no longer forced — but it remains a valid
v2 experiment worth comparing against, since it attacks exposure bias, which
logit-KD does not.

## Domain decision (gates Phase 1+)

**Domain: pandas / polars data wrangling.** Chosen for the largest pool of
public seed data and trivial execution-based verification (run the transform,
assert on the resulting DataFrame). Safest choice for a first distillation run.

The model will get good at DataFrame manipulation and worse at everything else.
That trade-off is the whole point and will be measured, not hidden.

## Inference smoke tests (serving path validated)

| Model | Repo | Peak mem | Gen speed |
|---|---|---|---|
| Student | `Qwen/Qwen2.5-Coder-1.5B` (fp16) | 3.1 GB | 150 tok/s |
| Teacher | `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` | 17.3 GB | 62–67 tok/s |

Teacher output on a pandas probe (temp 0.0) — correct and clean:

```python
def remove_duplicates_keep_last(df):
    return df.drop_duplicates(keep='last').reset_index(drop=True)
```

The teacher's 67 tok/s at 17 GB is the MoE payoff: 30B of knowledge, 3.3B
active, generating fast enough to build a dataset overnight on a laptop. With
64 GB total there's ample headroom — an 8-bit teacher (~32 GB) is a viable
quality upgrade for Phase 2 if the 4-bit yield disappoints.

Note: the student base model completes raw code oddly without a proper prompt
(expected for a base, not instruct, model) — a concrete reminder of why the
instruct model is the rival to beat and why fine-tuning is the point.

## Phase 0 — DONE ✅

- [x] Environment (uv/MLX/git-lfs), all libs import and run on GPU
- [x] Tokenizer gate resolved → **ALIGNABLE, Track B (logit-KD) viable**
- [x] Student downloaded, inference validated, 150 tok/s
- [x] Teacher downloaded, inference validated, 67 tok/s
- [x] Domain chosen → **pandas / polars data wrangling**

Next: Phase 1 — build the eval harness and measure three baselines before training.
