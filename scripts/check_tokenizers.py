"""Phase 0 — Tokenizer compatibility check.

Logit-level knowledge distillation (Track B) requires the teacher and student
to share an *identical* token->id mapping: the loss compares logits[i] between
the two models position-by-position, so vocab entry i must mean the same token
for both. A matching `vocab_size` is necessary but NOT sufficient — Qwen3 added
control tokens on top of the Qwen2.5 vocabulary, so we verify the mapping
empirically rather than trusting the reported size.

This script downloads only the tokenizer files (a few KB), not the model
weights. Its verdict gates Phase 5:

    identical vocab   -> Track B viable (logit-level KD)
    otherwise         -> Track A / on-policy distillation fallback

Usage:
    uv run scripts/check_tokenizers.py
    uv run scripts/check_tokenizers.py --teacher <repo> --student <repo>
"""

from __future__ import annotations

import argparse
import sys

from transformers import AutoTokenizer

DEFAULT_TEACHER = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
DEFAULT_STUDENT = "Qwen/Qwen2.5-Coder-1.5B"

# Code-flavored probes: distillation happens over code, so alignment must hold
# on the tokens we actually train on — indentation, operators, f-strings,
# comments, and multi-byte content all exercise different merge rules.
PROBES = [
    "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n",
    "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})\n",
    "x = f'{value:.2f}'  # format to two decimals\n",
    "async def main() -> None:\n    await asyncio.gather(*tasks)\n",
    "SELECT id, name FROM users WHERE created_at > NOW() - INTERVAL '7 days';",
    "// UTF-8 check: café, naïve, 你好, 🌑\nconst π = 3.14159;",
    "\t\tif (a && b || !c) { return [1, 2, 3].map(x => x ** 2); }",
]


def section(title: str) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default=DEFAULT_TEACHER)
    parser.add_argument("--student", default=DEFAULT_STUDENT)
    args = parser.parse_args()

    section("Loading tokenizers (downloads tokenizer files only)")
    print(f"teacher: {args.teacher}")
    print(f"student: {args.student}")
    teacher = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    student = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)

    # --- Vocabulary size ---------------------------------------------------
    section("Vocabulary size")
    t_vocab = teacher.get_vocab()
    s_vocab = student.get_vocab()
    print(f"teacher vocab entries: {len(t_vocab)}")
    print(f"student vocab entries: {len(s_vocab)}")
    print(f"teacher reported vocab_size: {teacher.vocab_size}")
    print(f"student reported vocab_size: {student.vocab_size}")

    # --- Full token->id mapping identity -----------------------------------
    section("Token -> id mapping")
    same_size = len(t_vocab) == len(s_vocab)
    identical_mapping = t_vocab == s_vocab
    if identical_mapping:
        print("IDENTICAL — every token maps to the same id in both tokenizers.")
    else:
        only_teacher = set(t_vocab) - set(s_vocab)
        only_student = set(s_vocab) - set(t_vocab)
        shared = set(t_vocab) & set(s_vocab)
        remapped = sum(1 for tok in shared if t_vocab[tok] != s_vocab[tok])
        print(f"shared tokens:           {len(shared)}")
        print(f"tokens only in teacher:  {len(only_teacher)}")
        print(f"tokens only in student:  {len(only_student)}")
        print(f"shared tokens w/ different id (remapped): {remapped}")
        if only_teacher:
            sample = sorted(only_teacher)[:10]
            print(f"  e.g. teacher-only: {sample}")
        if only_student:
            sample = sorted(only_student)[:10]
            print(f"  e.g. student-only: {sample}")

    # --- Special / added tokens -------------------------------------------
    section("Special tokens")
    for name, tok in [
        ("bos", (teacher.bos_token, student.bos_token)),
        ("eos", (teacher.eos_token, student.eos_token)),
        ("pad", (teacher.pad_token, student.pad_token)),
        ("unk", (teacher.unk_token, student.unk_token)),
    ]:
        mark = "OK " if tok[0] == tok[1] else "DIFF"
        print(f"  [{mark}] {name}: teacher={tok[0]!r}  student={tok[1]!r}")

    # --- Logit alignment on the overlapping id range -----------------------
    # This is the criterion that actually governs logit-level KD: over the
    # id range both models share, does id -> token agree? If so, their logit
    # vectors are aligned on that slice and KD works there, regardless of any
    # extra tokens appended to the tail of the larger vocabulary.
    section("Logit alignment (id -> token over the shared id range)")
    t_ids_to_tok = {i: tok for tok, i in t_vocab.items()}
    s_ids_to_tok = {i: tok for tok, i in s_vocab.items()}
    min_vocab = min(max(t_ids_to_tok) + 1, max(s_ids_to_tok) + 1)
    id_range_aligned = all(
        t_ids_to_tok.get(i) == s_ids_to_tok.get(i) for i in range(min_vocab)
    )
    print(f"overlapping id range: [0, {min_vocab})")
    print(f"id -> token agrees across the whole range: {id_range_aligned}")

    # --- Empirical id-level identity on code probes ------------------------
    section("Encoding probes (id-level identity on real code)")
    mismatches = 0
    for i, probe in enumerate(PROBES):
        t_ids = teacher.encode(probe, add_special_tokens=False)
        s_ids = student.encode(probe, add_special_tokens=False)
        ok = t_ids == s_ids
        mismatches += not ok
        label = probe.splitlines()[0][:44].replace("\t", "\\t")
        print(f"  [{'OK ' if ok else 'DIFF'}] probe {i}: {label!r}")
        if not ok:
            print(f"         teacher: {t_ids}")
            print(f"         student: {s_ids}")

    # --- Verdict -----------------------------------------------------------
    # Three outcomes, not two. What logit-level KD actually needs is aligned
    # logit vectors on the shared vocabulary, NOT byte-identical tokenizers.
    #   IDENTICAL  : vocabs match exactly            -> Track B, no reconciliation
    #   ALIGNABLE  : shared id range aligns, only    -> Track B, reconcile the
    #                tail/special tokens differ         vocab tail (trivial)
    #   DIVERGENT  : ids remapped or code probes      -> Track A / on-policy
    #                disagree                            fallback
    section("VERDICT")
    if identical_mapping and mismatches == 0:
        verdict, track = "IDENTICAL", "B"
        print("Tokenizers are byte-identical.")
        print("Phase 5: logit-level KD directly (KL over teacher logits).")
    elif id_range_aligned and remapped == 0 and mismatches == 0:
        verdict, track = "ALIGNABLE", "B"
        extra = len(t_vocab) - min_vocab if len(t_vocab) > min_vocab else 0
        print("Shared id range is fully aligned; only the vocabulary tail differs")
        print(f"({extra} extra token(s) on the larger side, e.g. control tokens).")
        print("Phase 5: logit-level KD IS viable — slice/pad logits to the shared")
        print("range so the two logit vectors line up. The extra tail tokens are")
        print("ones the student never emits in code, so ignoring them is safe.")
    else:
        verdict, track = "DIVERGENT", "A"
        print("Tokenizers genuinely diverge (ids remapped or code probes differ).")
        print("Phase 5 falls back to on-policy distillation (no shared tokenizer")
        print("required): the student generates, the teacher scores/corrects.")
        if same_size and not identical_mapping:
            print("\nNote: vocab sizes match but the mapping does not — exactly the")
            print("trap this check exists to catch. Do NOT assume alignment.")

    print(f"\n  verdict            : {verdict}")
    print(f"  identical mapping  : {identical_mapping}")
    print(f"  shared-range align : {id_range_aligned}  (remapped ids: {remapped})")
    print(f"  probe mismatches   : {mismatches}/{len(PROBES)}")
    print(f"  -> TRACK {track} " + ("(logit-level KD)" if track == "B" else "(on-policy fallback)"))

    return 0 if track == "B" else 1


if __name__ == "__main__":
    sys.exit(main())
