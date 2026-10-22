"""Robustness evals: does the assistant still work when the question is phrased differently?

The golden set's questions are clean and carefully worded. Real users type
"whats kestrels revnue for 2025" or "Q1 sales?". A pass rate only means
something if it survives that.

Each golden question is asked in several STYLES:

  original   the golden question, unchanged (the baseline, asked in the same run
             so drift between days can't masquerade as a robustness problem)
  reworded   hand-written: same meaning, different vocabulary ("sales" for
             "revenue", "head office" for "headquartered")
  casual     hand-written: short and informal, the way people type into a box
  typo       generated: one realistic typo in one word
  typos3     generated: typos in up to three words
  lower      generated: lowercase, no punctuation, no apostrophes ("whats kestrels")

Hand-written variants live in data/robustness/paraphrases_v1.jsonl. Generated
ones are made here, from a seed derived from the case id, so the same typos
come out on every run and on every machine (Python's built-in hash() changes
between runs, so it isn't used).

What's measured, per variant:
  passed        the golden-set rule grader, using the ORIGINAL case's expected
                answer (rewording must not change what the right answer is)
  passage_hit   did the retriever still fetch the paragraph with the answer?

Then, per style, compared with the original question in the same run:
  flip          the original passed and this variant failed. The headline number.
  flip cause    "retrieval" if the original's passages contained the answer and
                the variant's didn't (the search step lost it), otherwise "model"
                (the model had the right paragraph and still got it wrong).
                This split matters because the fixes are completely different.
  gained        the original failed and this variant passed (it happens: g022's
                original wording is a known retrieval miss)

Why the split is the point: the retriever is BM25, which matches whole words
exactly. "revnue" shares nothing with "revenue", and "kestrels" doesn't match
"kestrel". So a typo can make the right paragraph disappear before the model
ever sees the question. Without passage_hit, that would be blamed on the model,
the same mistake Run 2 caught for g022.

Out-of-scope cases are included: a reworded version of a question that can't be
answered should still be refused. The ambiguous case (g032) is left out, because
rewording an ambiguous question changes how ambiguous it is.
"""
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from app.retriever import STOPWORDS
from evals.metrics import passage_hit, retrieval_hit, score_case

HAND_STYLES = ("reworded", "casual")
GENERATED_STYLES = ("typo", "typos3", "lower")
STYLES = ("original",) + HAND_STYLES + GENERATED_STYLES
SKIP_CATEGORIES = {"ambiguous"}

# Keys next to each other on a QWERTY keyboard, for "fat finger" typos.
_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
NEIGHBORS = {}
for _row in _ROWS:
    for _i, _ch in enumerate(_row):
        NEIGHBORS[_ch] = (_row[_i - 1] if _i > 0 else "") + (_row[_i + 1] if _i < len(_row) - 1 else "")

TYPO_OPS = ("swap", "drop", "double", "neighbor")

# A word that can get a typo: letters only, at least 4 long. Numbers, years and
# short words like "Q1" are never touched, because changing them changes the
# MEANING of the question, not just its spelling.
WORD_RE = re.compile(r"[A-Za-z]{4,}")

# "Kestrel" is in every passage (each chunk starts with the document title), so
# BM25 gives it almost no weight: a typo in it changes nothing about retrieval.
# The first version of this generator put 8 of 27 single typos there, which made
# the "typo" style look more robust than it is. Typos go in words that matter.
NO_TYPO_WORDS = {"kestrel"}


class VariantDatasetError(ValueError):
    pass


# ---- Generating variants --------------------------------------------------

def seeded_rng(*parts: str) -> random.Random:
    """A random generator that gives the same sequence for the same inputs, everywhere."""
    digest = hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def typo_word(word: str, op: str, rng: random.Random) -> str:
    """Apply one typo to one word. The first letter is never changed: people
    rarely mistype it, and keeping it makes the result look like a real typo."""
    i = rng.randrange(1, len(word))   # a position after the first letter
    if op == "swap":                  # "revenue" -> "reevnue"
        i = min(i, len(word) - 2)
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "drop":                  # "revenue" -> "revnue"
        return word[:i] + word[i + 1:]
    if op == "double":                # "revenue" -> "revvenue"
        return word[:i] + word[i] + word[i:]
    # "neighbor": hit the key next to the right one. "revenue" -> "revenur"
    ch = word[i]
    options = NEIGHBORS.get(ch.lower(), "")
    if not options:
        return word[:i] + word[i + 1:]
    new = rng.choice(options)
    return word[:i] + (new.upper() if ch.isupper() else new) + word[i + 1:]


def add_typos(question: str, count: int, rng: random.Random) -> str:
    """Put a typo in `count` different eligible words (fewer if there aren't enough)."""
    words = [m for m in WORD_RE.finditer(question) if m.group().lower() not in STOPWORDS]
    spans = [m.span() for m in words if m.group().lower() not in NO_TYPO_WORDS]
    if not spans:
        # "Who is Kestrel's CEO?" has no other eligible word, and a "typo"
        # variant identical to the original would test nothing.
        spans = [m.span() for m in words]
    chosen = sorted(rng.sample(spans, min(count, len(spans))), reverse=True)
    text = question
    # Edit from the end of the string backwards, so earlier positions stay valid.
    for start, end in chosen:
        word = text[start:end]
        for _ in range(10):  # some ops can produce the same word ("ll" -> swap); retry
            new = typo_word(word, rng.choice(TYPO_OPS), rng)
            if new != word:
                break
        text = text[:start] + new + text[end:]
    return text


def lowercase_variant(question: str) -> str:
    """How people type in a hurry: lowercase, no apostrophes, no trailing punctuation.
    Decimal points and $ / % signs are kept, since they carry meaning."""
    text = question.lower().replace("'", "").replace("’", "")
    text = re.sub(r"[?!,;:]", "", text)
    return " ".join(text.split())


def generate_variant(case: dict, style: str) -> str:
    rng = seeded_rng(case["id"], style)
    if style == "typo":
        return add_typos(case["question"], 1, rng)
    if style == "typos3":
        return add_typos(case["question"], 3, rng)
    if style == "lower":
        return lowercase_variant(case["question"])
    raise ValueError(f"not a generated style: {style!r}")


# ---- Dataset ----------------------------------------------------------------

def load_hand_variants(path: Path, golden: list[dict]) -> list[dict]:
    """Read the hand-written variants and fail loudly on anything malformed."""
    golden_ids = {c["id"]: c for c in golden}
    variants, seen = [], set()
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        v = json.loads(line)

        def fail(msg):
            raise VariantDatasetError(f"Line {line_no} ({v.get('id', '?')}): {msg}")

        missing = {"id", "source_id", "style", "question"} - v.keys()
        if missing:
            fail(f"missing fields {sorted(missing)}")
        if v["id"] in seen:
            fail("duplicate id")
        if v["style"] not in HAND_STYLES:
            fail(f"style must be one of {HAND_STYLES}")
        if v["source_id"] not in golden_ids:
            fail(f"source_id {v['source_id']!r} is not in the golden set")
        if golden_ids[v["source_id"]]["category"] in SKIP_CATEGORIES:
            fail(f"{v['source_id']} is {golden_ids[v['source_id']]['category']}, which is skipped")
        if v["question"].strip() == golden_ids[v["source_id"]]["question"].strip():
            fail("identical to the original question")
        seen.add(v["id"])
        variants.append(v)
    return variants


def build_variants(golden: list[dict], hand: list[dict], styles=STYLES) -> list[dict]:
    """Every (case, style) pair to ask, originals first within each case."""
    by_source = defaultdict(list)
    for v in hand:
        by_source[v["source_id"]].append(v)
    out = []
    for case in golden:
        if case["category"] in SKIP_CATEGORIES:
            continue
        for style in styles:
            if style == "original":
                out.append({"id": case["id"], "source_id": case["id"], "style": "original",
                            "question": case["question"]})
            elif style in HAND_STYLES:
                out.extend(v for v in by_source[case["id"]] if v["style"] == style)
            else:
                out.append({"id": f"{case['id']}-{style}", "source_id": case["id"], "style": style,
                            "question": generate_variant(case, style)})
    return out


# ---- Running ------------------------------------------------------------------

def _base_row(variant: dict, case: dict) -> dict:
    return {"id": variant["id"], "source_id": case["id"], "style": variant["style"],
            "category": case["category"], "question": variant["question"]}


def retrieval_row(variant: dict, case: dict, retriever, k: int = 4) -> dict:
    """Retrieval only: no API call. Used by `robustness.py --retrieval-only`."""
    chunks = retriever.search(variant["question"], k=k)
    sources = sorted({c.doc for c in chunks})
    return {**_base_row(variant, case),
            "retrieval_hit": retrieval_hit(case, sources),
            "passage_hit": passage_hit(case, [c.text for c in chunks]),
            "retrieved_sources": sources,
            "answer": None, "passed": None, "reason": None}


def run_variant(variant: dict, case: dict, assistant) -> dict:
    """Ask one variant and grade it against the ORIGINAL case's expected answer."""
    result = assistant.answer(variant["question"])
    rule = score_case(case, result.answer)
    return {**_base_row(variant, case),
            "retrieval_hit": retrieval_hit(case, result.sources),
            "passage_hit": passage_hit(case, result.contexts),
            "retrieved_sources": result.sources,
            "answer": result.answer,
            "passed": rule.passed,
            "reason": rule.reason,
            "latency_s": round(result.llm.latency_s, 3),
            "input_tokens": result.llm.input_tokens,
            "output_tokens": result.llm.output_tokens}


# ---- Summary --------------------------------------------------------------------

def _rate(flags) -> float | None:
    known = [f for f in flags if f is not None]
    return round(sum(known) / len(known), 3) if known else None


def flip_cause(original: dict, variant: dict) -> str:
    """Why did a variant fail when its original passed?

    retrieval: the original's passages had the answer and the variant's didn't.
    model:     anything else, including out-of-scope cases (nothing to retrieve).
    """
    if original["passage_hit"] is True and variant["passage_hit"] is False:
        return "retrieval"
    return "model"


def summarize_robustness(rows: list[dict]) -> dict:
    """Per-style numbers, each compared with the original question from the same run."""
    originals = {r["source_id"]: r for r in rows if r["style"] == "original"}
    graded = any(r["passed"] is not None for r in rows)
    by_style = {}
    for style in STYLES:
        group = [r for r in rows if r["style"] == style]
        if not group:
            continue
        entry = {
            "variants": len(group),
            "passage_hit_rate": _rate([r["passage_hit"] for r in group]),
            "passage_misses": [r["id"] for r in group if r["passage_hit"] is False],
        }
        if style != "original":
            # Passages lost: the original found the answer's paragraph, this one didn't.
            entry["passages_lost"] = [
                r["id"] for r in group
                if r["passage_hit"] is False and originals.get(r["source_id"], {}).get("passage_hit") is True]
        if graded:
            entry["pass_rate"] = _rate([r["passed"] for r in group])
            if style != "original":
                comparable = [r for r in group if r["source_id"] in originals]
                held = [r for r in comparable if originals[r["source_id"]]["passed"]]
                flips = [r for r in held if not r["passed"]]
                entry.update({
                    "original_passed": len(held),
                    # Of the questions that passed in their original wording,
                    # the share that still pass in this style. 1.0 = fully robust.
                    "robustness": round((len(held) - len(flips)) / len(held), 3) if held else None,
                    "flips": [r["id"] for r in flips],
                    "flips_retrieval": [r["id"] for r in flips
                                        if flip_cause(originals[r["source_id"]], r) == "retrieval"],
                    "flips_model": [r["id"] for r in flips
                                    if flip_cause(originals[r["source_id"]], r) == "model"],
                    "gained": [r["id"] for r in comparable
                               if not originals[r["source_id"]]["passed"] and r["passed"]],
                })
        by_style[style] = entry

    variant_styles = [s for s in by_style if s != "original"]
    summary = {"questions": len(originals) or len({r["source_id"] for r in rows}),
               "variants": sum(1 for r in rows if r["style"] != "original"),
               "by_style": by_style}
    if graded:
        held = sum(by_style[s]["original_passed"] for s in variant_styles)
        flips = sum(len(by_style[s]["flips"]) for s in variant_styles)
        summary.update({
            "overall_robustness": round((held - flips) / held, 3) if held else None,
            "flips": flips,
            "flips_retrieval": sum(len(by_style[s]["flips_retrieval"]) for s in variant_styles),
            "flips_model": sum(len(by_style[s]["flips_model"]) for s in variant_styles),
        })
    return summary
