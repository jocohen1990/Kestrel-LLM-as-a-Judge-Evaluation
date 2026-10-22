"""Offline tests for the robustness evals.

Three groups:
  1. The typo generator: realistic, repeatable, and never changes the meaning
     (numbers and years are left alone).
  2. The dataset: every hand-written variant points at a real golden case, and
     every generated variant actually differs from its original.
  3. Grading and the summary: flips, and whether each flip is blamed on
     retrieval or on the model.
"""
import json
from types import SimpleNamespace

import pytest

from app.llm import LLMResponse
from app.paths import CORPUS_DIR, GOLDEN_DIR, ROBUSTNESS_DIR
from app.rag import RagAssistant
from evals.dataset import load_golden
from evals.robustness import (
    GENERATED_STYLES,
    HAND_STYLES,
    NEIGHBORS,
    SKIP_CATEGORIES,
    STYLES,
    VariantDatasetError,
    add_typos,
    build_variants,
    flip_cause,
    generate_variant,
    load_hand_variants,
    lowercase_variant,
    retrieval_row,
    run_variant,
    seeded_rng,
    summarize_robustness,
    typo_word,
)

GOLDEN = load_golden(GOLDEN_DIR / "golden_v3.jsonl")
HAND = load_hand_variants(ROBUSTNESS_DIR / "paraphrases_v1.jsonl", GOLDEN)
ELIGIBLE = [c for c in GOLDEN if c["category"] not in SKIP_CATEGORIES]


# ---- 1. Typo generator ------------------------------------------------------

def test_same_inputs_give_same_random_sequence():
    assert seeded_rng("g001", "typo").random() == seeded_rng("g001", "typo").random()
    assert seeded_rng("g001", "typo").random() != seeded_rng("g002", "typo").random()


@pytest.mark.parametrize("op", ["swap", "drop", "double", "neighbor"])
def test_typo_keeps_first_letter_and_changes_length_as_expected(op):
    for seed in range(30):
        new = typo_word("revenue", op, seeded_rng(str(seed)))
        assert new[0] == "r"
        expected_len = {"drop": 6, "double": 8}.get(op, 7)
        assert len(new) == expected_len


def test_swap_keeps_the_same_letters():
    for seed in range(30):
        assert sorted(typo_word("margin", "swap", seeded_rng(str(seed)))) == sorted("margin")


def test_neighbor_typo_uses_an_adjacent_key():
    for seed in range(30):
        new = typo_word("revenue", "neighbor", seeded_rng(str(seed)))
        diffs = [(a, b) for a, b in zip("revenue", new) if a != b]
        assert len(diffs) == 1
        right, typed = diffs[0]
        assert typed in NEIGHBORS[right]


def test_add_typos_changes_the_requested_number_of_words():
    q = "What was the gross margin for fiscal year 2025?"
    out = add_typos(q, 3, seeded_rng("x"))
    changed = [a for a, b in zip(q.split(), out.split()) if a != b]
    assert len(changed) == 3


def test_numbers_and_years_are_never_changed():
    # Changing "2025" to "2052" would change what the question asks.
    for case in GOLDEN:
        for style in ("typo", "typos3"):
            variant = generate_variant(case, style)
            assert [w for w in variant.split() if any(ch.isdigit() for ch in w)] == \
                   [w for w in case["question"].split() if any(ch.isdigit() for ch in w)]


def test_company_name_is_skipped_when_other_words_exist():
    out = add_typos("What was Kestrel's revenue?", 1, seeded_rng("y"))
    assert "Kestrel's" in out and "revenue" not in out


def test_company_name_is_used_when_nothing_else_is_eligible():
    # "CEO" is too short for a typo, so the only candidate is "Kestrel".
    out = add_typos("Who is Kestrel's CEO?", 1, seeded_rng("z"))
    assert out != "Who is Kestrel's CEO?" and out.endswith("'s CEO?")


def test_lowercase_variant():
    assert lowercase_variant("What was Kestrel's revenue in Q1 2026?") == \
        "what was kestrels revenue in q1 2026"
    assert lowercase_variant("EPS of $0.38, up 5%?") == "eps of $0.38 up 5%"


def test_every_generated_variant_differs_from_its_original():
    for case in ELIGIBLE:
        for style in GENERATED_STYLES:
            assert generate_variant(case, style) != case["question"], (case["id"], style)


def test_generated_variants_are_repeatable():
    first = [generate_variant(c, s) for c in ELIGIBLE for s in GENERATED_STYLES]
    second = [generate_variant(c, s) for c in ELIGIBLE for s in GENERATED_STYLES]
    assert first == second


# ---- 2. Dataset -------------------------------------------------------------

def test_every_eligible_case_has_one_variant_per_hand_style():
    for case in ELIGIBLE:
        styles = sorted(v["style"] for v in HAND if v["source_id"] == case["id"])
        assert styles == sorted(HAND_STYLES), case["id"]


def test_ambiguous_case_is_skipped():
    assert not [v for v in HAND if v["source_id"] == "g032"]
    ids = {v["source_id"] for v in build_variants(GOLDEN, HAND)}
    assert "g032" not in ids


def test_build_variants_covers_every_case_and_style():
    variants = build_variants(GOLDEN, HAND)
    assert len(variants) == len(ELIGIBLE) * len(STYLES)
    assert len({v["id"] for v in variants}) == len(variants)


def test_build_variants_filters_styles():
    variants = build_variants(GOLDEN, HAND, styles=("original", "typo"))
    assert {v["style"] for v in variants} == {"original", "typo"}


def _write(tmp_path, *rows):
    path = tmp_path / "v.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


@pytest.mark.parametrize("row, message", [
    ({"id": "x", "source_id": "g001", "style": "typo", "question": "q?"}, "style"),
    ({"id": "x", "source_id": "g999", "style": "casual", "question": "q?"}, "not in the golden"),
    ({"id": "x", "source_id": "g032", "style": "casual", "question": "q?"}, "skipped"),
    ({"id": "x", "source_id": "g001", "style": "casual",
      "question": "What was Kestrel's total revenue for fiscal year 2025?"}, "identical"),
    ({"id": "x", "source_id": "g001", "style": "casual"}, "missing"),
])
def test_loader_rejects_bad_rows(tmp_path, row, message):
    with pytest.raises(VariantDatasetError, match=message):
        load_hand_variants(_write(tmp_path, row), GOLDEN)


def test_loader_rejects_duplicate_ids(tmp_path):
    row = {"id": "x", "source_id": "g001", "style": "casual", "question": "FY25 sales?"}
    with pytest.raises(VariantDatasetError, match="duplicate"):
        load_hand_variants(_write(tmp_path, row, row), GOLDEN)


# ---- 3. Grading and summary -------------------------------------------------

CASE = next(c for c in GOLDEN if c["id"] == "g006")   # Q1 2026 revenue: $108.3M


class FakeAssistant:
    def __init__(self, answer, contexts):
        self._answer, self._contexts = answer, contexts

    def answer(self, question):
        llm = LLMResponse(text=self._answer, model="fake", input_tokens=5,
                          output_tokens=2, latency_s=0.1)
        return SimpleNamespace(answer=self._answer, sources=["q1_2026_earnings.md"],
                               contexts=self._contexts, llm=llm)


def test_run_variant_grades_against_the_original_case():
    variant = {"id": "g006-casual", "source_id": "g006", "style": "casual",
               "question": "Q1 2026 sales for Kestrel?"}
    row = run_variant(variant, CASE, FakeAssistant("It was $108.3 million.", ["... $108.3 ..."]))
    assert row["passed"] is True and row["passage_hit"] is True
    assert row["question"] == "Q1 2026 sales for Kestrel?"

    row = run_variant(variant, CASE, FakeAssistant("I don't know based on the provided documents.",
                                                    ["unrelated passage"]))
    assert row["passed"] is False and row["passage_hit"] is False


def test_retrieval_row_makes_no_model_call():
    row = retrieval_row({"id": "g006", "source_id": "g006", "style": "original",
                         "question": CASE["question"]}, CASE, RagAssistant(retriever_version="r1"))
    assert row["passage_hit"] is True and row["passed"] is None and row["answer"] is None
    assert row["search_query"] == CASE["question"] and row["corrections"] == {}


def test_original_questions_retrieve_their_paragraph():
    # Baseline for every comparison: only g022, the known Run 2 miss, fails at k=4.
    # If this changes, the retriever changed and old robustness results aren't comparable.
    r1 = RagAssistant(retriever_version="r1")
    misses = [c["id"] for c in ELIGIBLE
              if retrieval_row({"id": c["id"], "source_id": c["id"], "style": "original",
                                "question": c["question"]}, c, r1)["passage_hit"] is False]
    assert misses == ["g022"]


def row(id_, style, source, passed, hit):
    return {"id": id_, "style": style, "source_id": source, "passed": passed, "passage_hit": hit}


def test_flip_cause():
    assert flip_cause(row("a", "original", "a", True, True), row("a-t", "typo", "a", False, False)) == "retrieval"
    assert flip_cause(row("a", "original", "a", True, True), row("a-t", "typo", "a", False, True)) == "model"
    # Out of scope: nothing to retrieve, so it can only be the model.
    assert flip_cause(row("a", "original", "a", True, None), row("a-t", "typo", "a", False, None)) == "model"


def test_summary_counts_flips_by_cause_and_gains():
    rows = [
        row("a", "original", "a", True, True), row("a-typo", "typo", "a", False, False),
        row("b", "original", "b", True, True), row("b-typo", "typo", "b", False, True),
        row("c", "original", "c", True, True), row("c-typo", "typo", "c", True, True),
        row("d", "original", "d", False, False), row("d-typo", "typo", "d", True, True),
    ]
    s = summarize_robustness(rows)
    typo = s["by_style"]["typo"]
    assert typo["original_passed"] == 3
    assert typo["flips"] == ["a-typo", "b-typo"]
    assert typo["flips_retrieval"] == ["a-typo"] and typo["flips_model"] == ["b-typo"]
    assert typo["gained"] == ["d-typo"]
    assert typo["robustness"] == round(1 / 3, 3)
    assert typo["passages_lost"] == ["a-typo"]
    assert s["flips"] == 2 and s["flips_retrieval"] == 1 and s["flips_model"] == 1
    assert s["overall_robustness"] == round(1 / 3, 3)


def test_summary_in_retrieval_only_mode_has_no_pass_numbers():
    rows = [row("a", "original", "a", None, True), row("a-typo", "typo", "a", None, False)]
    s = summarize_robustness(rows)
    assert "overall_robustness" not in s
    assert s["by_style"]["typo"]["passages_lost"] == ["a-typo"]
