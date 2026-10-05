"""Unit tests for the grading functions.

If the grader is wrong, every score it produces is wrong, so the metrics
get tested like any other code. These run offline in milliseconds.
"""
import pytest

from evals.metrics import extract_numbers, is_refusal, number_present, score_case


@pytest.mark.parametrize("text, expected", [
    ("Revenue was $108.3 million.", [108.3]),
    ("The company had 1,240 employees.", [1240.0]),
    ("Diluted EPS was $0.38.", [0.38]),
    ("Guidance is $112-115 million.", [112.0, 115.0]),   # dash is not a minus sign
    ("Revenue of $412,600,000.", [412600000.0]),
    ("In February 2026, it launched.", [2026.0]),          # trailing comma isn't part of it
    ("Values 3, 4 and 5.", [3.0, 4.0, 5.0]),
    ("No numbers here.", []),
])
def test_extract_numbers(text, expected):
    assert extract_numbers(text) == expected


def test_number_present_handles_different_units():
    assert number_present(412.6, "Revenue was $412.6 million.")
    assert number_present(412.6, "Revenue was $412,600,000.")
    assert not number_present(412.6, "Revenue was $349.7 million.")


@pytest.mark.parametrize("text", [
    "I don't know based on the provided documents.",
    "I don’t know based on the provided documents.",   # curly apostrophe
    "The documents do not contain a ticker symbol.",
])
def test_is_refusal_detects_refusals(text):
    assert is_refusal(text)


def test_is_refusal_ignores_real_answers():
    assert not is_refusal("Kestrel's CEO is Dana Whitfield.")


def make_case(**overrides):
    case = {"category": "numeric", "expected_numbers": [], "must_include": []}
    case.update(overrides)
    return case


def test_score_case_passes_correct_numeric_answer():
    case = make_case(expected_numbers=[108.3])
    assert score_case(case, "Q1 2026 revenue was $108.3 million.").passed


def test_score_case_fails_wrong_number():
    result = score_case(make_case(expected_numbers=[108.3]), "Revenue was $116.0 million.")
    assert not result.passed
    assert "missing numbers" in result.reason


def test_score_case_requires_every_number_in_a_range():
    case = make_case(expected_numbers=[470, 480])
    assert not score_case(case, "Guidance was raised to $470 million.").passed


def test_score_case_is_case_insensitive_for_text():
    case = make_case(category="factual", must_include=["LaneSight"])
    assert score_case(case, "Kestrel launched lanesight.").passed


def test_out_of_scope_must_be_refused():
    case = make_case(category="out_of_scope")
    assert score_case(case, "I don't know based on the provided documents.").passed
    assert not score_case(case, "The ticker is KSTL.").passed


def test_refusing_an_answerable_question_fails():
    result = score_case(make_case(expected_numbers=[64]), "I don't know based on the provided documents.")
    assert not result.passed
    assert "refused" in result.reason


# ---- Phase 2 additions -------------------------------------------------

from evals.metrics import mentions_internals, refusal_is_exact  # noqa: E402

REFUSAL_TEXT = "I don't know based on the provided documents."


def test_must_include_any_accepts_any_listed_wording():
    case = make_case(category="false_premise", must_include_any=["rose", "increased", "grew"])
    assert score_case(case, "Revenue did not decline; it grew 19%.").passed
    result = score_case(case, "Revenue declined because of fuel prices.")
    assert not result.passed
    assert "contains none of" in result.reason


def test_refusal_is_exact():
    assert refusal_is_exact(REFUSAL_TEXT, REFUSAL_TEXT)
    assert refusal_is_exact("  " + REFUSAL_TEXT + "\n", REFUSAL_TEXT)        # whitespace is fine
    assert refusal_is_exact(REFUSAL_TEXT.replace("'", "’"), REFUSAL_TEXT)
    # The g020 answer from run 1: a refusal plus extra commentary.
    assert not refusal_is_exact(REFUSAL_TEXT + "\n\nThe passages do not mention succession.", REFUSAL_TEXT)


@pytest.mark.parametrize("answer, expected", [
    ("According to the context, the segment contributed 62%.", True),       # g005 in run 1
    ("Passage [2] says revenue was $116.0 million.", True),
    ("The context passages confirm Dana Whitfield is CEO.", True),
    ("The reference material shows revenue of $116.0 million.", True),     # prompt v2 wording
    ("Kestrel's revenue was $116.0 million.", False),
    ("According to the Q2 2026 earnings release, EPS was $0.38.", False),   # citing a real doc is fine
])
def test_mentions_internals(answer, expected):
    assert mentions_internals(answer) is expected


# ---- Fixes after run 2 ---------------------------------------------------

from evals.metrics import passage_hit  # noqa: E402


def test_passage_hit_needs_every_evidence_string():
    case = {"evidence": ["108.3", "116.0"]}
    both = ["Q1 revenue of $108.3 million", "Q2 revenue of $116.0 million"]
    assert passage_hit(case, both) is True
    # The g022 situation: right documents, but the Q2 passage was guidance, not results.
    guidance_only = ["Q1 revenue of $108.3 million", "Q2 outlook: $112 million to $115 million"]
    assert passage_hit(case, guidance_only) is False


def test_passage_hit_is_none_without_evidence():
    assert passage_hit({"evidence": []}, ["anything"]) is None
    assert passage_hit({}, ["anything"]) is None        # older dataset versions
