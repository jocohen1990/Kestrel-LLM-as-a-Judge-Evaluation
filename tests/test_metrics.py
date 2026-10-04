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
