"""Offline tests for repeat-run consistency, using scripted fake assistants and judges."""
from types import SimpleNamespace

from app.llm import LLMResponse
from evals.consistency import (
    answer_stability,
    check_case,
    normalize_answer,
    summarize_consistency,
    verdict_stability,
)
from evals.judge import JudgeError, Verdict

CASE = {"id": "g006", "category": "numeric", "question": "Q1 revenue?",
        "expected_answer": "$108.3 million", "expected_numbers": [108.3],
        "must_include": [], "must_include_any": [], "source_docs": ["q1_2026_earnings.md"],
        "evidence": ["108.3"]}


class ScriptedAssistant:
    """Returns the given answers in order, one per call."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = 0

    def answer(self, question):
        text = self.answers[self.calls]
        self.calls += 1
        llm = LLMResponse(text=text, model="fake", input_tokens=1, output_tokens=1, latency_s=0.0)
        return SimpleNamespace(answer=text, sources=[], contexts=[f"ctx{self.calls}"], llm=llm)


def scripted_judge(*verdicts):
    """Returns (correct, faithful) pairs in order; a JudgeError instance is raised instead."""
    calls = []

    def _judge(question, contexts, reference, answer, **kwargs):
        calls.append({"contexts": contexts, "answer": answer, **kwargs})
        v = verdicts[len(calls) - 1]
        if isinstance(v, Exception):
            raise v
        return Verdict(correct=v[0], faithful=v[1], reasoning="r")
    return _judge, calls


# ---- Building blocks ----------------------------------------------------

def test_whitespace_differences_dont_count_as_different_answers():
    assert normalize_answer("Revenue was\n $108.3M.  ") == "Revenue was $108.3M."
    assert answer_stability(["A  b", "A b", "A\nb"])["identical"] is True


def test_answer_stability_counts_distinct_answers():
    s = answer_stability(["x", "x", "y", "x", "z"])
    assert s == {"distinct_answers": 3, "identical": False, "most_common_share": 0.6}


def test_verdict_stability():
    assert verdict_stability([True, True, True])["stable"] is True
    flip = verdict_stability([True, False, True])
    assert flip["stable"] is False and flip["passes"] == 2 and flip["fails"] == 1


def test_errors_are_counted_but_not_treated_as_flips():
    s = verdict_stability([True, None, True])
    assert s["stable"] is True and s["errors"] == 1


def test_stability_is_unknown_with_fewer_than_two_verdicts():
    assert verdict_stability([True, None, None])["stable"] is None


# ---- check_case ---------------------------------------------------------

def test_stable_case():
    judge, _ = scripted_judge((True, True), (True, True), (True, True))
    row = check_case(CASE, ScriptedAssistant(*["Revenue was $108.3 million."] * 3), runs=3,
                     judge_fn=judge)
    assert row["assistant"]["identical"] is True
    assert row["rule"]["stable"] is True
    assert row["judge_correct"]["stable"] is True and row["judge_faithful"]["stable"] is True


def test_wording_change_without_a_verdict_change():
    # Like g001/g014/g031 in Run 3: different words, same outcome.
    row = check_case(CASE, ScriptedAssistant("It was $108.3 million.", "Revenue: $108.3M."),
                     runs=2, use_judge=False)
    assert row["assistant"]["distinct_answers"] == 2
    assert row["rule"]["stable"] is True


def test_rule_verdict_flip_is_detected():
    row = check_case(CASE, ScriptedAssistant("$108.3 million", "I don't know."), runs=2,
                     use_judge=False)
    assert row["rule"]["stable"] is False


def test_judge_grades_the_same_fixed_answer_every_time():
    judge, calls = scripted_judge((True, True), (False, True), (True, True))
    row = check_case(CASE, ScriptedAssistant("first 108.3", "second 108.3", "third 108.3"),
                     runs=3, judge_fn=judge, judge_prompt="judge-v1")
    # Every judge call saw the FIRST answer and its passages, so a change is the judge's own noise.
    assert {c["answer"] for c in calls} == {"first 108.3"}
    assert {tuple(c["contexts"]) for c in calls} == {("ctx1",)}
    assert {c["prompt_version"] for c in calls} == {"judge-v1"}
    assert row["judged_answer"] == "first 108.3"
    assert row["judge_correct"]["stable"] is False      # the g032 problem
    assert row["judge_faithful"]["stable"] is True


def test_judge_errors_are_recorded_not_scored():
    judge, _ = scripted_judge((True, True), JudgeError("garbled"), (True, True))
    row = check_case(CASE, ScriptedAssistant(*["108.3"] * 3), runs=3, judge_fn=judge)
    assert row["judge_errors"] == ["garbled"]
    assert row["judge_verdicts"][1] is None
    assert row["judge_correct"]["stable"] is True and row["judge_correct"]["errors"] == 1


def test_no_judge_mode_never_calls_the_judge():
    def exploding(*args, **kwargs):
        raise AssertionError("judge should not be called")

    row = check_case(CASE, ScriptedAssistant("108.3", "108.3"), runs=2, use_judge=False,
                     judge_fn=exploding)
    assert row["judge_correct"] is None


# ---- Summary ------------------------------------------------------------

def test_summary_names_the_unstable_questions():
    stable_judge, _ = scripted_judge((True, True), (True, True))
    flaky_judge, _ = scripted_judge((True, True), (False, True))
    rows = [
        check_case(CASE, ScriptedAssistant("108.3", "108.3"), runs=2, judge_fn=stable_judge),
        check_case({**CASE, "id": "g032"}, ScriptedAssistant("108.3", "It's 108.3"), runs=2,
                   judge_fn=flaky_judge),
    ]
    s = summarize_consistency(rows)
    assert s["questions"] == 2 and s["runs_per_question"] == 2
    assert s["identical_answer_rate"] == 0.5
    assert s["avg_distinct_answers"] == 1.5
    assert s["changed_wording"] == ["g032"]
    assert s["rule_flips"] == [] and s["rule_stable_rate"] == 1.0
    assert s["judge_correct_flips"] == ["g032"] and s["judge_correct_stable_rate"] == 0.5
    assert s["judge_faithful_stable_rate"] == 1.0


def test_summary_without_judge_reports_judge_rates_as_unmeasured():
    rows = [check_case(CASE, ScriptedAssistant("108.3", "108.3"), runs=2, use_judge=False)]
    s = summarize_consistency(rows)
    assert s["judge_correct_stable_rate"] is None and s["judge_correct_flips"] == []
