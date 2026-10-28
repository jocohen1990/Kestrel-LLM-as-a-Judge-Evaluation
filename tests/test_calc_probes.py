"""Offline tests for the ag017 calculator probes (data/agent/calc_probes_v1.jsonl).

The probes are a diagnostic, not a benchmark: each one changes one thing about
ag017 (the case where the agent skips the calculator in every run), so that
which probes do and don't use the calculator points at the cause. That only
works if each probe really is a fair calculator case, so, like the agent_v1
tests, every probe is checked against the real tools and corpus before any API
money is spent.
"""
import json

import pytest

from app.paths import AGENT_DIR, CORPUS_DIR
from app.tools import METRIC_TABLE, run_tool
from evals.agent_eval import calculator_value, grade_agent, load_agent_cases
from types import SimpleNamespace

PROBES = load_agent_cases(AGENT_DIR / "calc_probes_v1.jsonl")
AGENT_V1 = {c["id"]: c for c in load_agent_cases(AGENT_DIR / "agent_v1.jsonl")}
CORPUS = "\n".join(p.read_text(encoding="utf-8") for p in CORPUS_DIR.glob("*.md"))

# The arithmetic behind each probe, written out by hand.
EXPRESSIONS = {
    "cp01": "150 - 40",
    "cp02": "150 - 40",
    "cp03": "2850 - 1240",
    "cp04": "64 - 25",
    "cp05": "480 - 470",
    "cp06": "150 - 40",
}


def step(tool, args):
    r = run_tool(tool, args)
    return {"turn": 1, "tool": tool, "args": args, "output": r.output, "is_error": r.is_error}


def perfect_steps(case):
    steps = []
    for req in case["required_steps"]:
        req = req["any_of"][0] if "any_of" in req else req
        if req["tool"] == "get_financial_metric":
            steps.append(step("get_financial_metric", req["args"]))
        elif req["tool"] == "search_documents":
            steps.append(step("search_documents", {"query": req["evidence"]}))
        else:
            steps.append(step("calculator", {"expression": EXPRESSIONS[case["id"]]}))
    return steps


def result(steps, answer, question):
    return SimpleNamespace(question=question, answer=answer, steps=steps, stopped="answered")


def answer_for(case):
    return f"The answer is {case['expected_numbers'][0]:g}."


def test_six_probes_each_naming_what_it_changes():
    assert [c["id"] for c in PROBES] == [f"cp0{i}" for i in range(1, 7)]
    assert all(c.get("probe") for c in PROBES)


def test_cp01_is_ag017_word_for_word():
    # The control: if cp01 behaves differently from ag017, the probe run isn't
    # comparable to Runs 14-15 and the other probes can't be read against it.
    ag017 = AGENT_V1["ag017"]
    cp01 = PROBES[0]
    for key in ("question", "required_steps", "max_steps", "expected_numbers", "first_tool"):
        assert cp01[key] == ag017[key], key


def test_no_other_probe_repeats_an_agent_v1_question():
    questions = {c["question"] for c in AGENT_V1.values()}
    assert [c["id"] for c in PROBES if c["question"] in questions] == ["cp01"]


@pytest.mark.parametrize("case", PROBES, ids=lambda c: c["id"])
def test_every_probe_needs_the_calculator_for_a_number_no_filing_states(case):
    calc = [r for r in case["required_steps"] if r.get("tool") == "calculator"]
    assert len(calc) == 1
    target = calc[0]["result"][0]
    assert calculator_value(step("calculator", {"expression": EXPRESSIONS[case["id"]]})["output"]) == target
    # If the filings stated the answer, reading it would be correct and the
    # probe would be unfair. Checked as a whole figure next to "$" or "," text.
    for written in (f"${target:g} million", f" {target:,g} ", f"{target:,g} enterprise"):
        assert written not in CORPUS, written


@pytest.mark.parametrize("case", PROBES, ids=lambda c: c["id"])
def test_every_evidence_string_and_metric_step_is_real(case):
    for req in case["required_steps"]:
        for alt in req.get("any_of", [req]):
            if alt["tool"] == "search_documents":
                assert alt["evidence"] in CORPUS
                assert alt["evidence"] in step("search_documents", {"query": alt["evidence"]})["output"]
            elif alt["tool"] == "get_financial_metric":
                assert (alt["args"]["metric"], alt["args"]["period"]) in METRIC_TABLE


@pytest.mark.parametrize("case", PROBES, ids=lambda c: c["id"])
def test_perfect_run_is_fully_correct(case):
    grade = grade_agent(case, result(perfect_steps(case), answer_for(case), case["question"]))
    assert grade["fully_correct"], grade
    assert grade["unsupported_numbers"] == []


@pytest.mark.parametrize("case", PROBES, ids=lambda c: c["id"])
def test_skipping_the_calculator_is_right_answer_wrong_path(case):
    # The one behavior the probes exist to detect.
    steps = [s for s in perfect_steps(case) if s["tool"] != "calculator"]
    grade = grade_agent(case, result(steps, answer_for(case), case["question"]))
    assert grade["answer_ok"] and not grade["fully_correct"] and not grade["args_ok"]


@pytest.mark.parametrize("case_id", ["cp01", "cp02", "cp03"])
def test_unsupported_numbers_flag_also_catches_the_skip(case_id):
    case = next(c for c in PROBES if c["id"] == case_id)
    steps = [s for s in perfect_steps(case) if s["tool"] != "calculator"]
    grade = grade_agent(case, result(steps, answer_for(case), case["question"]))
    assert grade["unsupported_numbers"] == [f"{case['expected_numbers'][0]:g}"]


@pytest.mark.parametrize("case_id,why", [
    ("cp04", "net income $38.9 million is in the same search results and rounds to 39"),
    ("cp05", "the 1 in 'Q1' counts as a number, and 1/100 in thousands is 10"),
])
def test_unsupported_numbers_flag_misses_small_whole_numbers(case_id, why):
    # A known limit of the heuristic, pinned so it can't change unnoticed: a
    # small whole number is "supported" by almost anything nearby. The step
    # grader (no calculator step) still fails these runs; only the second,
    # independent check is lost. See README "Known gaps".
    case = next(c for c in PROBES if c["id"] == case_id)
    steps = [s for s in perfect_steps(case) if s["tool"] != "calculator"]
    grade = grade_agent(case, result(steps, answer_for(case), case["question"]))
    assert grade["unsupported_numbers"] == [], why
    assert not grade["args_ok"]
