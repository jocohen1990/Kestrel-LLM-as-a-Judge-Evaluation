"""Offline tests for drift comparison between two results files."""
from evals.drift import case_outcomes, compare_runs, outcome, settings

SETTINGS = {"model": "haiku", "prompt_version": "v2", "k": 4, "golden_file": "golden_v3.jsonl",
            "judge_model": "sonnet", "judge_prompt_version": "judge-v2", "judge_output_mode": "tool"}


def run_file(cases, timestamp="2026-10-05T23:41:09+00:00", **overrides):
    """A run_evals-style results file."""
    rows = [{"id": i, "answer": ans, "passed": passed, "judge_correct": jc, "judge_faithful": jf}
            for i, ans, passed, jc, jf in cases]
    return {**SETTINGS, **overrides, "timestamp": timestamp, "cases": rows}


def consistency_file(cases, timestamp="2026-10-15T03:40:33+00:00", **overrides):
    """A consistency.py-style results file: several answers per question."""
    rows = []
    for i, answers, rules, verdicts in cases:
        rows.append({"id": i, "answers": answers, "rule_verdicts": rules,
                     "judged_answer": answers[0],
                     "judge_verdicts": [{"correct": c, "faithful": f} for c, f in verdicts]})
    return {**SETTINGS, **overrides, "timestamp": timestamp, "cases": rows}


def test_outcome_collapses_repeated_verdicts():
    assert outcome([True, True]) is True
    assert outcome([False, None]) is False
    assert outcome([True, False]) == "mixed"
    assert outcome([None]) is None


def test_old_files_without_an_output_mode_count_as_text():
    assert settings({"judge_model": "sonnet"})["judge_output_mode"] == "text"
    assert settings({"judge_model": None})["judge_output_mode"] is None


def test_reads_both_file_formats():
    r = case_outcomes(run_file([("g001", "A", True, True, True)]))
    c = case_outcomes(consistency_file([("g001", ["A", "A"], [True, True], [(True, True), (False, True)])]))
    assert r["g001"]["answers"] == ["A"] and r["g001"]["judge_correct"] is True
    assert c["g001"]["answers"] == ["A", "A"] and c["g001"]["judge_correct"] == "mixed"


def test_no_drift():
    old = run_file([("g001", "Revenue was $412.6M.", True, True, True)])
    new = consistency_file([("g001", ["Revenue was  $412.6M."] * 3, [True] * 3, [(True, True)] * 3)])
    result = compare_runs(old, new)
    assert result["hours_apart"] == 220.0   # Oct 5 23:41 -> Oct 15 03:40
    assert result["assistant_comparable"] and result["judge_comparable"]
    assert result["summary"]["answers_changed"] == []      # whitespace doesn't count
    assert result["summary"]["judge_questions_compared"] == 1


def test_wording_drift_without_a_verdict_change():
    old = run_file([("g014", "3,120 customers.", True, True, True)])
    new = run_file([("g014", "Kestrel had 3,120 enterprise customers.", True, True, True)])
    s = compare_runs(old, new)["summary"]
    assert s["answers_changed"] == ["g014"] and s["rule_changed"] == []


def test_judge_only_compared_on_the_identical_answer():
    old = run_file([("g001", "A", True, True, True), ("g032", "Same", True, True, True)])
    new = run_file([("g001", "B", True, False, True), ("g032", "Same", True, False, True)])
    result = compare_runs(old, new)
    # g001's verdict changed but so did its answer, so it isn't judge drift.
    assert result["summary"]["judge_changed_on_same_answer"] == ["g032"]
    assert result["summary"]["judge_questions_compared"] == 1


def test_different_judge_settings_are_reported_and_not_compared():
    old = run_file([("g032", "Same", True, True, True)], judge_prompt_version="judge-v1",
                   judge_output_mode=None)
    new = run_file([("g032", "Same", True, False, True)])
    result = compare_runs(old, new)
    assert result["judge_comparable"] is False
    assert result["settings_differ"]["judge_prompt_version"] == ("judge-v1", "judge-v2")
    assert result["settings_differ"]["judge_output_mode"] == ("text", "tool")
    assert result["summary"]["judge_changed_on_same_answer"] == []


def test_different_assistant_prompt_is_not_called_drift():
    old = run_file([("g001", "A", True, None, None)], prompt_version="v1")
    new = run_file([("g001", "B", True, None, None)])
    result = compare_runs(old, new)
    assert result["assistant_comparable"] is False
    assert result["summary"]["answers_changed"] == ["g001"]   # still listed, but flagged


def test_an_answer_shared_with_an_unstable_run_is_not_drift():
    old = consistency_file([("g001", ["A", "B"], [True, True], [])])
    new = consistency_file([("g001", ["B", "B"], [True, True], [])])
    assert compare_runs(old, new)["summary"]["answers_changed"] == []


def test_questions_in_only_one_file_are_skipped_and_listed():
    old = run_file([("g001", "A", True, None, None), ("g033", "X", True, None, None)])
    new = run_file([("g001", "A", True, None, None)])
    result = compare_runs(old, new)
    assert result["only_in_one_file"] == ["g033"] and result["summary"]["questions"] == 1
