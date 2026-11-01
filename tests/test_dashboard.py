"""Offline tests for the results dashboard (evals/dashboard.py, dashboard.py).

Each test builds a small results folder in a temp directory, so nothing here
depends on which results files happen to be saved on this machine.
"""
import json

import dashboard
from app.agent import DEFAULT_AGENT_VERSION, DEFAULT_MAX_TURNS
from app.llm import DEFAULT_MODEL
from app.rag import DEFAULT_PROMPT_VERSION, DEFAULT_RETRIEVER_VERSION
from evals.dashboard import (build_dashboard, fmt, gate_status, gate_totals, history, open_issues,
                             render_html, short_model, version_label)
from evals.judge import DEFAULT_JUDGE_MODEL, DEFAULT_JUDGE_PROMPT
from evals.scorecard import SUITES

RAG = {"prompt_version": DEFAULT_PROMPT_VERSION, "retriever_version": DEFAULT_RETRIEVER_VERSION, "k": 4}


def case(i, **extra):
    row = {"id": f"c{i:02d}", "latency_s": 1.0, "input_tokens": 100, "output_tokens": 10}
    row.update(extra)
    return row


def golden_run(ts, pass_rate=1.0, n=4, model=DEFAULT_MODEL, prompt=DEFAULT_PROMPT_VERSION, cases=None):
    return {"timestamp": ts, "model": model, "golden_file": "golden_v3.jsonl", **RAG,
            "prompt_version": prompt, "judge_model": DEFAULT_JUDGE_MODEL,
            "judge_prompt_version": DEFAULT_JUDGE_PROMPT,
            "summary": {"pass_rate": pass_rate, "judge_faithfulness_rate": 1.0, "judge_correct_rate": 1.0},
            "cases": cases if cases is not None else [case(i, passed=True) for i in range(n)]}


def latest_runs(ts="2026-10-30T03:00:00+00:00"):
    """A latest file for every suite, all made with the default setup, all passing."""
    return {
        "latest.json": golden_run(ts),
        "injection_latest.json": {
            "timestamp": ts, "model": DEFAULT_MODEL, "attacks_file": "injection_v1.jsonl", **RAG,
            "summary": {"attack_success_rate": 1 / 21, "task_preserved_rate": 1.0,
                        "succeeded_ids": ["inj007"], "mentions_internals_ids": []},
            "cases": [case(i) for i in range(21)]},
        "jailbreak_latest.json": {
            "timestamp": ts, "model": DEFAULT_MODEL, "cases_file": "jailbreak_v1.jsonl", **RAG,
            "summary": {"attack_success_rate": 0.0, "over_refusal_rate": 0.0, "succeeded_ids": [],
                        "over_refused_ids": [], "mentions_internals_ids": ["jb108"]},
            "cases": [case(i) for i in range(28)]},
        "bias_latest.json": {
            "timestamp": ts, "model": DEFAULT_MODEL, "cases_file": "bias_v1.jsonl", **RAG,
            "summary": {"verdict_differs_ids": [], "consistency_rate": 0.875,
                        "flagged_ids": ["bias04-negative"]},
            "cases": [case(i) for i in range(28)]},
        "agent_eval_latest.json": {
            "timestamp": ts, "model": DEFAULT_MODEL, "cases_file": "agent_v1.jsonl",
            "agent_version": DEFAULT_AGENT_VERSION, "max_turns": DEFAULT_MAX_TURNS,
            "summary": {"fully_correct_rate": 26 / 27, "answer_accuracy": 1.0, "tool_choice_accuracy": 26 / 27,
                        "right_answer_wrong_path_ids": ["ag017"], "wrong_answer_ids": []},
            "cases": [case(i) for i in range(27)]},
        "robustness_latest.json": {
            "timestamp": ts, "model": DEFAULT_MODEL, "variants_file": "paraphrases_v1.jsonl",
            "golden_file": "golden_v3.jsonl", "retrieval_only": False, **RAG,
            "summary": {"overall_robustness": 0.92,
                        "by_style": {"original": {"flips": []}, "typo": {"flips": ["g022-typo"]}}},
            "cases": [case(i) for i in range(10)]},
    }


def write(folder, files: dict):
    for name, content in files.items():
        (folder / name).write_text(json.dumps(content), encoding="utf-8")


# ---- Gates -------------------------------------------------------------------------

def test_every_ci_gate_is_checked(tmp_path):
    write(tmp_path, latest_runs())
    gates = gate_status(tmp_path)
    assert [g["suite"] for g in gates] == list(SUITES)
    totals = gate_totals(gates)
    assert totals == {"passed": 9, "failed": 0, "not_run": 0,
                      "total": sum(len(s.gates) for s in SUITES.values())}
    assert all(g["default_setup"] for g in gates)


def test_a_failing_gate_is_counted_and_shown(tmp_path):
    files = latest_runs()
    files["latest.json"] = golden_run("2026-10-30T03:00:00+00:00", pass_rate=0.80)
    write(tmp_path, files)
    data = build_dashboard(tmp_path)
    assert data["gate_totals"]["failed"] == 1
    page = render_html(data, generated_at="test")
    assert "1 of 9 gates failing" in page
    assert "✕</span> Fail" in page


def test_a_missing_results_file_is_not_run_not_passed(tmp_path):
    files = latest_runs()
    del files["robustness_latest.json"]
    write(tmp_path, files)
    totals = gate_totals(gate_status(tmp_path))
    assert totals["not_run"] == 1 and totals["passed"] == 8
    assert "8 of 9 gates passing, 1 not run" in render_html(build_dashboard(tmp_path), "test")


def test_latest_file_from_a_non_default_setup_is_called_out(tmp_path):
    files = latest_runs()
    files["latest.json"] = golden_run("2026-10-30T03:00:00+00:00", prompt="v1")
    write(tmp_path, files)
    golden = next(g for g in gate_status(tmp_path) if g["suite"] == "golden")
    assert golden["default_setup"] is False
    assert "not the default setup" in render_html(build_dashboard(tmp_path), "test")


def test_a_broken_results_file_is_treated_as_missing(tmp_path):
    files = latest_runs()
    write(tmp_path, files)
    (tmp_path / "bias_latest.json").write_text("{not json", encoding="utf-8")
    bias = next(g for g in gate_status(tmp_path) if g["suite"] == "bias")
    assert bias["run"] is None


# ---- History -----------------------------------------------------------------------

def test_history_keeps_full_default_model_runs_in_time_order(tmp_path):
    write(tmp_path, {
        "run_20261005_000000.json": golden_run("2026-10-05T00:00:00+00:00", 0.90, prompt="v1"),
        "run_20261022_000000.json": golden_run("2026-10-22T00:00:00+00:00", 1.0),
        # Another model: belongs to the comparison, not the history.
        "run_20261030_000000.json": golden_run("2026-10-30T00:00:00+00:00", 0.5, model="claude-haiku-5-5"),
        # A 2-question re-run: not the suite's score.
        "run_20261023_000000.json": golden_run("2026-10-23T00:00:00+00:00", 0.0, n=2),
        # Another dataset.
        "run_20261004_000000.json": {**golden_run("2026-10-04T00:00:00+00:00"), "golden_file": "golden_v1.jsonl"},
        # The latest copy is never counted twice.
        "latest.json": golden_run("2026-10-22T00:00:00+00:00", 1.0),
    })
    h = history(tmp_path)["golden"]
    assert [(p["label"], p["value"]) for p in h["points"]] == [("v1 r3", 0.90), (f"{DEFAULT_PROMPT_VERSION} r3", 1.0)]
    assert h["gate"] == {"op": ">=", "limit": 0.85}


def test_history_skips_retrieval_only_robustness_files(tmp_path):
    base = latest_runs()["robustness_latest.json"]
    write(tmp_path, {
        "robustness_20261022_000000.json": base,
        "robustness_retrieval_20261022_000000.json": {**base, "retrieval_only": True,
                                                      "summary": {"overall_robustness": 0.1}},
    })
    assert [p["value"] for p in history(tmp_path)["robustness"]["points"]] == [0.92]


def test_history_with_no_saved_runs_is_empty_not_an_error(tmp_path):
    h = history(tmp_path)
    assert all(v["points"] == [] for v in h.values())
    assert "No saved full runs" in render_html(build_dashboard(tmp_path), "test")


def test_version_labels():
    assert version_label("golden", {"prompt_version": "v6", "retriever_version": "r3"}) == "v6 r3"
    assert version_label("golden", {"prompt_version": "v2", "retriever_version": "r1"}) == "v2"
    assert version_label("golden", {}) == "v1"   # Run 1 had no prompt versions yet
    assert version_label("agent", {"agent_version": "a2"}) == "a2"


# ---- Open issues -------------------------------------------------------------------

def test_open_issues_separate_counted_failures_from_flags(tmp_path):
    write(tmp_path, latest_runs())
    rows = {(r["suite"], r["what"]): r for r in open_issues(tmp_path)}
    assert rows[("injection", "Attack worked")]["ids"] == ["inj007"]
    assert rows[("injection", "Attack worked")]["severity"] == "fail"
    assert rows[("agent", "Right answer, wrong path (tools, arguments or order)")]["severity"] == "fail"
    assert rows[("robustness", "Passed as written, failed when reworded or misspelled")]["ids"] == ["g022-typo"]
    assert rows[("bias", "Answer used a flagged word (verdict unchanged)")]["severity"] == "flag"
    assert rows[("jailbreak", "Answer mentions internal names")]["ids"] == ["jb108"]
    # Empty lists make no row.
    assert ("jailbreak", "Jailbreak worked") not in rows


def test_golden_failures_carry_the_reason(tmp_path):
    files = latest_runs()
    files["latest.json"] = golden_run("2026-10-30T03:00:00+00:00", cases=[
        case(1, passed=False, reason="missing number 41.2"),
        case(2, passed=True, judge_correct=False),
        case(3, passed=True, mentions_internals=True)])
    write(tmp_path, files)
    rows = [r for r in open_issues(tmp_path) if r["suite"] == "golden"]
    assert {"suite": "golden", "ids": ["c01"], "what": "Rule check failed: missing number 41.2",
            "severity": "fail"} in rows
    assert any(r["ids"] == ["c02"] and r["severity"] == "flag" for r in rows)
    assert any(r["ids"] == ["c03"] and "internal" in r["what"] for r in rows)


# ---- The page ----------------------------------------------------------------------

def comparison_card():
    def model(name, cost, passed=True, judge_self=False):
        return {"model": name, "gates_passed": 8 if passed else 7, "gates_total": 8, "all_gates_passed": passed,
                "cost_per_1k_questions_usd": cost, "avg_latency_s": 1.5, "p95_latency_s": 2.0,
                "temperature_sent": False, "graded_by_itself": judge_self, "suites_missing": [],
                "failed_gates": []}
    results = {n: {"golden": {"metrics": {"pass_rate": 1.0}, "gates": [
        {"metric": "pass_rate", "op": ">=", "limit": 0.85, "value": 1.0, "passed": True}]}}
        for n in ("claude-haiku-5-5", DEFAULT_MODEL)}
    return {"suites": ["golden"], "suite_labels": {"golden": "Golden set"},
            "metric_labels": {"golden": {"pass_rate": {"label": "Rule pass rate", "higher_is_better": True,
                                                       "kind": "rate"}}},
            "prices_checked": "2026-10-29", "timestamp": "2026-10-30T03:46:19+00:00",
            "models": [model("claude-haiku-5-5", 0.15), model(DEFAULT_MODEL, 1.03)],
            "results": results,
            "recommendation": {"model": "claude-haiku-5-5", "reason": "Cheapest that passes."}}


def test_page_shows_the_comparison_and_says_when_the_pick_isnt_the_default(tmp_path):
    files = latest_runs()
    files["model_comparison_latest.json"] = comparison_card()
    write(tmp_path, files)
    page = render_html(build_dashboard(tmp_path), "test")
    assert "Scorecard pick: Haiku 5.5." in page
    assert "isn't the default model (Haiku 4.5)" in page
    assert "$0.15" in page and "$1.03" in page
    assert "doesn't accept temperature=0" in page


def test_page_without_a_comparison_says_how_to_make_one(tmp_path):
    write(tmp_path, latest_runs())
    assert "python compare_models.py" in render_html(build_dashboard(tmp_path), "test")


def test_page_is_self_contained_and_escapes_text(tmp_path):
    files = latest_runs()
    files["bias_latest.json"]["summary"]["flagged_ids"] = ["<script>alert(1)</script>"]
    write(tmp_path, files)
    page = render_html(build_dashboard(tmp_path), "test")
    assert page.startswith("<!DOCTYPE html>")
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "<script>alert(1)</script>" not in page
    # No outside requests: it has to open offline and from a CI artifact.
    assert "http://" not in page and "https://" not in page


def test_formatting():
    assert fmt(1.0) == "100%" and fmt(0.963) == "96.3%" and fmt(0.048) == "4.8%"
    assert fmt(3, "count") == "3" and fmt(None) == "n/a"
    assert short_model("claude-haiku-4-5-20251001") == "Haiku 4.5"
    assert short_model("claude-sonnet-5-5") == "Sonnet 5.5"
    assert short_model("gpt-x") == "gpt-x"


def test_cli_writes_the_page(tmp_path, capsys):
    write(tmp_path, latest_runs())
    out = tmp_path / "site" / "index.html"
    assert dashboard.main(["--results-dir", str(tmp_path), "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
    printed = capsys.readouterr().out
    assert "Gates: 9 passed, 0 failed, 0 not run (of 9)" in printed


def test_cli_default_output_is_in_the_results_folder(tmp_path):
    write(tmp_path, latest_runs())
    dashboard.main(["--results-dir", str(tmp_path)])
    assert (tmp_path / "dashboard.html").exists()
