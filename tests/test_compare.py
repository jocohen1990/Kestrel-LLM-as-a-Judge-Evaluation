"""Offline tests for comparing two trace files (evals/compare.py, compare_traces.py).

Traces are made by the real tracing code with a fake model (tests/test_monitors.py's
helpers), so the comparison is tested on the shape the app records.
"""
import copy
import json

import compare_traces as compare_cli
from app.rag import REFUSAL
from evals import compare as c
from evals import monitors as m
from tests.test_monitors import rag_trace

ANSWER = "Kestrel's head office is in Omaha, Nebraska."


def as_agent(trace):
    """A RAG-shaped trace relabelled as the agent's: enough for the comparison,
    which only needs the app, the outcome and the flags."""
    t = copy.deepcopy(trace)
    t["attributes"]["app.kind"] = "agent"
    t["attributes"]["kestrel.agent_version"] = "a3"
    t["attributes"].pop("kestrel.prompt_version", None)
    t["attributes"].pop("kestrel.retriever_version", None)
    t["trace_id"] = "f" + t["trace_id"][1:]
    return t


def release(trace, prompt="v7"):
    t = copy.deepcopy(trace)
    t["attributes"]["kestrel.prompt_version"] = prompt
    t["trace_id"] = "e" + t["trace_id"][1:]
    return t


# ---- What a reply did --------------------------------------------------------------

def test_outcome_of_each_kind_of_reply(monkeypatch):
    assert c.outcome(m.view(rag_trace(monkeypatch, ANSWER))) == "answered"
    assert c.outcome(m.view(rag_trace(monkeypatch, REFUSAL))) == "declined"
    assert c.outcome(m.view(rag_trace(monkeypatch, "Which period do you mean?"))) == "asked_back"
    assert c.outcome(m.view(rag_trace(monkeypatch, ANSWER, capture=False))) == "no_content"
    failed = rag_trace(monkeypatch, ANSWER)
    failed["status"] = "error"
    assert c.outcome(m.view(failed)) == "error"


def test_a_refusal_with_more_text_still_declined(monkeypatch):
    """Run 22 u022: the refusal sentence, then a sentence about the material."""
    t = rag_trace(monkeypatch, REFUSAL + " The material states that Kestrel was founded in 2017.")
    assert c.outcome(m.view(t)) == "declined"


def test_setup_reads_the_versions_and_spots_a_mixed_file(monkeypatch):
    t = rag_trace(monkeypatch, ANSWER)
    setup = c.setup_of([t])
    assert setup["app"] == "rag" and "prompt_version=v6" in setup["label"]
    assert c.setup_of([t, release(t)])["app"] == "mixed"


# ---- Across apps -------------------------------------------------------------------

def two_apps(monkeypatch):
    """u001-style: RAG refuses, the agent answers. Plus one request both get right,
    and one only the agent has a trace for (Run 22's lost u019)."""
    rag = [rag_trace(monkeypatch, REFUSAL, question="What city is Kestrel's head office in?",
                     case_id="u001"),
           rag_trace(monkeypatch, "Q1 2026 revenue was $108.3 million.", case_id="u002")]
    agent = [as_agent(rag_trace(monkeypatch, ANSWER, question="What city is Kestrel's head office in?",
                                case_id="u001")),
             as_agent(rag_trace(monkeypatch, "Q1 2026 revenue was $108.3 million.", case_id="u002")),
             as_agent(rag_trace(monkeypatch, REFUSAL, question="Stock price?", case_id="u019"))]
    return rag, agent


def test_answered_vs_declined_is_queued_and_new_to_review(monkeypatch):
    rag, agent = two_apps(monkeypatch)
    result = c.compare(rag, agent)
    assert result["mode"] == "apps"
    by_case = {r["case"]: r for r in result["rows"]}
    assert by_case["u001"]["kind"] == "answer_vs_decline"
    assert by_case["u002"]["kind"] is None   # same outcome; wording isn't compared across apps
    assert by_case["u019"]["kind"] == "missing" and result["only_in_b"] == ["u019"]
    assert result["queue"] == ["u001", "u019"]
    # the wrong refusal is the exact sentence, so no monitor queued it: only the comparison can
    assert "u001" in result["new_to_review"]
    assert result["paired"] == 2


def test_a_flag_on_one_side_is_reported_but_not_queued(monkeypatch):
    rag = [rag_trace(monkeypatch, "- revenue was $108.3 million", case_id="t001")]
    agent = [as_agent(rag_trace(monkeypatch, "Revenue was $108.3 million.", case_id="t001"))]
    result = c.compare(rag, agent)
    assert result["rows"][0]["kind"] == "flags_changed"
    assert result["queue"] == []   # list_or_bold already put it in RAG's monitor queue
    assert result["monitor_queues"]["a"] == ["t001"]


def test_scoring_counts_a_problem_on_either_side(monkeypatch):
    rag, agent = two_apps(monkeypatch)
    result = c.compare(rag, agent)
    reviews = [{"app": "rag", "case_id": "u001", "verdict": "wrong"},
               {"app": "agent", "case_id": "u019", "verdict": "rule_break"}]   # unpaired: ignored
    s = c.score_against_reviews(result, reviews, reviews)
    assert s["wrong"] == ["u001"]
    assert s["precision"] == 1.0 and s["recall_wrong"] == 1.0
    assert s["monitors_alone"]["recall_wrong"] == 0.0     # the exact refusal raises no flag
    assert s["monitors_plus_comparison"]["recall_wrong"] == 1.0
    assert s["problem_side"] == {"u001": "a"}


def test_reviews_are_matched_by_app(monkeypatch):
    """An agent problem in the review isn't counted against the RAG side."""
    rag, agent = two_apps(monkeypatch)
    result = c.compare(rag, agent)
    s = c.score_against_reviews(result, [{"app": "agent", "case_id": "u002", "verdict": "wrong"}],
                                [{"app": "agent", "case_id": "u002", "verdict": "wrong"}])
    assert s["wrong"] == ["u002"] and s["queued_with_problem"] == 0


# ---- Across releases ---------------------------------------------------------------

def test_releases_report_fixed_flags_and_reworded_answers(monkeypatch):
    before = [rag_trace(monkeypatch, "- revenue was $108.3 million", case_id="t001"),
              rag_trace(monkeypatch, REFUSAL, case_id="t002"),
              rag_trace(monkeypatch, "Net income was $9.2 million.", case_id="t003")]
    after = [release(rag_trace(monkeypatch, "Revenue was $108.3 million.", case_id="t001")),
             release(rag_trace(monkeypatch, "Kestrel was founded in 2017.", case_id="t002")),
             release(rag_trace(monkeypatch, "Q1 2026 net income was $9.2 million.", case_id="t003"))]
    result = c.compare(before, after)
    assert result["mode"] == "releases"
    by_case = {r["case"]: r for r in result["rows"]}
    assert by_case["t001"]["kind"] == "flags_changed"
    assert by_case["t001"]["resolved_flags"] == ["list_or_bold"] and by_case["t001"]["new_flags"] == []
    assert by_case["t002"]["kind"] == "answer_vs_decline"   # a fix or a regression: read it
    assert by_case["t003"]["kind"] == "answer_changed"
    assert result["queue"] == ["t002"]
    assert c.stale_reviews(result) == ["t001", "t002", "t003"]
    text = c.format_comparison(result)
    assert "Across releases" in text and "resolved list_or_bold" in text
    assert "answer_changed 1" in text   # summarized, not listed, without --all
    assert "t003  [answer_changed]" in c.format_comparison(result, show_all=True)


def test_releases_scored_with_only_the_baseline_review_warn_about_stale_rows(monkeypatch):
    before = [rag_trace(monkeypatch, REFUSAL, case_id="u001")]
    after = [release(rag_trace(monkeypatch, ANSWER, case_id="u001"))]
    result = c.compare(before, after)
    s = c.score_against_reviews(result, [{"app": "rag", "case_id": "u001", "verdict": "wrong"}], None)
    assert s["stale_reviews"] == ["u001"]
    assert "doesn't cover them" in c.format_comparison(result, s)


# ---- The script --------------------------------------------------------------------

def write(path, traces):
    path.write_text("".join(json.dumps(t) + "\n" for t in traces), encoding="utf-8")


def test_cli_compares_scores_and_saves(monkeypatch, tmp_path, capsys):
    rag, agent = two_apps(monkeypatch)
    write(tmp_path / "a.jsonl", rag)
    write(tmp_path / "b.jsonl", agent)
    reviews = tmp_path / "reviews.jsonl"
    reviews.write_text(json.dumps({"app": "rag", "case_id": "u001", "verdict": "wrong"}) + "\n",
                       encoding="utf-8")
    code = compare_cli.main([str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl"),
                             "--reviews", str(reviews), "--results-dir", str(tmp_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "u001  [answer_vs_decline]" in out and "WARNING: only in B: u019" in out
    assert "wrong answers 100%" in out
    saved = json.loads((tmp_path / "comparison_latest.json").read_text(encoding="utf-8"))
    assert saved["files"] == {"a": "a.jsonl", "b": "b.jsonl"}
    assert saved["review_scores"]["recall_wrong"] == 1.0


def test_cli_strict_fails_on_a_new_alert(monkeypatch, tmp_path):
    before = [rag_trace(monkeypatch, ANSWER, case_id="t001")]
    after = [release(rag_trace(monkeypatch, ANSWER + " See https://example.com", case_id="t001"))]
    write(tmp_path / "a.jsonl", before)
    write(tmp_path / "b.jsonl", after)
    args = [str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl"), "--no-save"]
    assert compare_cli.main(args) == 0
    assert compare_cli.main(args + ["--strict"]) == 1


def test_cli_refuses_files_with_nothing_in_common(monkeypatch, tmp_path, capsys):
    write(tmp_path / "a.jsonl", [rag_trace(monkeypatch, ANSWER, case_id="t001")])
    write(tmp_path / "b.jsonl", [rag_trace(monkeypatch, ANSWER, case_id="u001")])
    assert compare_cli.main([str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl"), "--no-save"]) == 2
    assert "share no traffic case ids" in capsys.readouterr().out


def test_dashboard_shows_the_comparison_and_the_candidates(monkeypatch, tmp_path):
    from evals import candidates as cd
    from evals.dashboard import build_dashboard, render_html
    rag, agent = two_apps(monkeypatch)
    result = c.compare(rag, agent)
    result["files"] = {"a": "rag.jsonl", "b": "agent.jsonl"}
    result["review_scores"] = c.score_against_reviews(
        result, [{"app": "rag", "case_id": "u001", "verdict": "wrong"}], None)
    (tmp_path / "comparison_latest.json").write_text(json.dumps(result), encoding="utf-8")
    traffic = {"u001": {"file": "traffic_v2.jsonl", "id": "u001", "question": "<i>head</i> office?"}}
    cands = cd.build_candidates({"rag.jsonl": rag, "agent.jsonl": agent}, [], traffic)
    (tmp_path / "golden_candidates_latest.json").write_text(json.dumps(cands), encoding="utf-8")
    page = render_html(build_dashboard(tmp_path))
    assert "Trace comparison: Across apps" in page and "answer_vs_decline" in page
    assert "with the comparison: 100%" in page
    assert "Golden-set candidates" in page
    assert "<i>head</i>" not in page   # questions are escaped
