"""Offline tests for the production monitors (evals/monitors.py, monitor.py).

Traces are made by the real tracing code with a fake model (as in
tests/test_tracing.py), so the checks are tested on the exact shape the app
records. Every check gets a case where it should fire and one where it shouldn't.
"""
import json
from types import SimpleNamespace

import pytest

import monitor as monitor_cli
import traffic
from app import llm, rag, tracing
from app.agent import KestrelAgent, record_result, to_result
from app.llm import LLMResponse
from app.paths import TRAFFIC_DIR
from app.rag import REFUSAL
from app.service import handle
from evals import monitors as m
from evals.dashboard import build_dashboard, render_html
from tests.test_agent import call, step_through, text, use_model
from tests.test_traces import eval_questions, normalize


# ---- Making traces ----------------------------------------------------------------

def rag_trace(monkeypatch, answer, question="What was Kestrel's revenue in Q1 2026?",
              rewrite="revenue first quarter 2026", case_id="t001", capture=True, k=4):
    def fake(system, user, model=None, **kwargs):
        with llm._llm_span("claude-haiku-4-5", 80) as span:
            span.set({"gen_ai.usage.input_tokens": 100, "gen_ai.usage.output_tokens": 10})
        reply = rewrite if system == rag.REWRITE_SYSTEM else answer
        return LLMResponse(text=reply, model="claude-haiku-4-5", input_tokens=100,
                           output_tokens=10, latency_s=0.01)

    monkeypatch.setattr(rag, "generate", fake)
    exporter = tracing.MemoryExporter()
    handle(question, rag.RagAssistant(k=k), exporter=exporter, capture_content=capture,
           extra={"traffic.case_id": case_id})
    return exporter.traces[0]


def agent_trace(monkeypatch, question, *replies, case_id="t001", capture=True):
    use_model(monkeypatch, *replies)
    exporter = tracing.MemoryExporter()
    agent = KestrelAgent()
    attrs = {"app.kind": "agent", "traffic.case_id": case_id, "kestrel.agent_version": "a3"}
    with tracing.trace("kestrel.request", attrs=attrs, exporter=exporter,
                       capture_content=capture) as root:
        root.set_content("input.question", question)
        with tracing.span("agent.run", kind="agent") as span:
            result = step_through(agent, question)
            record_result(span, result)
        root.set_content("output.answer", result.answer)
    return exporter.traces[0]


def checks_on(trace) -> set[str]:
    return {f["check"] for f in m.run_checks(m.view(trace))}


# ---- The RAG checks ---------------------------------------------------------------

def test_a_clean_answer_raises_nothing_but_info(monkeypatch):
    t = rag_trace(monkeypatch, "Kestrel's Q1 2026 revenue was $108.3 million.")
    flags = m.run_checks(m.view(t))
    assert all(f["severity"] == "info" for f in flags)
    assert "refusal" not in checks_on(t)


def test_exact_refusal_is_a_refusal_and_nothing_else(monkeypatch):
    t = rag_trace(monkeypatch, REFUSAL)
    assert checks_on(t) - {"low_top_score"} == {"refusal"}


def test_partial_answer_that_says_the_documents_lack_something(monkeypatch):
    """Run 20, t023: two figures, then a false claim about the documents."""
    t = rag_trace(monkeypatch, "Kestrel's 2025 revenue was $412.6 million and net income was "
                               "$38.9 million. The provided documents do not contain information "
                               "about headcount or customer count.")
    got = checks_on(t)
    assert {"says_documents_lack", "mentions_internals"} <= got
    assert "refusal" not in got   # a partial answer, judged on its first sentence


@pytest.mark.parametrize("answer", [
    "The annual report doesn't state the total revenue figure.",
    "I don't have access to Q4 2025 data.",
    "Headcount is not provided.",
])
def test_absence_wordings(monkeypatch, answer):
    assert "says_documents_lack" in checks_on(rag_trace(monkeypatch, answer))


def test_refusal_followed_by_more(monkeypatch):
    t = rag_trace(monkeypatch, REFUSAL + " That would be investment advice.")
    assert "refusal_with_more" in checks_on(t)
    assert "mentions_internals" not in checks_on(t)   # the stock sentence itself isn't counted


def test_lists_bold_and_asking_back(monkeypatch):
    t = rag_trace(monkeypatch, "Which period do you mean?\n- **2025**: one\n- Q2 2026: two")
    assert {"list_or_bold", "asked_back"} <= checks_on(t)


def test_count_equals_passages(monkeypatch):
    t = rag_trace(monkeypatch, "Kestrel lists four risks: a, b, c and d.", k=4)
    assert "count_equals_passages" in checks_on(t)
    t = rag_trace(monkeypatch, "Kestrel lists six risks.", k=4)
    assert "count_equals_passages" not in checks_on(t)


def test_rewrite_that_isnt_a_query(monkeypatch):
    t = rag_trace(monkeypatch, REFUSAL, rewrite="I need a complete user message to rephrase.")
    assert "rewrite_not_a_query" in checks_on(t)
    assert "rewrite_not_a_query" not in checks_on(rag_trace(monkeypatch, REFUSAL))


@pytest.mark.parametrize("rewrite,flagged", [
    ("Kestrel revenue fiscal year 2025", False),
    ("chief executive officer Kestrel", False),
    ("I'm ready to help you convert a message", True),
    ("What does the user want?", True),
    (" ".join(["word"] * 26), True),
])
def test_rewrite_rule(rewrite, flagged):
    assert m.rewrite_not_a_query(rewrite) is flagged


def test_alert_checks(monkeypatch):
    t = rag_trace(monkeypatch, "See https://example.com. No outside knowledge, no guessing.")
    flags = {f["check"]: f["severity"] for f in m.run_checks(m.view(t))}
    assert flags["link_in_answer"] == "alert" and flags["prompt_leak"] == "alert"


def test_slow_and_low_score_need_no_content(monkeypatch):
    t = rag_trace(monkeypatch, "x", question="weather Omaha", capture=False)
    t["duration_ms"] = 99_000
    got = checks_on(t)
    assert {"slow_request", "low_top_score"} <= got
    # text checks are skipped without content
    assert not got & {"refusal", "says_documents_lack", "rewrite_not_a_query"}


def test_an_errored_request_is_an_alert():
    t = {"schema": tracing.TRACE_SCHEMA, "trace_id": "e" * 32, "status": "error", "duration_ms": 5,
         "content_captured": True, "attributes": {"app.kind": "rag", "traffic.case_id": "t9"},
         "spans": [{"name": "kestrel.request", "kind": "request", "status": "error",
                    "error": "ConnectionError: down", "attributes": {}, "span_id": "a",
                    "parent_id": None, "start_time": "x", "duration_ms": 5}]}
    flags = m.run_checks(m.view(t))
    assert flags == [{"check": "request_error", "severity": "alert",
                      "detail": "ConnectionError: down"}]


# ---- The agent checks -------------------------------------------------------------

def test_declined_without_tools(monkeypatch):
    t = agent_trace(monkeypatch, "When was Kestrel founded?", [text(REFUSAL)])
    assert "declined_without_tools" in checks_on(t)
    t = agent_trace(monkeypatch, "Who founded Kestrel?",
                    [call("search_documents", {"query": "founder"}, "c1")], [text(REFUSAL)])
    assert "declined_without_tools" not in checks_on(t)


def test_unsupported_numbers_from_the_trajectory(monkeypatch):
    """Run 21, t021: two lookups, then a difference nobody calculated."""
    t = agent_trace(monkeypatch, "Compare gross margin in Q1 2025 with Q2 2026.",
                    [call("get_financial_metric", {"metric": "gross_margin", "period": "Q1 2025"}, "c1"),
                     call("get_financial_metric", {"metric": "gross_margin", "period": "Q2 2026"}, "c2")],
                    [text("It rose from 69.8% to 72.1%, an increase of 2.3 percentage points.")])
    flag = next(f for f in m.run_checks(m.view(t)) if f["check"] == "unsupported_numbers")
    assert flag["detail"] == "2.3"


def test_calculated_numbers_are_supported(monkeypatch):
    t = agent_trace(monkeypatch, "Q1 to Q2 2026 revenue change?",
                    [call("calculator", {"expression": "116.0 - 108.3"}, "c1")],
                    [text("Revenue rose by $7.7 million.")])
    assert "unsupported_numbers" not in checks_on(t)


def test_not_reported_sentence(monkeypatch):
    ok = agent_trace(monkeypatch, "q", [call("get_financial_metric",
                                             {"metric": "net_income", "period": "FY2024"}, "c1")],
                     [text("Kestrel did not report net income for FY2024.")])
    assert "not_reported_misused" not in checks_on(ok) and "refusal" in checks_on(ok)
    bad = agent_trace(monkeypatch, "q", [call("search_documents", {"query": "cash flow"}, "c1")],
                      [text("Kestrel did not report operating cash flow for FY2025.")])
    assert "not_reported_misused" in checks_on(bad)


def test_agent_count_uses_the_search_tool_passages(monkeypatch):
    t = agent_trace(monkeypatch, "What risks?", [call("search_documents", {"query": "risks"}, "c1")],
                    [text("Kestrel lists three main risks: x, y and z.")])
    v = m.view(t)
    assert v.passage_counts == [3]   # SEARCH_K
    assert "count_equals_passages" in checks_on(t)


def test_agent_mentions_lookups(monkeypatch):
    t = agent_trace(monkeypatch, "q", [text("I can look up FY2025 revenue if you like.")])
    assert "mentions_internals" in checks_on(t)


def test_step_limit_is_an_alert(monkeypatch):
    t = agent_trace(monkeypatch, "q", [text("x")])
    run = next(s for s in t["spans"] if s["name"] == "agent.run")
    run["attributes"]["agent.stopped"] = "step_limit"
    assert "agent_stopped_early" in checks_on(t)


# ---- Helpers ----------------------------------------------------------------------

def test_not_reported_parsing():
    assert m.not_reported_parts("Kestrel did not report diluted EPS for FY2024.") == \
        ("diluted EPS", "FY2024")
    assert m.is_valid_not_reported("Kestrel did not report diluted EPS for FY2024.")
    assert not m.is_valid_not_reported("Kestrel did not report EBITDA for FY2024.")
    assert m.not_reported_parts("Revenue was $1.") is None


def test_stated_counts_and_sentences():
    assert m.stated_counts("It lists three main risks and 2 segments.") == [3, 2]
    assert m.stated_counts("Revenue grew 19%.") == []
    assert m.sentences("One. Two! Three? Four.") == 4
    assert m.sentences("Revenue was $412.6 million.") == 1


def test_sample_is_deterministic_and_about_the_right_size():
    ids = [f"{i:032x}" for i in range(2000)]
    picked = [i for i in ids if m.in_sample(i, 0.25)]
    assert picked == [i for i in ids if m.in_sample(i, 0.25)]
    assert 400 < len(picked) < 600
    assert not m.in_sample(ids[0], 0) and m.in_sample(ids[0], 1)


# ---- The judge, faked -------------------------------------------------------------

def fake_judge(faithful=True, error=None):
    calls = []

    def judge(question, contexts, reference, answer):
        calls.append({"question": question, "contexts": contexts, "reference": reference})
        if error:
            raise error
        return SimpleNamespace(faithful=faithful, unsupported_claims=[] if faithful else ["claim X"],
                               reasoning="r", input_tokens=10, cached_input_tokens=0, output_tokens=5)
    judge.calls = calls
    return judge


def test_judge_skips_refusals_and_uses_the_passages(monkeypatch):
    answer = rag_trace(monkeypatch, "Revenue was $108.3 million.", case_id="t1")
    refusal = rag_trace(monkeypatch, REFUSAL, case_id="t2")
    judge = fake_judge(faithful=False)
    result = m.monitor([answer, refusal], judge_sample=1.0, judge=judge)
    assert len(judge.calls) == 1
    assert judge.calls[0]["reference"] == m.NO_REFERENCE
    assert len(judge.calls[0]["contexts"]) == 4 and "108.3" in " ".join(judge.calls[0]["contexts"])
    assert result["judge"]["judged"] == 1 and result["judge"]["unfaithful"] == 1
    assert result["checks"]["unfaithful"]["cases"] == ["t1"]
    assert result["checks"]["unfaithful"]["eligible"] == 1


def test_a_judge_error_is_recorded_not_counted(monkeypatch):
    t = rag_trace(monkeypatch, "Revenue was $108.3 million.")
    result = m.monitor([t], judge_sample=1.0, judge=fake_judge(error=RuntimeError("API down")))
    assert result["judge"]["errors"] == 1 and result["judge"]["judged"] == 0
    assert result["checks"]["unfaithful"]["count"] == 0


def test_agent_judge_contexts_are_tool_outputs(monkeypatch):
    t = agent_trace(monkeypatch, "Q1 2026 revenue?",
                    [call("get_financial_metric", {"metric": "revenue", "period": "Q1 2026"}, "c1")],
                    [text("Revenue was $108.3 million in Q1 2026.")])
    judge = fake_judge()
    m.monitor([t], judge_sample=1.0, judge=judge)
    assert judge.calls[0]["contexts"][0].startswith("revenue, Q1 2026: $108.3 million")


def test_no_judge_calls_at_rate_zero(monkeypatch):
    t = rag_trace(monkeypatch, "Revenue was $108.3 million.")
    judge = fake_judge()
    m.monitor([t], judge_sample=0, judge=judge)
    assert judge.calls == []


# ---- Window rules -----------------------------------------------------------------

def window(monkeypatch, answers, app="rag"):
    return [rag_trace(monkeypatch, a, case_id=f"t{i:03d}") for i, a in enumerate(answers)]


def rules_fired(result):
    return {a["rule"]: a["level"] for a in result["alerts"]}


def test_small_windows_dont_judge_rates_but_still_alert(monkeypatch):
    traces = window(monkeypatch, ["See www.example.com"] * 3)
    fired = rules_fired(m.monitor(traces))
    assert fired == {"link_in_answer": "alert", "too_few_requests": "info"}


def test_a_healthy_window_is_quiet(monkeypatch):
    answers = ["Revenue was $108.3 million."] * 15 + [REFUSAL] * 5
    assert m.monitor(window(monkeypatch, answers))["alerts"] == []


def test_window_rules(monkeypatch):
    answers = (["Revenue was $108.3 million."] * 10
               + ["The documents do not contain that."] * 10)
    traces = window(monkeypatch, answers)
    traces[0]["status"] = "error"
    for t in traces:
        t["duration_ms"] = 4000
    result = m.monitor(traces)
    fired = rules_fired(result)
    assert fired["error_rate"] == "alert" and fired["request_error"] == "alert"
    assert fired["p95_latency"] == "warn"
    assert fired["says_documents_lack"] == "warn"
    assert fired["queue_size"] == "warn"


def test_refusal_rate_band(monkeypatch):
    fired = rules_fired(m.monitor(window(monkeypatch, [REFUSAL] * 20)))
    assert fired["refusal_rate"] == "warn"


def test_cost_budget(monkeypatch):
    traces = window(monkeypatch, ["Revenue was $108.3 million."] * 15 + [REFUSAL] * 5)
    for t in traces:
        for s in t["spans"]:
            if s["kind"] == "llm":
                s["attributes"]["gen_ai.usage.input_tokens"] = 2_000
    assert rules_fired(m.monitor(traces))["cost"] == "warn"


# ---- Scoring the checks against a person's review --------------------------------

def test_score_against_reviews():
    result = {"apps": ["rag"], "review_queue": [{"case": "t1", "app": "rag"},
                                                {"case": "t2", "app": "rag"}]}
    reviews = [{"app": "rag", "case_id": "t1", "verdict": "wrong"},
               {"app": "rag", "case_id": "t3", "verdict": "rule_break"},
               {"app": "agent", "case_id": "t2", "verdict": "wrong"}]
    s = m.score_against_reviews(result, reviews, ["t1", "t2", "t3", "t4"])["rag"]
    assert (s["problems"], s["queued"], s["queued_with_problem"]) == (2, 2, 1)
    assert s["precision"] == 0.5 and s["recall"] == 0.5
    assert s["missed"] == ["t3"] and s["false_alarms"] == ["t2"]
    assert s["by_verdict"]["wrong"] == {"total": 1, "queued": 1}


def test_reviews_v1_is_well_formed():
    reviews = m.load_reviews(TRAFFIC_DIR / "reviews_v1.jsonl")
    ids = {r["id"] for r in traffic.load_traffic(TRAFFIC_DIR / "traffic_v1.jsonl")}
    assert {r["case_id"] for r in reviews} <= ids
    assert {r["app"] for r in reviews} == {"rag", "agent"}
    assert all(r["note"] for r in reviews)
    keys = [(r["app"], r["case_id"]) for r in reviews]
    assert len(keys) == len(set(keys))


def test_load_reviews_rejects_bad_rows(tmp_path):
    bad = tmp_path / "r.jsonl"
    bad.write_text(json.dumps({"app": "rag", "case_id": "t1", "verdict": "meh"}) + "\n",
                   encoding="utf-8")
    with pytest.raises(ValueError):
        m.load_reviews(bad)


def test_traffic_v2_is_held_out():
    v2 = traffic.load_traffic(TRAFFIC_DIR / "traffic_v2.jsonl")
    v1 = traffic.load_traffic(TRAFFIC_DIR / "traffic_v1.jsonl")
    assert len(v2) == 30 and [r["id"] for r in v2] == [f"u{i:03d}" for i in range(1, 31)]
    taken = eval_questions() | {normalize(r["question"]) for r in v1}
    assert [r["id"] for r in v2 if normalize(r["question"]) in taken] == []
    assert not {r["session"] for r in v2} & {r["session"] for r in v1}


# ---- The command and the dashboard ------------------------------------------------

def test_cli_writes_results_and_exits_1_on_alert(monkeypatch, tmp_path, capsys):
    path = tmp_path / "traces_x.jsonl"
    exporter = tracing.JsonlExporter(path)
    for t in window(monkeypatch, ["Revenue was $108.3 million."] * 20 + ["www.x.com"]):
        exporter.export(t)
    code = monitor_cli.main([str(path), "--results-dir", str(tmp_path), "--queue"])
    out = capsys.readouterr().out
    assert code == 1 and "ALERT" in out and "link_in_answer" in out
    saved = json.loads((tmp_path / "monitor_rag_latest.json").read_text(encoding="utf-8"))
    assert saved["trace_file"] == "traces_x.jsonl" and saved["requests"] == 21


def test_cli_strict_fails_on_warnings(monkeypatch, tmp_path):
    path = tmp_path / "traces_y.jsonl"
    exporter = tracing.JsonlExporter(path)
    for t in window(monkeypatch, ["Revenue was $108.3 million."] * 10
                                 + ["The documents do not contain that."] * 10):
        exporter.export(t)
    assert monitor_cli.main([str(path), "--no-save"]) == 0
    assert monitor_cli.main([str(path), "--no-save", "--strict"]) == 1


def test_cli_rejects_bad_sample_rate(tmp_path):
    assert monitor_cli.main([str(tmp_path / "none.jsonl"), "--judge-sample", "2"]) == 2


def test_dashboard_monitoring_section(monkeypatch, tmp_path):
    assert "No monitor results yet" in render_html(build_dashboard(tmp_path))
    result = m.monitor(window(monkeypatch, ["Revenue was $108.3 million."] * 20 + ["<b>x</b> www.x.com"]))
    result["trace_file"] = "traces_z.jsonl"
    (tmp_path / "monitor_rag_latest.json").write_text(json.dumps(result), encoding="utf-8")
    page = render_html(build_dashboard(tmp_path))
    assert "How is production doing?" in page and "RAG assistant" in page
    assert "Alerts firing" in page and "traces_z.jsonl" in page
    assert "<b>x</b>" not in page   # answers are escaped


# ---- Fixes after Run 22 (m2) ------------------------------------------------------

def test_french_digit_groups_are_one_number(monkeypatch):
    """Run 22, u025: "2 850" (French thousands) was read as 2 and 850."""
    t = agent_trace(monkeypatch, "Combien de clients entreprises fin 2025 ?",
                    [call("get_financial_metric",
                          {"metric": "enterprise_customers", "period": "FY2025"}, "c1")],
                    [text("Kestrel avait 2 850 clients entreprises à la fin de 2025.")])
    assert "unsupported_numbers" not in checks_on(t)
    assert m.DIGIT_GROUP_RE.sub("", "Q2 2026 and 2 850") == "Q2 2026 and 2850"


def test_do_not_identify_counts_as_saying_the_documents_lack_it(monkeypatch):
    """Run 22, agent u022."""
    t = agent_trace(monkeypatch, "Who founded Kestrel?",
                    [call("search_documents", {"query": "founder"}, "c1")],
                    [text("Kestrel was founded in 2017. The documents do not identify the founder.")])
    assert "says_documents_lack" in checks_on(t)


def test_reviews_v2_is_well_formed():
    reviews = m.load_reviews(TRAFFIC_DIR / "reviews_v2.jsonl")
    ids = {r["id"] for r in traffic.load_traffic(TRAFFIC_DIR / "traffic_v2.jsonl")}
    assert {r["case_id"] for r in reviews} <= ids and all(r["note"] for r in reviews)


def test_decimal_comma_is_a_decimal_point(monkeypatch):
    """m3: v7 answers in the user's language, so "412,6" means 412.6."""
    t = agent_trace(monkeypatch, "Chiffre d'affaires 2025 ?",
                    [call("get_financial_metric", {"metric": "revenue", "period": "FY2025"}, "c1")],
                    [text("Le chiffre d'affaires était de 412,6 millions de dollars en 2025.")])
    assert "unsupported_numbers" not in checks_on(t)
    assert m.plain_numbers("3,120 et 412,6 et 2 850") == "3,120 et 412.6 et 2850"
