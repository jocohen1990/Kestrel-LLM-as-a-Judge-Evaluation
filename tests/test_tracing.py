"""Offline tests for production-style tracing (app/tracing.py, app/service.py).

Three things matter most and each has tests here:
  1. Outside a trace nothing changes: every eval runs exactly as before.
  2. Inside a trace the spans form the right tree, with the right numbers.
  3. Tracing never breaks a request: errors are recorded and re-raised as they
     were, and a failing exporter only prints a warning.
"""
import json
import threading
from types import SimpleNamespace

import pytest

from app import llm, rag, tracing
from app.agent import KestrelAgent
from app.llm import LLMResponse
from app.service import handle
from tests.test_agent import call, step_through, text, use_model


@pytest.fixture
def exporter():
    return tracing.MemoryExporter()


def by_name(trace, name):
    return [s for s in trace["spans"] if s["name"] == name]


def one(trace, name):
    spans = by_name(trace, name)
    assert len(spans) == 1, f"expected one {name!r} span, got {len(spans)}"
    return spans[0]


def parent_of(trace, span):
    return next(s for s in trace["spans"] if s["span_id"] == span["parent_id"])


# ---- The tracer ----------------------------------------------------------------

def test_outside_a_trace_spans_do_nothing():
    assert not tracing.active()
    with tracing.span("anything", kind="llm") as s:
        assert s is tracing.NO_SPAN
        s.set({"a": 1}).set_content("b", "text")   # accepted, ignored
    assert tracing.current_span() is None


def test_nested_spans_make_a_tree(exporter):
    with tracing.trace("req", attrs={"request.id": "r1"}, exporter=exporter) as root:
        with tracing.span("outer", kind="chain") as outer:
            with tracing.span("inner", kind="llm") as inner:
                inner.set({"gen_ai.usage.input_tokens": 10, "gen_ai.usage.output_tokens": 2,
                           "gen_ai.request.model": "m"})
        root.set(done=True)
    [t] = exporter.traces
    assert t["schema"] == tracing.TRACE_SCHEMA
    assert [s["name"] for s in t["spans"]] == ["req", "outer", "inner"]   # start order
    req, outer_d, inner_d = t["spans"]
    assert req["parent_id"] is None
    assert outer_d["parent_id"] == req["span_id"] and inner_d["parent_id"] == outer_d["span_id"]
    assert {s["span_id"] for s in t["spans"]} == {req["span_id"], outer.span_id, inner.span_id}
    assert all(s["duration_ms"] >= 0 for s in t["spans"])
    assert req["duration_ms"] >= outer_d["duration_ms"] >= inner_d["duration_ms"]
    assert len(t["trace_id"]) == 32 and len(req["span_id"]) == 16
    # the root carries the request's totals
    assert t["attributes"]["request.id"] == "r1" and t["attributes"]["done"] is True
    assert t["attributes"]["llm.calls"] == 1
    assert t["attributes"]["usage.input_tokens"] == 10
    assert t["attributes"]["llm.models"] == ["m"]
    assert t["status"] == "ok"
    assert not tracing.active()   # nothing left open


def test_an_error_is_recorded_and_still_raised(exporter):
    with pytest.raises(ZeroDivisionError):
        with tracing.trace("req", exporter=exporter):
            with tracing.span("step", kind="tool"):
                1 / 0
    [t] = exporter.traces
    assert t["status"] == "error"
    step = one(t, "step")
    assert step["status"] == "error" and step["error"].startswith("ZeroDivisionError")
    assert not tracing.active()


def test_a_failing_exporter_never_breaks_the_request(capsys):
    class Broken:
        def export(self, trace):
            raise OSError("disk full")

    with tracing.trace("req", exporter=Broken()) as root:
        root.set(ok=True)
    assert "was not exported: disk full" in capsys.readouterr().err


def test_a_trace_inside_a_trace_is_just_a_span(exporter):
    with tracing.trace("outer", exporter=exporter):
        with tracing.trace("inner", exporter=exporter):
            pass
    assert len(exporter.traces) == 1
    assert [s["name"] for s in exporter.traces[0]["spans"]] == ["outer", "inner"]


def test_content_is_dropped_when_capture_is_off(exporter):
    with tracing.trace("req", exporter=exporter, capture_content=False) as root:
        root.set_content("input.question", "secret question")
        root.set(count=3)
    [t] = exporter.traces
    assert "input.question" not in t["attributes"] and t["attributes"]["count"] == 3
    assert t["content_captured"] is False


def test_capture_default_comes_from_the_environment(monkeypatch, exporter):
    monkeypatch.setenv("KESTREL_TRACE_CONTENT", "0")
    with tracing.trace("req", exporter=exporter) as root:
        root.set_content("input.question", "q")
    assert "input.question" not in exporter.traces[0]["attributes"]


def test_long_text_is_clipped(exporter):
    with tracing.trace("req", exporter=exporter) as root:
        root.set_content("long", "x" * (tracing.MAX_CONTENT_CHARS + 50))
        root.set_content("list", ["y" * (tracing.MAX_CONTENT_CHARS + 1)])
    a = exporter.traces[0]["attributes"]
    assert a["long"].endswith("[50 more chars]")
    assert a["list"][0].endswith("[1 more chars]")


def test_unknown_span_kind_is_rejected():
    with pytest.raises(ValueError):
        with tracing.span("x", kind="banana"):
            pass


def test_explicit_parent_works_in_a_thread_that_cant_see_the_trace(exporter):
    """The agent's case: a node may run in a worker thread. A plain Thread starts
    with an EMPTY context, so without `parent=` its spans would be lost."""
    seen = {}
    with tracing.trace("req", exporter=exporter):
        parent = tracing.current_span()

        def worker():
            seen["active_before"] = tracing.active()
            with tracing.span("in-thread", kind="chain", parent=parent):
                with tracing.span("nested", kind="llm"):
                    pass

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
    assert seen["active_before"] is False
    t = exporter.traces[0]
    assert parent_of(t, one(t, "in-thread"))["name"] == "req"
    assert parent_of(t, one(t, "nested"))["name"] == "in-thread"


def test_jsonl_exporter_writes_one_line_per_trace(tmp_path):
    path = tmp_path / "sub" / "traces.jsonl"
    exporter = tracing.JsonlExporter(path)
    for i in range(2):
        with tracing.trace("req", attrs={"i": i}, exporter=exporter):
            pass
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["attributes"]["i"] for line in lines] == [0, 1]


# ---- The model wrapper (app/llm.py) -----------------------------------------------

class FakeClient:
    def __init__(self, content=None, stop_reason="end_turn"):
        self.messages = self
        self.content = content or [SimpleNamespace(type="text", text="hello")]
        self.stop_reason = stop_reason

    def create(self, **request):
        return SimpleNamespace(content=self.content, stop_reason=self.stop_reason,
                               usage=SimpleNamespace(input_tokens=30, output_tokens=4,
                                                     cache_read_input_tokens=10))


@pytest.fixture
def fake_client(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(llm, "_client", client)
    monkeypatch.setattr(llm, "_NO_TEMPERATURE_MODELS", set())
    monkeypatch.setattr(llm, "_NO_FORCED_TOOL_MODELS", set())
    return client


def test_every_model_call_is_an_llm_span(fake_client, exporter):
    with tracing.trace("req", exporter=exporter):
        llm.generate("sys", "hi", model="claude-haiku-4-5", max_tokens=50)
    span = one(exporter.traces[0], "llm.chat")
    a = span["attributes"]
    assert span["kind"] == "llm"
    assert a["gen_ai.request.model"] == "claude-haiku-4-5"
    assert a["gen_ai.request.max_tokens"] == 50
    assert a["gen_ai.usage.input_tokens"] == 40          # 30 + 10 cached, as llm.py counts
    assert a["gen_ai.usage.cache_read_input_tokens"] == 10
    assert a["gen_ai.usage.output_tokens"] == 4
    assert a["gen_ai.response.finish_reasons"] == ["end_turn"]
    assert a["gen_ai.request.temperature"] == 0.0
    assert a["gen_ai.output"] == "hello"
    assert a["llm.api_latency_ms"] >= 0


def test_tool_turn_span_counts_tool_calls(fake_client, exporter):
    fake_client.content = [SimpleNamespace(type="text", text="Let me check."),
                           SimpleNamespace(type="tool_use", id="t1", name="calculator",
                                           input={"expression": "1+1"})]
    with tracing.trace("req", exporter=exporter):
        llm.generate_with_tools("sys", [{"role": "user", "content": "q"}], [], model="m")
    a = one(exporter.traces[0], "llm.chat")["attributes"]
    assert a["llm.tool_calls"] == 1 and a["gen_ai.output"] == "Let me check."


def test_structured_call_is_traced(fake_client, exporter):
    fake_client.content = [SimpleNamespace(type="tool_use", name="grade", input={"ok": True})]
    with tracing.trace("req", exporter=exporter):
        llm.generate_structured("sys", "u", {"type": "object"}, "grade", "d", model="m")
    a = one(exporter.traces[0], "llm.chat")["attributes"]
    assert a["llm.forced_tool_choice"] is True


def test_a_failed_model_call_is_an_error_span(monkeypatch, exporter):
    class Down:
        messages = None

        def create(self, **request):
            raise ConnectionError("API unreachable")

    client = Down()
    client.messages = client
    monkeypatch.setattr(llm, "_client", client)
    with pytest.raises(ConnectionError):
        with tracing.trace("req", exporter=exporter):
            llm.generate("sys", "hi", model="m")
    span = one(exporter.traces[0], "llm.chat")
    assert span["status"] == "error" and "API unreachable" in span["error"]


# ---- The RAG assistant ------------------------------------------------------------

def fake_generate(calls):
    def _fake(system, user, model=None, **kwargs):
        calls.append(system)
        # Calls the real span code, the way app/llm.py does, so the tree has
        # llm spans without a network.
        with llm._llm_span(model or "claude-haiku-4-5", kwargs.get("max_tokens", 400)) as span:
            reply = "revenue Q1 2026" if system == rag.REWRITE_SYSTEM else "Revenue was $108.3 million."
            span.set({"gen_ai.usage.input_tokens": 100, "gen_ai.usage.output_tokens": 10})
        return LLMResponse(text=reply, model="claude-haiku-4-5", input_tokens=100,
                           output_tokens=10, latency_s=0.01)
    return _fake


def test_rag_request_trace_tree(monkeypatch, exporter):
    calls = []
    monkeypatch.setattr(rag, "generate", fake_generate(calls))
    assistant = rag.RagAssistant(k=3, retriever_version="r3")
    result = handle("What was Kestrel's revenue in Q1 2026?", assistant, request_id="req-1",
                    session_id="s1", channel="web", exporter=exporter,
                    extra={"traffic.case_id": "t999"})
    assert result.answer == "Revenue was $108.3 million."
    [t] = exporter.traces
    names = [s["name"] for s in t["spans"]]
    assert names == ["kestrel.request", "rag.answer", "retrieve", "query_rewrite", "llm.chat",
                     "bm25.search", "generate", "llm.chat"]
    assert parent_of(t, one(t, "query_rewrite"))["name"] == "retrieve"
    assert parent_of(t, one(t, "generate"))["name"] == "rag.answer"

    root = t["attributes"]
    assert root["request.id"] == "req-1" and root["session.id"] == "s1"
    assert root["request.channel"] == "web" and root["app.kind"] == "rag"
    assert root["kestrel.prompt_version"] == rag.DEFAULT_PROMPT_VERSION
    assert root["kestrel.retriever_version"] == "r3"
    assert root["traffic.case_id"] == "t999"
    assert root["llm.calls"] == 2 and root["usage.input_tokens"] == 200
    assert root["input.question"].startswith("What was Kestrel's revenue")
    assert root["output.answer"] == result.answer

    search = one(t, "bm25.search")["attributes"]
    assert search["retrieval.docs"][0] == "q1_2026_earnings.md"
    assert len(search["retrieval.scores"]) == 3
    assert search["retrieval.scores"] == sorted(search["retrieval.scores"], reverse=True)
    assert search["retrieval.top_score"] == search["retrieval.scores"][0] > 0
    assert len(search["retrieval.passages"]) == 3
    assert search["retrieval.query"].endswith("revenue Q1 2026")
    assert one(t, "query_rewrite")["attributes"]["rewrite.output"] == "revenue Q1 2026"


def test_r1_has_no_rewrite_span(monkeypatch, exporter):
    monkeypatch.setattr(rag, "generate", fake_generate([]))
    handle("Who is Kestrel's CEO?", rag.RagAssistant(retriever_version="r1"), exporter=exporter)
    assert not by_name(exporter.traces[0], "query_rewrite")
    assert exporter.traces[0]["attributes"]["llm.calls"] == 1


def test_tracing_does_not_change_the_answer(monkeypatch, exporter):
    """The same question, traced and not: identical result, so the evals (never
    traced) measure the same app that production runs."""
    monkeypatch.setattr(rag, "generate", fake_generate([]))
    assistant = rag.RagAssistant()
    plain = assistant.answer("What was Kestrel's gross margin in Q2 2026?")
    traced = handle("What was Kestrel's gross margin in Q2 2026?", assistant, exporter=exporter)
    assert plain == traced


def test_search_scored_matches_search():
    retriever = rag.RagAssistant().retriever
    for q in ("Who is the CEO?", "LaneSight congestion", "revolving credit facility drawn"):
        scored = retriever.search_scored(q, k=4)
        assert [c for c, _ in scored] == retriever.search(q, k=4)
        assert [s for _, s in scored] == sorted((s for _, s in scored), reverse=True)


def test_no_content_trace_keeps_numbers_only(monkeypatch, exporter):
    monkeypatch.setattr(rag, "generate", fake_generate([]))
    handle("What was Kestrel's revenue in Q1 2026?", rag.RagAssistant(), exporter=exporter,
           capture_content=False)
    blob = json.dumps(exporter.traces[0])
    assert "Kestrel's revenue" not in blob and "108.3" not in blob and "Annual Report" not in blob
    search = one(exporter.traces[0], "bm25.search")["attributes"]
    assert "retrieval.top_score" in search and "retrieval.passages" not in search


# ---- The agent --------------------------------------------------------------------

def test_agent_turns_and_tools_are_spans(monkeypatch, exporter):
    use_model(monkeypatch,
              [call("get_financial_metric", {"metric": "revenue", "period": "Q1 2026"}, "c1")],
              [text("Revenue was $108.3 million in Q1 2026.")])
    with tracing.trace("req", exporter=exporter):
        result = step_through(KestrelAgent(), "Q1 2026 revenue?")
    [t] = exporter.traces
    turns = by_name(t, "agent.turn")
    assert [s["attributes"]["agent.turn"] for s in turns] == [1, 2]
    assert turns[0]["attributes"]["agent.tool_calls"] == ["get_financial_metric"]
    assert turns[1]["attributes"]["agent.tool_calls"] == []
    tool = one(t, "tool.get_financial_metric")
    assert tool["kind"] == "tool" and tool["attributes"]["tool.is_error"] is False
    assert json.loads(tool["attributes"]["tool.arguments"]) == {"metric": "revenue",
                                                                "period": "Q1 2026"}
    assert "108.3" in tool["attributes"]["tool.output"]
    assert result.answer.startswith("Revenue was $108.3 million")


def test_agent_tool_error_is_recorded(monkeypatch, exporter):
    use_model(monkeypatch, [call("calculator", {"expression": "import os"}, "c1")],
              [text("I don't know.")])
    with tracing.trace("req", exporter=exporter):
        step_through(KestrelAgent(), "q")
    assert one(exporter.traces[0], "tool.calculator")["attributes"]["tool.is_error"] is True


def test_agent_nodes_use_the_parent_from_the_run_config(monkeypatch, exporter):
    """What LangGraph does: call the node with the run's config, possibly in a
    thread that can't see the trace. The node's spans must still hang under the
    span run() put in the config, not get lost."""
    from app.agent import TRACE_PARENT_KEY, initial_state
    use_model(monkeypatch, [call("calculator", {"expression": "2+2"}, "c1")])
    agent = KestrelAgent()
    with tracing.trace("req", exporter=exporter):
        with tracing.span("agent.run", kind="agent") as run_span:
            config = {"configurable": {TRACE_PARENT_KEY: run_span}}
            state = initial_state("q")

            def node_in_thread():
                state.update(agent.call_model(state, config))
                state.update(agent.run_tools(state, config))

            thread = threading.Thread(target=node_in_thread)
            thread.start()
            thread.join()
    t = exporter.traces[0]
    for name in ("agent.turn", "tool.calculator"):
        assert parent_of(t, one(t, name))["name"] == "agent.run"


def test_agent_run_span_records_the_outcome(exporter):
    from app.agent import AgentResult, record_result
    result = AgentResult(question="q", answer="a", turns=2, stopped="answered", model="m",
                         agent_version="a3",
                         steps=[{"tool": "calculator", "is_error": True},
                                {"tool": "calculator", "is_error": False}])
    with tracing.trace("req", exporter=exporter):
        with tracing.span("agent.run", kind="agent") as span:
            record_result(span, result)
    a = one(exporter.traces[0], "agent.run")["attributes"]
    assert a["agent.stopped"] == "answered" and a["agent.turns"] == 2
    assert a["agent.tools_used"] == ["calculator", "calculator"] and a["agent.tool_errors"] == 1
    assert a["output.answer"] == "a"


def test_agent_nodes_untraced_still_work(monkeypatch):
    use_model(monkeypatch, [text("hello")])
    assert step_through(KestrelAgent(), "q").answer == "hello"


def test_service_rejects_an_unknown_app():
    with pytest.raises(TypeError):
        handle("q", object())


def test_service_routes_agents_to_run(exporter):
    class FakeAgent:
        agent_version = "a3"

        def run(self, question):
            return SimpleNamespace(answer=f"ran: {question}")

    result = handle("hi", FakeAgent(), exporter=exporter)
    assert result.answer == "ran: hi"
    a = exporter.traces[0]["attributes"]
    assert a["app.kind"] == "agent" and a["kestrel.agent_version"] == "a3"
    assert "kestrel.tools_version" in a


def test_real_langgraph_run_is_one_tree(monkeypatch, exporter):
    """The real graph (skipped without LangGraph). Whatever thread LangGraph runs a
    node in, every turn and tool span must hang under agent.run."""
    pytest.importorskip("langgraph")
    from app.agent import trace_parent
    use_model(monkeypatch,
              [call("get_financial_metric", {"metric": "revenue", "period": "Q2 2026"}, "c1")],
              [text("Revenue was $116.0 million in Q2 2026.")])
    received = []

    class RecordingAgent(KestrelAgent):
        def call_model(self, state, config=None):
            received.append(trace_parent(config))
            return super().call_model(state, config)

    result = handle("Q2 2026 revenue?", RecordingAgent(), exporter=exporter)
    # LangGraph really handed the run's config to the node
    assert len(received) == 2 and all(p is not None and p.name == "agent.run" for p in received)
    [t] = exporter.traces
    run = one(t, "agent.run")
    assert parent_of(t, run)["name"] == "kestrel.request"
    for name in ("agent.turn", "tool.get_financial_metric"):
        for s in by_name(t, name):
            assert s["parent_id"] == run["span_id"]
    assert len(by_name(t, "agent.turn")) == 2
    assert run["attributes"]["agent.stopped"] == "answered"
    assert result.answer.startswith("Revenue was $116.0 million")
