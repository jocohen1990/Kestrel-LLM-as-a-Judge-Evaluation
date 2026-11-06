"""Offline tests for reading traces (evals/traces.py), the traffic data and the
traffic.py / traces.py scripts. The model is faked, so nothing here costs money."""
import json

import pytest

import traces as traces_cli
import traffic
from app import rag, tracing
from app.paths import TRAFFIC_DIR
from evals import traces as tr
from tests.test_tracing import fake_generate


# ---- Building traces to read ---------------------------------------------------

def make_trace(duration_ms=1000.0, model="claude-haiku-4-5", tokens=(1000, 100),
               status="ok", case_id="t001", top_score=5.0, extra_spans=()):
    root_id = "a" * 16
    spans = [
        {"name": "kestrel.request", "kind": "request", "span_id": root_id, "parent_id": None,
         "start_time": "2026-11-05T00:00:00.000+00:00", "duration_ms": duration_ms,
         "status": status, "error": None if status == "ok" else "ConnectionError: down",
         "attributes": {}},
        {"name": "bm25.search", "kind": "retrieval", "span_id": "b" * 16, "parent_id": root_id,
         "start_time": "x", "duration_ms": 2.0, "status": "ok", "error": None,
         "attributes": {"retrieval.top_score": top_score, "retrieval.docs": ["q1.md", "q1.md"]}},
        {"name": "llm.chat", "kind": "llm", "span_id": "c" * 16, "parent_id": root_id,
         "start_time": "x", "duration_ms": duration_ms - 10, "status": "ok", "error": None,
         "attributes": {"gen_ai.request.model": model, "gen_ai.usage.input_tokens": tokens[0],
                        "gen_ai.usage.output_tokens": tokens[1],
                        "gen_ai.response.finish_reasons": ["end_turn"]}},
        *extra_spans,
    ]
    return {"schema": tracing.TRACE_SCHEMA, "trace_id": (case_id * 8)[:32], "name": "kestrel.request",
            "start_time": "2026-11-05T00:00:00.000+00:00", "duration_ms": duration_ms,
            "status": status, "content_captured": True,
            "attributes": {"app.kind": "rag", "service.version": "dev",
                           "kestrel.prompt_version": "v6", "kestrel.retriever_version": "r3",
                           "traffic.case_id": case_id, "request.id": f"req-{case_id}",
                           "request.channel": "web", "llm.calls": 1,
                           "usage.input_tokens": tokens[0], "usage.output_tokens": tokens[1],
                           "llm.models": [model], "input.question": "Q?",
                           "output.answer": "A."},
            "spans": spans}


# ---- Numbers -------------------------------------------------------------------

def test_percentile_is_nearest_rank():
    values = list(range(1, 21))   # 1..20
    assert tr.percentile(values, 50) == 10
    assert tr.percentile(values, 95) == 19
    assert tr.percentile(values, 100) == 20
    assert tr.percentile([7], 95) == 7
    assert tr.percentile([], 95) is None


def test_cost_uses_the_price_table():
    # Haiku 4.5: $1 per MTok in, $5 per MTok out
    assert tr.trace_cost(make_trace(tokens=(1_000_000, 0))) == pytest.approx(1.0)
    assert tr.trace_cost(make_trace(tokens=(0, 1_000_000))) == pytest.approx(5.0)
    assert tr.trace_cost(make_trace(model="claude-haiku-4-5-20251001", tokens=(1_000_000, 0))) \
        == pytest.approx(1.0)
    assert tr.trace_cost(make_trace(model="some-unpriced-model")) is None


def test_summary_numbers():
    traces = [make_trace(duration_ms=d * 100.0, case_id=f"t{d:03d}", top_score=d)
              for d in range(1, 21)]
    traces.append(make_trace(duration_ms=50.0, status="error", case_id="t099"))
    s = tr.summarize(traces)
    assert s["requests"] == 21 and s["errors"] == 1
    assert s["error_rate"] == round(1 / 21, 3)
    assert s["error_messages"] == {"ConnectionError: down": 1}
    assert s["latency"]["p50_ms"] == 1000.0 and s["latency"]["max_ms"] == 2000.0
    assert s["span_latency"]["llm.chat"]["count"] == 21
    assert "kestrel.request" not in s["span_latency"]
    assert s["llm_calls_per_request"] == 1.0
    assert s["cost_usd"] == pytest.approx(21 * (1000 * 1 + 100 * 5) / 1e6)
    assert s["cost_per_1000_requests"] == pytest.approx(1.5)
    assert s["retrieval_top_score"]["min"] == 1
    assert s["setups"] == {"kind=rag version=dev prompt_version=v6 retriever_version=r3": 21}
    json.dumps(s)   # the summary must be saveable as JSON


def test_summary_counts_agent_runs_and_tools():
    agent_spans = [
        {"name": "agent.run", "kind": "agent", "span_id": "d" * 16, "parent_id": "a" * 16,
         "start_time": "x", "duration_ms": 900, "status": "ok", "error": None,
         "attributes": {"agent.stopped": "answered", "agent.turns": 3}},
        {"name": "tool.calculator", "kind": "tool", "span_id": "e" * 16, "parent_id": "d" * 16,
         "start_time": "x", "duration_ms": 1, "status": "ok", "error": None,
         "attributes": {"tool.is_error": True}},
    ]
    s = tr.summarize([make_trace(extra_spans=agent_spans)])
    assert s["agent"]["runs"] == 1 and s["agent"]["avg_turns"] == 3
    assert s["agent"]["tool_calls"] == {"calculator": 1} and s["agent"]["tool_errors"] == 1
    assert "Agent: 1 runs" in tr.format_summary(s)


def test_empty_summary():
    s = tr.summarize([])
    assert s["requests"] == 0 and s["latency"]["p50_ms"] is None
    assert tr.format_summary(s) == "No traces."


def test_find_trace_by_case_request_or_trace_id():
    t = make_trace(case_id="t007")
    assert tr.find_trace([t], "t007") is t
    assert tr.find_trace([t], "req-t007") is t
    assert tr.find_trace([t], t["trace_id"][:6]) is t
    assert tr.find_trace([t], "nope") is None


def test_load_skips_bad_lines(tmp_path, capsys):
    path = tmp_path / "traces_1.jsonl"
    good = make_trace()
    path.write_text("\n".join([json.dumps(good), "{not json", json.dumps({"schema": "tr0"}), ""]),
                    encoding="utf-8")
    assert tr.load_traces(path) == [good]
    assert "skipped 2" in capsys.readouterr().out


def test_load_a_folder_reads_every_file_in_order(tmp_path):
    for name, case in (("traces_20261102_000000.jsonl", "t002"),
                       ("traces_20261101_000000.jsonl", "t001")):
        (tmp_path / name).write_text(json.dumps(make_trace(case_id=case)) + "\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    loaded = tr.load_traces(tmp_path)
    assert [t["attributes"]["traffic.case_id"] for t in loaded] == ["t001", "t002"]


def test_tree_shows_each_span_once_under_its_parent():
    text = tr.render_tree(make_trace(case_id="t003"))
    lines = text.splitlines()
    assert lines[0].startswith("trace ")
    assert "Q: Q?" in text and "A: A." in text and "cost $" in text
    assert sum("llm.chat" in l for l in lines) == 1
    assert "haiku-4-5  1000 in / 100 out  stop=end_turn" in text
    assert "q1 x2  top score 5.0" in text
    # children are indented one level deeper than the root
    root = next(l for l in lines if "kestrel.request" in l)
    child = next(l for l in lines if "llm.chat" in l)
    assert child.index("llm.chat") == root.index("kestrel.request") + 2


def test_tree_marks_errors():
    assert "ERROR ConnectionError: down" in tr.render_tree(make_trace(status="error"))


# ---- The traffic data ----------------------------------------------------------

EVAL_FILES = ["golden/golden_v1.jsonl", "golden/golden_v2.jsonl", "golden/golden_v3.jsonl",
              "robustness/paraphrases_v1.jsonl", "robustness/paraphrases_v2.jsonl",
              "attacks/injection_v1.jsonl", "attacks/jailbreak_v1.jsonl", "bias/bias_v1.jsonl",
              "agent/agent_v1.jsonl", "agent/calc_probes_v1.jsonl", "agent/calc_probes_v2.jsonl"]


def eval_questions() -> set[str]:
    from app.paths import ROOT
    questions = set()
    for name in EVAL_FILES:
        for line in (ROOT / "data" / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                for key in ("question", "prompt", "attack"):
                    if isinstance(row.get(key), str):
                        questions.add(normalize(row[key]))
                for variant in row.get("variants", []) or []:
                    if isinstance(variant, dict) and isinstance(variant.get("question"), str):
                        questions.add(normalize(variant["question"]))
    return questions


def normalize(q: str) -> str:
    return " ".join(q.lower().split()).rstrip("?.! ")


def test_traffic_v1_is_well_formed():
    rows = traffic.load_traffic(TRAFFIC_DIR / "traffic_v1.jsonl")
    assert len(rows) == 40
    assert [r["id"] for r in rows] == [f"t{i:03d}" for i in range(1, 41)]
    assert {r["channel"] for r in rows} <= {"web", "api", "slack"}
    assert {r["kind"] for r in rows} <= {"in_scope", "multi_step", "out_of_scope", "noisy",
                                          "attack", "follow_up"}
    assert {r["intended"] for r in rows} <= {"answer", "refuse", "either"}
    # sessions are contiguous: one user's requests are sent one after another
    seen, last = set(), None
    for r in rows:
        if r["session"] != last:
            assert r["session"] not in seen, f"session {r['session']} is split up"
            seen.add(r["session"])
            last = r["session"]


def test_traffic_overlaps_no_eval_question():
    """Traffic that is also a test case tells a monitor nothing new."""
    evals = eval_questions()
    assert len(evals) > 100   # the loader really found the eval questions
    clashes = [r["id"] for r in traffic.load_traffic() if normalize(r["question"]) in evals]
    assert clashes == []


def test_load_traffic_rejects_duplicates_and_missing_fields(tmp_path):
    bad = tmp_path / "t.jsonl"
    row = {"id": "t001", "session": "s1", "channel": "web", "question": "q"}
    bad.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        traffic.load_traffic(bad)
    bad.write_text(json.dumps({**row, "question": ""}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="question"):
        traffic.load_traffic(bad)


# ---- The scripts, end to end, offline ------------------------------------------

def test_replay_then_read(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(rag, "generate", fake_generate([]))
    monkeypatch.setattr(traffic, "warmup", lambda model=None: None)
    out = tmp_path / "traces_test.jsonl"
    assert traffic.main(["--limit", "3", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "3 answered, 0 failed" in printed and "Requests: 3" in printed

    loaded = tr.load_traces(out)
    assert [t["attributes"]["traffic.case_id"] for t in loaded] == ["t001", "t002", "t003"]
    assert loaded[0]["attributes"]["session.id"] == "s01"

    assert traces_cli.main([str(out), "--show", "t002"]) == 0
    assert "And how much of that was profit?" in capsys.readouterr().out
    assert traces_cli.main([str(out), "--json"]) == 0
    assert json.loads(capsys.readouterr().out.split("\n", 2)[2])["requests"] == 3


def test_a_failing_request_is_recorded_and_the_replay_continues(monkeypatch, tmp_path):
    good = fake_generate([])
    calls = {"n": 0}

    def flaky(system, user, model=None, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:   # the second request's rewrite call
            raise ConnectionError("API unreachable")
        return good(system, user, model=model, **kwargs)

    monkeypatch.setattr(rag, "generate", flaky)
    exporter = tracing.MemoryExporter()
    rows = traffic.load_traffic()[:3]
    counts = traffic.replay(rows, rag.RagAssistant(), exporter, echo=lambda *_: None)
    assert counts == {"answered": 2, "failed": 1}
    assert [t["status"] for t in exporter.traces] == ["ok", "error", "ok"]
    s = tr.summarize(exporter.traces)
    assert s["errors"] == 1 and "API unreachable" in next(iter(s["error_messages"]))


def test_no_content_replay_stores_no_text(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "generate", fake_generate([]))
    monkeypatch.setattr(traffic, "warmup", lambda model=None: None)
    out = tmp_path / "traces_nc.jsonl"
    traffic.main(["--limit", "2", "--no-content", "--out", str(out)])
    blob = out.read_text(encoding="utf-8")
    assert "kestrel" not in blob.lower().replace("kestrel.", "").replace("kestrel_", "")


def test_traces_cli_with_no_files(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(traces_cli, "TRACES_DIR", tmp_path)
    assert traces_cli.main([]) == 1
    assert "No trace files yet" in capsys.readouterr().out
