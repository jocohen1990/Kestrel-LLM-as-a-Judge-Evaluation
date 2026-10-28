"""Offline tests for the Phase 4 agent (app/agent.py), with a scripted fake model.

The fake model replays a list of replies, one per turn, and records what it was
sent. That lets each test say "the model asks for X, then Y" and check the
agent ran the right tools, sent the results back correctly, and recorded the
trajectory.

Most tests step through the nodes directly (`step_through` below), so they run
without LangGraph installed. The tests at the bottom build the real LangGraph
graph and are skipped if the `langgraph` package isn't installed.
"""
import difflib
import re

import pytest

from app import agent as agent_module
from app.agent import (AGENT_PROMPTS, DEFAULT_AGENT_VERSION, METRIC_NAMES, KestrelAgent,
                       initial_state, not_reported_answer, to_result)
from app.llm import ToolTurn
from app.paths import AGENT_DIR
from app.rag import REFUSAL
from app.tools import METRIC_TABLE, METRICS


def text(t):
    return {"type": "text", "text": t}


def call(name, args, id_):
    return {"type": "tool_use", "id": id_, "name": name, "input": args}


class ScriptedModel:
    """Stands in for generate_with_tools: returns the scripted replies in order."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.sent = []   # (system, messages) per call

    def __call__(self, system, messages, tools, model=None, **kwargs):
        self.sent.append((system, [dict(m) for m in messages]))
        content = self.replies.pop(0)
        return ToolTurn(content=content, model="fake-model", input_tokens=100,
                        output_tokens=10, latency_s=0.5)


def use_model(monkeypatch, *replies) -> ScriptedModel:
    fake = ScriptedModel(*replies)
    # Patch the name where it's USED (app.agent), as in tests/test_rag.py.
    monkeypatch.setattr(agent_module, "generate_with_tools", fake)
    return fake


def step_through(agent: KestrelAgent, question: str):
    """Run the nodes in the order the graph would, without LangGraph:
    model -> (tools -> model)* -> end. Each node returns a partial update."""
    state = initial_state(question)
    while True:
        state.update(agent.call_model(state))
        if agent.route(state) == "end":
            return to_result(state, "fake-model", agent.agent_version)
        state.update(agent.run_tools(state))


# ---- Paths through the graph ----------------------------------------------------

def test_answer_without_tools(monkeypatch):
    use_model(monkeypatch, [text(REFUSAL)])
    result = step_through(KestrelAgent(), "Tell me a joke")
    assert result.answer == REFUSAL
    assert result.steps == [] and result.turns == 1 and result.stopped == "answered"


def test_one_lookup_then_answer(monkeypatch):
    fake = use_model(monkeypatch,
                     [call("get_financial_metric", {"metric": "net_income", "period": "Q2 2026"}, "t1")],
                     [text("Net income was $11.4 million in Q2 2026.")])
    result = step_through(KestrelAgent(), "Q2 2026 net income?")

    assert result.answer == "Net income was $11.4 million in Q2 2026."
    assert result.tools_used == ["get_financial_metric"]
    assert result.steps[0]["args"] == {"metric": "net_income", "period": "Q2 2026"}
    assert "$11.4 million" in result.steps[0]["output"] and not result.steps[0]["is_error"]
    assert result.turns == 2 and result.stopped == "answered"

    # The second model call saw the tool result, tied to the right call id.
    _, messages = fake.sent[1]
    tool_result = messages[-1]["content"][0]
    assert messages[-1]["role"] == "user"
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "t1"
    assert "$11.4 million" in tool_result["content"]


def test_two_lookups_in_one_turn_then_calculator(monkeypatch):
    use_model(monkeypatch,
              [text("Let me look those up."),
               call("get_financial_metric", {"metric": "gross_margin", "period": "Q1 2025"}, "a"),
               call("get_financial_metric", {"metric": "gross_margin", "period": "Q1 2026"}, "b")],
              [call("calculator", {"expression": "71.4 - 69.8"}, "c")],
              [text("Gross margin rose 1.6 percentage points, from 69.8% to 71.4%.")])
    result = step_through(KestrelAgent(), "Margin change Q1 2025 to Q1 2026?")

    assert result.tools_used == ["get_financial_metric", "get_financial_metric", "calculator"]
    assert [s["turn"] for s in result.steps] == [1, 1, 2]   # which model turn asked for it
    assert result.steps[2]["output"] == "71.4 - 69.8 = 1.6"
    assert result.turns == 3
    assert "1.6 percentage points" in result.answer


def test_tool_error_goes_back_to_the_model_and_the_retry_is_recorded(monkeypatch):
    fake = use_model(monkeypatch,
                     [call("get_financial_metric", {"metric": "revenues", "period": "Q2 2026"}, "x")],
                     [call("get_financial_metric", {"metric": "revenue", "period": "Q2 2026"}, "y")],
                     [text("Revenue was $116.0 million.")])
    result = step_through(KestrelAgent(), "Q2 2026 revenue?")

    assert [s["is_error"] for s in result.steps] == [True, False]
    error_result = fake.sent[1][1][-1]["content"][0]
    assert error_result["is_error"] is True and "Use one of" in error_result["content"]


def test_unknown_tool_is_an_error_result_not_a_crash(monkeypatch):
    use_model(monkeypatch, [call("web_search", {"q": "kestrel"}, "w")], [text(REFUSAL)])
    result = step_through(KestrelAgent(), "?")
    assert result.steps[0]["is_error"] and result.stopped == "answered"


def test_agent_that_never_stops_is_cut_off_at_max_turns(monkeypatch):
    replies = [[text("Checking."), call("search_documents", {"query": "revenue"}, f"s{i}")]
               for i in range(10)]
    fake = use_model(monkeypatch, *replies)
    result = step_through(KestrelAgent(max_turns=3), "Revenue?")

    assert result.stopped == "step_limit"
    assert result.turns == 3 and len(fake.sent) == 3
    assert len(result.steps) == 2      # the 3rd turn's tool call is never run
    assert result.answer == "Checking."


def test_usage_is_summed_over_every_turn(monkeypatch):
    use_model(monkeypatch, [call("calculator", {"expression": "1+1"}, "c")], [text("2")])
    result = step_through(KestrelAgent(), "1+1?")
    assert (result.input_tokens, result.output_tokens, result.latency_s) == (200, 20, 1.0)


def test_conversation_alternates_roles_as_the_api_requires(monkeypatch):
    use_model(monkeypatch, [call("calculator", {"expression": "2*3"}, "c")], [text("6")])
    result = step_through(KestrelAgent(), "2*3?")
    assert [m["role"] for m in result.messages] == ["user", "assistant", "user", "assistant"]


# ---- Settings ----------------------------------------------------------------------

def test_system_prompt_is_the_versioned_one(monkeypatch):
    fake = use_model(monkeypatch, [text("ok")])
    step_through(KestrelAgent(agent_version="a1"), "?")
    assert fake.sent[0][0] == AGENT_PROMPTS["a1"]
    assert REFUSAL in AGENT_PROMPTS["a1"]


@pytest.mark.parametrize("version", sorted(AGENT_PROMPTS))
def test_every_version_is_sent_as_written(monkeypatch, version):
    fake = use_model(monkeypatch, [text("ok")])
    step_through(KestrelAgent(agent_version=version), "?")
    assert fake.sent[0][0] == AGENT_PROMPTS[version]
    assert REFUSAL in AGENT_PROMPTS[version]


def test_default_stays_a1_until_a2_has_a_graded_run():
    # Like the RAG prompts: a new version becomes the default only after a run
    # shows it's better. Change this test in the same commit as the default.
    assert DEFAULT_AGENT_VERSION == "a1"


# ---- Prompt a2 (after Run 14) ------------------------------------------------------

def _changed_lines(old, new):
    return [l for l in difflib.ndiff(old.splitlines(), new.splitlines()) if l[:2] in ("- ", "+ ")]


def test_a2_changes_only_the_calculator_and_not_reported_rules():
    # One run can only say what a change did if nothing else changed with it.
    removed = [l[2:] for l in _changed_lines(AGENT_PROMPTS["a1"], AGENT_PROMPTS["a2"]) if l[0] == "-"]
    added = [l[2:] for l in _changed_lines(AGENT_PROMPTS["a1"], AGENT_PROMPTS["a2"]) if l[0] == "+"]
    assert len(removed) == 2 and len(added) == 3
    assert removed[0].startswith("- calculator:") and added[0].startswith("- calculator:")
    assert removed[1].startswith("- If your tools don't give you the answer")
    assert added[1].startswith("- If the metric tool says a figure isn't reported")
    # The general refusal rule is unchanged apart from "Otherwise, " in front.
    assert added[2] == "- Otherwise, i" + removed[1][3:]


def test_a2_requires_the_calculator_for_whole_numbers():
    calc_line = next(l for l in AGENT_PROMPTS["a2"].splitlines() if l.startswith("- calculator:"))
    assert "whole numbers" in calc_line and "Never do arithmetic in your head" in calc_line


def test_a2_states_the_not_reported_template_and_a_worked_example():
    prompt = AGENT_PROMPTS["a2"]
    assert '"Kestrel did not report <metric> for <period>."' in prompt
    assert f'"{not_reported_answer("diluted_eps", "FY2024")}"' in prompt
    # Every reader-facing metric name is the one the prompt's tool list uses.
    tool_line = next(l for l in prompt.splitlines() if l.startswith("- get_financial_metric:"))
    for name in METRIC_NAMES.values():
        assert name in tool_line


def _agent_cases():
    import json
    lines = (AGENT_DIR / "agent_v1.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def test_a2_examples_arent_taken_from_the_eval_cases():
    # A prompt that quotes the test's own numbers would pass the test without
    # learning the rule. The not-reported example must be a real gap that no
    # case asks about; the calculator example's numbers must not appear in any
    # case.
    assert ("diluted_eps", "FY2024") not in METRIC_TABLE
    asked = {(s["args"]["metric"], s["args"]["period"])
             for c in _agent_cases() for s in c["required_steps"] if s.get("tool") == "get_financial_metric"}
    assert ("diluted_eps", "FY2024") not in asked

    example_numbers = {60.0, 25.0, 35.0}   # "60 - 25", and its result
    case_numbers = set()
    for c in _agent_cases():
        case_numbers |= {float(n) for n in c["expected_numbers"]}
        case_numbers |= {float(n) for s in c["required_steps"] for n in s.get("result", [])}
        case_numbers |= {float(n) for n in re.findall(r"\d+(?:\.\d+)?", c["question"])}
    assert not example_numbers & case_numbers


def test_metric_names_cover_every_metric():
    assert set(METRIC_NAMES) == set(METRICS)


@pytest.mark.parametrize("metric,period,sentence", [
    ("net_income", "FY2024", "Kestrel did not report net income for FY2024."),
    ("diluted_eps", "FY2024", "Kestrel did not report diluted EPS for FY2024."),
    ("enterprise_customers", "Q1 2026", "Kestrel did not report enterprise customers for Q1 2026."),
])
def test_not_reported_answer(metric, period, sentence):
    assert not_reported_answer(metric, period) == sentence


def test_unknown_version_and_bad_max_turns_are_rejected():
    with pytest.raises(ValueError):
        KestrelAgent(agent_version="a99")
    with pytest.raises(ValueError):
        KestrelAgent(max_turns=0)


# ---- The real LangGraph graph (skipped if langgraph isn't installed) ----------------

def test_graph_gives_the_same_result_as_stepping_through(monkeypatch):
    pytest.importorskip("langgraph")
    script = ([call("get_financial_metric", {"metric": "revenue", "period": "Q1 2026"}, "a"),
               call("get_financial_metric", {"metric": "revenue", "period": "Q2 2026"}, "b")],
              [call("calculator", {"expression": "108.3 + 116.0"}, "c")],
              [text("Combined revenue was $224.3 million.")])
    use_model(monkeypatch, *script)
    stepped = step_through(KestrelAgent(), "H1 2026 revenue?")
    use_model(monkeypatch, *script)
    graphed = KestrelAgent().run("H1 2026 revenue?")

    assert graphed.answer == stepped.answer
    assert graphed.steps == stepped.steps
    assert graphed.turns == stepped.turns == 3
    assert graphed.messages == stepped.messages
    assert graphed.stopped == "answered" and graphed.model == "fake-model"


def test_graph_respects_max_turns(monkeypatch):
    pytest.importorskip("langgraph")
    use_model(monkeypatch, *[[call("search_documents", {"query": "x"}, f"s{i}")] for i in range(20)])
    result = KestrelAgent(max_turns=4).run("?")
    assert result.stopped == "step_limit" and result.turns == 4
