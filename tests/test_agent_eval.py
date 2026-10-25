"""Offline tests for the agent eval (evals/agent_eval.py) and its dataset.

Three kinds of test:
  1. The dataset is checked against the REAL tools and corpus: every metric
     step's figure exists (or is really "not reported"), every evidence string
     is really in a filing, every calculator target is what the arithmetic
     gives. A typo in the dataset fails here, not in an expensive API run.
  2. A perfect trajectory for every case, built with the real tools, gets full
     marks. If it didn't, the grader (or the case) would be wrong.
  3. Each failure the eval is meant to catch (arithmetic in its head, the wrong
     period, calculating before looking up, a forbidden tool...) is caught.
"""
import json
from types import SimpleNamespace

import pytest

from app.agent import KestrelAgent, initial_state, to_result
from app.llm import ToolTurn
from app.paths import AGENT_DIR, CORPUS_DIR
from app.rag import REFUSAL
from app.tools import METRIC_TABLE, run_tool
from evals import agent_eval
from evals.agent_eval import (AgentDatasetError, DECLINE_CATEGORIES, agent_mentions_internals,
                              calculator_after_inputs, calculator_value, declined_agent,
                              grade_agent, load_agent_cases, match_steps, repeated_calls,
                              run_case, summarize_agent, unsupported_numbers)

CASES = load_agent_cases(AGENT_DIR / "agent_v1.jsonl")
BY_ID = {c["id"]: c for c in CASES}
CORPUS = "\n".join(p.read_text(encoding="utf-8") for p in CORPUS_DIR.glob("*.md"))

# The arithmetic behind every calculator step, written out by hand. The test
# runs it through the real calculator and checks the dataset's target matches.
EXPRESSIONS = {
    "ag011": "108.3 + 116.0",
    "ag012": "11.4 - 9.2",
    "ag013": "11.4 / 116.0 * 100",
    "ag014": "3120 - 2850",
    "ag015": "72.1 - 71.4",
    "ag016": "(11.4 - 9.2) / 9.2 * 100",
    "ag017": "150 - 40",
    "ag018": "64 / 116.0 * 100",
}


def step(tool, args, output=None, is_error=False, turn=1):
    if output is None:
        result = run_tool(tool, args)
        output, is_error = result.output, result.is_error
    return {"turn": turn, "tool": tool, "args": args, "output": output, "is_error": is_error}


def metric(m, p):
    return step("get_financial_metric", {"metric": m, "period": p})


def search(query):
    return step("search_documents", {"query": query})


def calc(expression):
    return step("calculator", {"expression": expression})


def result(steps, answer, stopped="answered", question="q"):
    return SimpleNamespace(question=question, answer=answer, steps=steps, turns=len(steps) + 1,
                           stopped=stopped, input_tokens=100, output_tokens=10, latency_s=1.0,
                           tools_used=[s["tool"] for s in steps])


def perfect_steps(case):
    """The ideal trajectory for a case, run through the real tools."""
    steps = []
    for req in case["required_steps"]:
        req = req["any_of"][0] if "any_of" in req else req
        if req["tool"] == "get_financial_metric":
            steps.append(step("get_financial_metric", req["args"]))
        elif req["tool"] == "search_documents":
            steps.append(search(req["evidence"]))
        else:
            steps.append(calc(EXPRESSIONS[case["id"]]))
    return steps


def perfect_answer(case):
    if case["category"] in DECLINE_CATEGORIES:
        return REFUSAL
    parts = [f"{n:g}" for n in case["expected_numbers"]] + case["must_include"]
    parts += case["must_include_any"][:1]
    return "Kestrel: " + ", ".join(parts) + "."


# ---- 1. The dataset against the real tools ---------------------------------------

def test_dataset_loads_with_every_category():
    assert len(CASES) == 27
    assert {c["category"] for c in CASES} == agent_eval.CATEGORIES


def all_required(case):
    for req in case["required_steps"]:
        yield from (req["any_of"] if "any_of" in req else [req])


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_every_metric_step_is_real(case):
    for req in all_required(case):
        if req["tool"] != "get_financial_metric":
            continue
        key = (req["args"]["metric"], req["args"]["period"])
        output = run_tool("get_financial_metric", req["args"]).output
        if case["category"] == "not_reported":
            assert key not in METRIC_TABLE and output.startswith("Not reported")
        else:
            assert key in METRIC_TABLE


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_every_evidence_string_is_in_a_filing(case):
    for req in all_required(case):
        if req["tool"] == "search_documents":
            assert req["evidence"] in CORPUS


@pytest.mark.parametrize("case_id", sorted(EXPRESSIONS))
def test_every_calculator_target_is_the_real_arithmetic(case_id):
    targets = [r["result"] for r in BY_ID[case_id]["required_steps"]
               if r.get("tool") == "calculator"]
    assert len(targets) == 1
    value = calculator_value(calc(EXPRESSIONS[case_id])["output"])
    assert any(abs(value - t) <= abs(t) * 1e-4 for t in targets[0])


def test_every_case_with_a_calculator_step_has_an_expression():
    with_calc = {c["id"] for c in CASES
                 if any(r.get("tool") == "calculator" for r in c["required_steps"])}
    assert with_calc == set(EXPRESSIONS)


# ---- 2. A perfect run gets full marks ---------------------------------------------

@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_perfect_trajectory_is_fully_correct(case):
    grade = grade_agent(case, result(perfect_steps(case), perfect_answer(case),
                                     question=case["question"]))
    assert grade["fully_correct"], grade
    assert grade["unsupported_numbers"] == []
    assert grade["mentions_internals"] is False
    if case["category"] in DECLINE_CATEGORIES:
        assert grade["refusal_exact"] is True


# ---- 3. The failures it's meant to catch ------------------------------------------

def test_arithmetic_in_its_head_is_right_answer_wrong_path():
    case = BY_ID["ag011"]
    steps = [metric("revenue", "Q1 2026"), metric("revenue", "Q2 2026")]
    grade = grade_agent(case, result(steps, "Combined revenue was $224.3 million."))
    assert grade["answer_ok"] and not grade["tool_choice_ok"] and not grade["args_ok"]
    assert grade["missing_tools"] == ["calculator"]
    summary = summarize_agent([{**grade, "id": "ag011", "category": "calculation",
                                "num_steps": 2, "turns": 3, "stopped": "answered",
                                "input_tokens": 1, "output_tokens": 1, "latency_s": 1}])
    assert summary["right_answer_wrong_path_ids"] == ["ag011"]


def test_wrong_period_is_an_argument_error_not_a_tool_error():
    grade = grade_agent(BY_ID["ag003"], result([metric("gross_margin", "Q1 2026")],
                                               "Gross margin was 71.4%."))
    assert grade["tool_choice_ok"] and not grade["args_ok"] and not grade["answer_ok"]
    assert grade["unmatched_steps"] == [{"tool": "get_financial_metric",
                                         "args": {"metric": "gross_margin", "period": "Q2 2026"}}]


def test_calculating_before_looking_up_is_out_of_order():
    case = BY_ID["ag012"]
    steps = [calc("11.4 - 9.2"), metric("net_income", "Q1 2026"), metric("net_income", "Q2 2026")]
    grade = grade_agent(case, result(steps, "Net income rose by $2.2 million."))
    assert grade["args_ok"] and grade["order_ok"] is False and not grade["trajectory_ok"]


def test_calculator_graded_on_result_not_expression():
    case = BY_ID["ag013"]
    for expression in ("11.4 / 116.0 * 100", "11.4/116", "(11.4) / (116)"):
        steps = [metric("net_income", "Q2 2026"), metric("revenue", "Q2 2026"), calc(expression)]
        assert grade_agent(case, result(steps, "Net margin was 9.8%."))["fully_correct"], expression


def test_a_wrong_calculation_fails_arguments():
    case = BY_ID["ag016"]
    steps = [metric("net_income", "Q1 2026"), metric("net_income", "Q2 2026"),
             calc("(11.4 - 9.2) / 11.4 * 100")]   # divided by the wrong base
    grade = grade_agent(case, result(steps, "Net income grew 19.3%."))
    assert not grade["args_ok"] and not grade["answer_ok"]


def test_error_then_retry_is_fine_within_budget():
    case = BY_ID["ag001"]
    steps = [step("get_financial_metric", {"metric": "revenue", "period": "2024"}),
             metric("revenue", "FY2024")]
    grade = grade_agent(case, result(steps, "Revenue was $349.7 million in FY2024."))
    assert steps[0]["is_error"] and grade["tool_errors"] == 1
    assert grade["fully_correct"]


def test_over_budget_fails_trajectory():
    case = BY_ID["ag001"]
    steps = [search("revenue 2024"), search("fiscal 2024 revenue"), metric("revenue", "FY2024")]
    grade = grade_agent(case, result(steps, "Revenue was $349.7 million."))
    assert not grade["within_budget"] and not grade["trajectory_ok"]
    assert grade["first_tool_ok"] is False and not grade["tool_choice_ok"]


def test_repeated_identical_call_fails_trajectory():
    case = BY_ID["ag002"]
    steps = [metric("enterprise_customers", "Q2 2026"), metric("enterprise_customers", "Q2 2026")]
    grade = grade_agent(case, result(steps, "Kestrel had 3,120 enterprise customers."))
    assert grade["repeated_calls"] == 1 and not grade["trajectory_ok"]


def test_step_limit_fails_trajectory():
    case = BY_ID["ag006"]
    steps = [search("Marcus Oyelaran serves as Chief Financial Officer")]
    grade = grade_agent(case, result(steps, "Marcus Oyelaran", stopped="step_limit"))
    assert not grade["trajectory_ok"] and not grade["fully_correct"]


def test_forbidden_tool_on_a_tool_misuse_request():
    case = BY_ID["ag026"]
    grade = grade_agent(case, result([calc("17 * 23")], "17 times 23 is 391."))
    assert grade["forbidden_tools_used"] == ["calculator"]
    assert not grade["tool_choice_ok"] and not grade["answer_ok"]


def test_not_reported_in_its_own_words_is_right_but_not_exact():
    # The smoke test's actual answer to a not-reported question.
    case = BY_ID["ag019"]
    grade = grade_agent(case, result([metric("net_income", "FY2024")],
                                     "Kestrel did not report net income for fiscal year 2024."))
    assert grade["answer_ok"] and grade["fully_correct"] and grade["refusal_exact"] is False


def test_estimating_a_not_reported_figure_fails():
    case = BY_ID["ag019"]
    grade = grade_agent(case, result([metric("net_income", "FY2024")],
                                     "Net income for FY2024 was about $30 million."))
    assert not grade["answer_ok"] and grade["unsupported_numbers"] == ["30"]


def test_search_for_a_metric_counts_when_alternatives_are_allowed():
    case = BY_ID["ag024"]
    steps = [search("Q2 2026 revenue")]
    grade = grade_agent(case, result(steps, "Revenue didn't fall: it rose 19% to $116.0 million."))
    assert grade["fully_correct"], grade


def test_alternative_shows_both_tools_when_missing():
    grade = grade_agent(BY_ID["ag024"], result([], "Revenue fell because of fuel prices."))
    assert grade["missing_tools"] == ["get_financial_metric/search_documents"]


# ---- Helpers ----------------------------------------------------------------------

def test_calculator_value():
    assert calculator_value("71.4 - 69.8 = 1.6") == 1.6
    assert calculator_value("Division by zero.") is None
    assert calculator_value("a = b = 2") == 2


def test_match_steps_shares_one_search_between_requirements():
    req = [{"tool": "search_documents", "evidence": "$150 million revolving credit facility"},
           {"tool": "search_documents", "evidence": "$40 million was drawn"}]
    assert match_steps(req, [search("credit facility drawn")]) == [0, 0]


def test_order_check_is_none_without_both_sides():
    assert calculator_after_inputs([{"tool": "calculator", "result": [1]}], [0]) is None
    assert calculator_after_inputs([{"tool": "search_documents", "evidence": "x"},
                                    {"tool": "calculator", "result": [1]}], [0, None]) is None


def test_repeated_calls_ignores_argument_order():
    a = step("get_financial_metric", {"metric": "revenue", "period": "FY2025"})
    b = step("get_financial_metric", {"period": "FY2025", "metric": "revenue"})
    assert repeated_calls([a, b]) == 1


@pytest.mark.parametrize("answer", [
    REFUSAL, "Kestrel didn't report that figure.", "That figure isn't disclosed in the filings.",
    "The documents do not provide FY2024 net income.", "I can't help with that.",
])
def test_declined_agent(answer):
    assert declined_agent(answer)


def test_declined_agent_doesnt_catch_answers():
    assert not declined_agent("Net income was $38.9 million in fiscal 2025.")


@pytest.mark.parametrize("answer,expected", [
    ("Revenue was $412,600,000.", []),           # written out in full
    ("Net margin was 9.83%.", []),                # ratio from the calculator, as a percent
    ("Net income grew about 24%.", []),           # rounded
    ("Revenue was $412.6 million, up 25%.", ["25"]),
])
def test_unsupported_numbers(answer, expected):
    steps = [metric("revenue", "FY2025"), calc("11.4 / 116.0"), calc("(11.4 - 9.2) / 9.2 * 100")]
    assert unsupported_numbers(answer, "What happened in FY2025?", steps) == expected


def test_numbers_from_errored_calls_dont_count_as_support():
    bad = step("calculator", {"expression": "1/0"}, output="Division by zero: 999", is_error=True)
    assert unsupported_numbers("It was 999.", "q", [bad]) == ["999"]


@pytest.mark.parametrize("answer,flagged", [
    ("I looked up the figure: $11.4 million.", True),
    ("According to my search, the CFO is Marcus Oyelaran.", True),
    ("Using the calculator, the margin is 9.8%.", True),
    ("According to the Q2 2026 earnings release, net income was $11.4 million.", False),
])
def test_agent_mentions_internals(answer, flagged):
    assert agent_mentions_internals(answer) is flagged


# ---- Dataset validation -----------------------------------------------------------

GOOD = {"id": "x1", "category": "lookup", "question": "q", "max_steps": 2,
        "required_steps": [{"tool": "get_financial_metric",
                            "args": {"metric": "revenue", "period": "FY2025"}}],
        "expected_numbers": [412.6]}


def write_cases(tmp_path, *cases):
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in cases), encoding="utf-8")
    return path


def test_good_case_loads(tmp_path):
    case = load_agent_cases(write_cases(tmp_path, GOOD))[0]
    assert case["forbidden_tools"] == [] and case["first_tool"] is None


@pytest.mark.parametrize("change,message", [
    ({"category": "nope"}, "category"),
    ({"required_steps": [{"tool": "web_search"}]}, "unknown tool"),
    ({"required_steps": [{"tool": "get_financial_metric",
                          "args": {"metric": "ebitda", "period": "FY2025"}}]}, "valid metric"),
    ({"required_steps": [{"tool": "search_documents"}]}, "evidence"),
    ({"required_steps": [{"tool": "calculator", "result": []}]}, "result"),
    ({"required_steps": [{"any_of": [{"any_of": []}]}]}, "nested"),
    ({"max_steps": 0}, "max_steps"),
    ({"forbidden_tools": ["shell"]}, "unknown tool"),
    ({"category": "out_of_scope"}, "no answer"),
    ({"expected_numbers": []}, "check the answer"),
])
def test_bad_cases_fail_loudly(tmp_path, change, message):
    with pytest.raises(AgentDatasetError, match=message):
        load_agent_cases(write_cases(tmp_path, {**GOOD, **change}))


def test_duplicate_ids_fail(tmp_path):
    with pytest.raises(AgentDatasetError, match="duplicate"):
        load_agent_cases(write_cases(tmp_path, GOOD, GOOD))


# ---- Summary and the full path through run_case -----------------------------------

class ScriptedModel:
    def __init__(self, *replies):
        self.replies = list(replies)

    def __call__(self, system, messages, tools, model=None, **kwargs):
        return ToolTurn(content=self.replies.pop(0), model="fake-model", input_tokens=100,
                        output_tokens=10, latency_s=0.5)


class SteppingAgent(KestrelAgent):
    """Runs the nodes in graph order without LangGraph, like tests/test_agent.py."""

    def run(self, question):
        state = initial_state(question)
        while True:
            state.update(self.call_model(state))
            if self.route(state) == "end":
                return to_result(state, "fake-model", self.agent_version)
            state.update(self.run_tools(state))


def tool_use(name, args, id_):
    return {"type": "tool_use", "id": id_, "name": name, "input": args}


def text(t):
    return {"type": "text", "text": t}


GOOD_RUN = [
    [tool_use("get_financial_metric", {"metric": "net_income", "period": "Q1 2026"}, "a"),
     tool_use("get_financial_metric", {"metric": "net_income", "period": "Q2 2026"}, "b")],
    [tool_use("calculator", {"expression": "11.4 - 9.2"}, "c")],
    [text("Net income rose by $2.2 million, from $9.2 million in Q1 2026 to $11.4 million in Q2 2026.")],
]


def test_run_case_end_to_end(monkeypatch):
    monkeypatch.setattr("app.agent.generate_with_tools", ScriptedModel(*GOOD_RUN))
    row = run_case(BY_ID["ag012"], SteppingAgent())
    assert row["fully_correct"] and row["num_steps"] == 3 and row["turns"] == 3
    assert row["tools_used"] == ["get_financial_metric", "get_financial_metric", "calculator"]
    assert row["input_tokens"] == 300
    json.dumps(row)   # results are saved as JSON


def test_summary_rates_and_categories(monkeypatch):
    monkeypatch.setattr("app.agent.generate_with_tools", ScriptedModel(
        *GOOD_RUN,
        # ag019: declines in its own words
        [tool_use("get_financial_metric", {"metric": "net_income", "period": "FY2024"}, "d")],
        [text("Kestrel didn't report net income for FY2024.")],
        # ag011: does the sum in its head
        [tool_use("get_financial_metric", {"metric": "revenue", "period": "Q1 2026"}, "e"),
         tool_use("get_financial_metric", {"metric": "revenue", "period": "Q2 2026"}, "f")],
        [text("Combined revenue was $224.3 million.")],
    ))
    agent = SteppingAgent()
    rows = [run_case(BY_ID[i], agent) for i in ("ag012", "ag019", "ag011")]
    s = summarize_agent(rows)
    assert s["cases"] == 3
    assert s["fully_correct_rate"] == pytest.approx(0.667, abs=0.001)
    assert s["answer_accuracy"] == 1.0
    assert s["right_answer_wrong_path_ids"] == ["ag011"]
    assert s["inexact_refusal_ids"] == ["ag019"] and s["exact_refusal_rate"] == 0.0
    assert s["by_category"]["calculation"] == {"cases": 2, "fully_correct": 1, "answer_ok": 2}
    assert s["avg_steps"] == pytest.approx(2.0)


def test_run_case_through_the_real_graph(monkeypatch):
    pytest.importorskip("langgraph")
    monkeypatch.setattr("app.agent.generate_with_tools", ScriptedModel(*GOOD_RUN))
    row = run_case(BY_ID["ag012"], KestrelAgent())
    assert row["fully_correct"] and row["stopped"] == "answered"
