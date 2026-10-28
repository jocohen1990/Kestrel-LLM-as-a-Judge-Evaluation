"""Agent evals (Phase 4): grade the steps, not just the answer.

The RAG evals ask one question: is the answer right? An agent can get the right
answer the wrong way (it did the arithmetic in its head instead of using the
calculator, or it guessed a figure it never looked up), and a wrong answer can
come from one bad step (it looked up the wrong period). So each run is graded on
four things, from the trajectory the agent records (app/agent.py):

  tool_choice  Did it use the right TOOLS? Every tool a case needs was called,
               no forbidden tool was called, and when a case names a
               `first_tool`, it reached for that one first.
  arguments    Did it call them with the right ARGUMENTS? Each required step
               says what a correct call looks like (see "Required steps").
  trajectory   Was the PATH sound? It finished on its own (no step limit),
               stayed within the case's step budget, ran the calculator only
               after it had the numbers to calculate with, and never repeated
               an identical call.
  answer       Is the final ANSWER right? The golden-set rule grader for
               questions with an answer; for questions with no answer
               (not_reported, out_of_scope), it must decline.

  fully_correct = all four. That's the headline number. The gap between it and
  answer accuracy is the point of Phase 4: answers that are right by luck.

Required steps (one dict per step in the dataset, data/agent/agent_v1.jsonl):

  {"tool": "get_financial_metric", "args": {"metric": "revenue", "period": "Q2 2026"}}
      A call with exactly these arguments, that didn't error.
  {"tool": "search_documents", "evidence": "$40 million was drawn"}
      A search whose RESULT contains this text. Search queries are free text and
      many are fine, so the query isn't graded; whether it found the paragraph is.
  {"tool": "calculator", "result": [9.827586, 0.09827586]}
      A calculation whose result is one of these values (to 0.01%). Graded on
      the result, not the expression, so "11.4/116*100" and "11.4/116" both pass.
  {"any_of": [step, step]}
      Either step is fine (Q2 revenue from the metric tool OR from a search).

Two flags are reported but don't count toward fully_correct, because each is a
heuristic that can misfire (read the flagged answers before reporting them):

  unsupported_numbers  Numbers in the answer that no tool returned and the
                       question didn't contain, allowing for rounding, millions
                       written out, and ratio <-> percent. A cheap, rule-based
                       hallucination check.
  mentions_internals   The answer mentions tools, searches or lookups, which
                       prompt a1 forbids (the user can't see them).

And one strict format check, like the RAG evals' exact-refusal rate:

  refusal_exact  For questions with no answer: declined with exactly the
                 required sentence, and nothing else. Out of scope: the stock
                 "I don't know based on the provided documents." Not reported
                 (spec decided after Run 14): "Kestrel did not report <metric>
                 for <period>.", built from the case's metric step with
                 app.agent.not_reported_answer, so the agent's prompt and the
                 grader can't drift apart. Prompt a1 never asked for that
                 sentence, so a1 isn't expected to pass it; a2 is.

Everything here is a pure function of (case, AgentResult): no API calls.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

from app.agent import not_reported_answer
from app.rag import REFUSAL
from app.tools import METRICS, PERIODS, TOOLS
from evals.injection import declined, task_check
from evals.metrics import NUMBER_RE, mentions_internals, refusal_is_exact

CATEGORIES = {
    "lookup",         # one figure, metric tool
    "search",         # one fact, search tool (including figures the metric tool doesn't have)
    "calculation",    # two lookups, then the calculator
    "mixed",          # search + calculator (and sometimes a lookup)
    "not_reported",   # the metric tool says "not reported": decline, don't estimate
    "guidance",       # forecasts, which only search can find
    "false_premise",  # the question assumes something the filings contradict
    "out_of_scope",   # nothing to find, or a request to misuse a tool
}
# Categories whose correct answer is to decline.
DECLINE_CATEGORIES = {"not_reported", "out_of_scope"}
REQUIRED_FIELDS = {"id", "category", "question", "required_steps", "max_steps"}

# How an agent says a figure isn't in the filings. evals/injection.py's
# `declined` covers "I don't know" and "not available"; these are the
# not_reported wordings it misses ("Kestrel did not report diluted EPS for
# FY2025", from the Phase 4 smoke test).
NOT_REPORTED_PHRASES = (
    "not report", "n't report", "wasn't reported", "isn't reported", "not reported",
    "not disclose", "n't disclose", "not disclosed", "not given", "not provide", "n't provide",
    "no figure", "not include", "n't include",
)

# On top of the RAG pipeline's internals (metrics.mentions_internals): an
# agent's answer shouldn't mention its own machinery either.
AGENT_INTERNALS_PATTERN = re.compile(
    r"\b(tools?|calculator|look(?:ed)?[ -]?ups?|looked up|my search(?:es)?|the search(?:es)?"
    r"|search results?|i searched|function calls?)\b",
    re.IGNORECASE,
)


class AgentDatasetError(ValueError):
    pass


# ---- Dataset ------------------------------------------------------------------

def _check_step(step: dict, fail) -> None:
    if "any_of" in step:
        if not step["any_of"] or any("any_of" in s for s in step["any_of"]):
            fail("any_of needs at least one step, and can't be nested")
        for alternative in step["any_of"]:
            _check_step(alternative, fail)
        return
    tool = step.get("tool")
    if tool not in TOOLS:
        fail(f"unknown tool {tool!r}")
    if tool == "get_financial_metric":
        args = step.get("args", {})
        if args.get("metric") not in METRICS or args.get("period") not in PERIODS:
            fail(f"metric step needs a valid metric and period, got {args}")
    elif tool == "search_documents":
        if not step.get("evidence"):
            fail("a search step needs the `evidence` text its result must contain")
    elif tool == "calculator":
        if not step.get("result") or not all(isinstance(v, (int, float)) for v in step["result"]):
            fail("a calculator step needs a list of acceptable `result` values")


def load_agent_cases(path: Path) -> list[dict]:
    """Read the agent .jsonl file and fail loudly on anything malformed."""
    cases, seen = [], set()
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        case = json.loads(line)
        for key in ("expected_numbers", "must_include", "must_include_any", "forbidden_tools"):
            case.setdefault(key, [])
        case.setdefault("first_tool", None)

        def fail(msg):
            raise AgentDatasetError(f"Line {line_no} ({case.get('id', '?')}): {msg}")

        missing = REQUIRED_FIELDS - case.keys()
        if missing:
            fail(f"missing fields {sorted(missing)}")
        if case["id"] in seen:
            fail("duplicate id")
        if case["category"] not in CATEGORIES:
            fail(f"category must be one of {sorted(CATEGORIES)}")
        for step in case["required_steps"]:
            _check_step(step, fail)
        for tool in case["forbidden_tools"] + ([case["first_tool"]] if case["first_tool"] else []):
            if tool not in TOOLS:
                fail(f"unknown tool {tool!r}")
        if not isinstance(case["max_steps"], int) or case["max_steps"] < 0:
            fail("max_steps must be a whole number, 0 or more")
        if len(case["required_steps"]) > case["max_steps"]:
            fail("max_steps is smaller than the number of required steps")
        has_answer = bool(case["expected_numbers"] or case["must_include"] or case["must_include_any"])
        if case["category"] in DECLINE_CATEGORIES and has_answer:
            fail("a question with no answer can't have expected answer values")
        if case["category"] not in DECLINE_CATEGORIES and not has_answer:
            fail("an answerable question needs something to check the answer against")
        if case["category"] == "not_reported" and not _metric_step(case):
            fail("a not_reported case needs a get_financial_metric step: "
                 "its exact refusal names that metric and period")
        seen.add(case["id"])
        cases.append(case)
    return cases


# ---- Matching steps -------------------------------------------------------------

def calculator_value(output: str) -> float | None:
    """The number after the last ' = ' in a calculator result ("71.4 - 69.8 = 1.6")."""
    if " = " not in output:
        return None
    try:
        return float(output.rsplit(" = ", 1)[1])
    except ValueError:
        return None


def step_matches(required: dict, step: dict) -> bool:
    """Does one recorded step satisfy one required step? Errored calls never do."""
    if "any_of" in required:
        return any(step_matches(alt, step) for alt in required["any_of"])
    if step["tool"] != required["tool"] or step["is_error"]:
        return False
    if required["tool"] == "get_financial_metric":
        return step["args"] == required["args"]
    if required["tool"] == "search_documents":
        return required["evidence"] in step["output"]
    value = calculator_value(step["output"])
    return value is not None and any(
        abs(value - target) <= max(1e-9, abs(target) * 1e-4) for target in required["result"])


def required_tools(required: dict) -> set[str]:
    if "any_of" in required:
        return {t for alt in required["any_of"] for t in required_tools(alt)}
    return {required["tool"]}


def match_steps(required_steps: list[dict], steps: list[dict]) -> list[int | None]:
    """For each required step, the index of the first recorded step that
    satisfies it, or None. Steps may be shared (a search can provide two facts)."""
    return [next((i for i, s in enumerate(steps) if step_matches(req, s)), None)
            for req in required_steps]


def is_calculator_step(required: dict) -> bool:
    return "any_of" not in required and required["tool"] == "calculator"


def calculator_after_inputs(required_steps: list[dict], matched: list[int | None]) -> bool | None:
    """Did the agent calculate only after it had the inputs? A calculator call
    before the lookups means the numbers it used came from somewhere else (its
    own memory, or a guess). None when there's nothing to compare."""
    inputs = [m for r, m in zip(required_steps, matched) if not is_calculator_step(r) and m is not None]
    calcs = [m for r, m in zip(required_steps, matched) if is_calculator_step(r) and m is not None]
    if not inputs or not calcs:
        return None
    return min(calcs) > max(inputs)


def repeated_calls(steps: list[dict]) -> int:
    """How many calls repeat an earlier call exactly (same tool, same arguments)."""
    seen, repeats = set(), 0
    for s in steps:
        key = (s["tool"], json.dumps(s["args"], sort_keys=True))
        repeats += key in seen
        seen.add(key)
    return repeats


# ---- Answer checks --------------------------------------------------------------

def _metric_step(case: dict) -> dict | None:
    return next((r for r in case["required_steps"]
                 if "any_of" not in r and r["tool"] == "get_financial_metric"), None)


def expected_refusal(case: dict) -> str | None:
    """The exact sentence a no-answer case must decline with, or None for a
    question that has an answer."""
    if case["category"] == "not_reported":
        args = _metric_step(case)["args"]
        return not_reported_answer(args["metric"], args["period"])
    if case["category"] in DECLINE_CATEGORIES:
        return REFUSAL
    return None


def declined_agent(answer: str) -> bool:
    text = " ".join((answer or "").replace("’", "'").lower().split())
    return declined(answer) or any(p in text for p in NOT_REPORTED_PHRASES)


def agent_mentions_internals(answer: str) -> bool:
    return mentions_internals(answer or "") or bool(AGENT_INTERNALS_PATTERN.search(answer or ""))


def _decimals(text: str) -> int:
    return len(text.split(".", 1)[1]) if "." in text else 0


def unsupported_numbers(answer: str, question: str, steps: list[dict]) -> list[str]:
    """Numbers in the answer that don't come from the question or a tool result.

    An answer number is supported if some source number rounds to it at the
    precision it's written with ("19%" from 18.97, "9.83%" from 9.827586),
    allowing for figures written out ("$412,600,000" for 412.6 million) and for
    a ratio given as a percent (0.0983 -> 9.83%).
    """
    sources = [question] + [s["output"] for s in steps if not s["is_error"]]
    source_numbers = []
    for text in sources:
        for raw in NUMBER_RE.findall(text):
            value = float(raw.replace(",", ""))
            source_numbers += [value, value * 100, value / 100]
    unsupported = []
    for raw in NUMBER_RE.findall(answer or ""):
        n = float(raw.replace(",", ""))
        tolerance = 0.5 * 10 ** -_decimals(raw) + 1e-9
        supported = any(abs(src - n / scale) <= tolerance / scale
                        for scale in (1, 1e3, 1e6, 1e9) for src in source_numbers)
        if not supported:
            unsupported.append(raw)
    return unsupported


# ---- Grading --------------------------------------------------------------------

def grade_agent(case: dict, result) -> dict:
    """Grade one agent run (an app.agent.AgentResult). Pure function: no API calls."""
    steps = result.steps
    used = [s["tool"] for s in steps]
    required = case["required_steps"]
    matched = match_steps(required, steps)

    # Tool choice: the right tools, nothing forbidden, the right one first.
    # A step with alternatives is shown as "get_financial_metric/search_documents".
    missing_tools = sorted({"/".join(sorted(required_tools(r))) for r in required
                            if not required_tools(r) & set(used)})
    forbidden_used = sorted(set(case["forbidden_tools"]) & set(used))
    first_tool_ok = None if not case["first_tool"] else bool(used) and used[0] == case["first_tool"]
    tool_choice_ok = not missing_tools and not forbidden_used and first_tool_ok is not False

    # Arguments: every required step was satisfied by some call.
    args_ok = all(m is not None for m in matched)
    unmatched = [r for r, m in zip(required, matched) if m is None]

    # Trajectory.
    order_ok = calculator_after_inputs(required, matched)
    repeats = repeated_calls(steps)
    within_budget = len(steps) <= case["max_steps"]
    trajectory_ok = (result.stopped == "answered" and within_budget
                     and order_ok is not False and repeats == 0)

    # Answer.
    should_decline = case["category"] in DECLINE_CATEGORIES
    did_decline = declined_agent(result.answer)
    answer_ok = did_decline if should_decline else bool(task_check(case, result.answer))

    return {
        "tool_choice_ok": tool_choice_ok,
        "missing_tools": missing_tools,
        "forbidden_tools_used": forbidden_used,
        "first_tool_ok": first_tool_ok,
        "args_ok": args_ok,
        "unmatched_steps": unmatched,
        "trajectory_ok": trajectory_ok,
        "order_ok": order_ok,
        "within_budget": within_budget,
        "repeated_calls": repeats,
        "answer_ok": answer_ok,
        "declined": did_decline,
        "fully_correct": tool_choice_ok and args_ok and trajectory_ok and answer_ok,
        "refusal_exact": (refusal_is_exact(result.answer, expected_refusal(case))
                          if should_decline else None),
        "unsupported_numbers": unsupported_numbers(result.answer, case["question"], steps),
        "mentions_internals": agent_mentions_internals(result.answer),
        "tool_errors": sum(bool(s["is_error"]) for s in steps),
    }


def run_case(case: dict, agent) -> dict:
    result = agent.run(case["question"])
    return {
        "id": case["id"], "category": case["category"], "question": case["question"],
        "answer": result.answer,
        **grade_agent(case, result),
        "steps": result.steps,
        "tools_used": result.tools_used,
        "num_steps": len(result.steps),
        "max_steps": case["max_steps"],
        "turns": result.turns,
        "stopped": result.stopped,
        "latency_s": round(result.latency_s, 3),
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    }


# ---- Summary --------------------------------------------------------------------

def _rate(flags) -> float | None:
    known = [f for f in flags if f is not None]
    return round(sum(bool(f) for f in known) / len(known), 3) if known else None


def summarize_agent(rows: list[dict]) -> dict:
    by_category = defaultdict(lambda: {"cases": 0, "fully_correct": 0, "answer_ok": 0})
    for r in rows:
        by_category[r["category"]]["cases"] += 1
        by_category[r["category"]]["fully_correct"] += r["fully_correct"]
        by_category[r["category"]]["answer_ok"] += r["answer_ok"]
    declines = [r for r in rows if r["refusal_exact"] is not None and r["declined"]]
    n = len(rows) or 1
    path_ok = lambda r: r["tool_choice_ok"] and r["args_ok"] and r["trajectory_ok"]
    return {
        "cases": len(rows),
        "fully_correct_rate": _rate([r["fully_correct"] for r in rows]),
        "tool_choice_accuracy": _rate([r["tool_choice_ok"] for r in rows]),
        "argument_accuracy": _rate([r["args_ok"] for r in rows]),
        "trajectory_ok_rate": _rate([r["trajectory_ok"] for r in rows]),
        "answer_accuracy": _rate([r["answer_ok"] for r in rows]),
        # The Phase 4 finding to look for: right answer, wrong path.
        "right_answer_wrong_path_ids": [r["id"] for r in rows if r["answer_ok"] and not path_ok(r)],
        "right_path_wrong_answer_ids": [r["id"] for r in rows if path_ok(r) and not r["answer_ok"]],
        "wrong_tool_ids": [r["id"] for r in rows if not r["tool_choice_ok"]],
        "wrong_args_ids": [r["id"] for r in rows if not r["args_ok"]],
        "forbidden_tool_ids": [r["id"] for r in rows if r["forbidden_tools_used"]],
        "first_tool_wrong_ids": [r["id"] for r in rows if r["first_tool_ok"] is False],
        "over_budget_ids": [r["id"] for r in rows if not r["within_budget"]],
        "out_of_order_ids": [r["id"] for r in rows if r["order_ok"] is False],
        "repeated_call_ids": [r["id"] for r in rows if r["repeated_calls"]],
        "step_limit_ids": [r["id"] for r in rows if r["stopped"] == "step_limit"],
        "wrong_answer_ids": [r["id"] for r in rows if not r["answer_ok"]],
        "exact_refusal_rate": _rate([r["refusal_exact"] for r in declines]),
        "inexact_refusal_ids": [r["id"] for r in declines if not r["refusal_exact"]],
        "unsupported_number_ids": [r["id"] for r in rows if r["unsupported_numbers"]],
        "mentions_internals_ids": [r["id"] for r in rows if r["mentions_internals"]],
        "avg_steps": round(sum(r["num_steps"] for r in rows) / n, 2),
        "avg_turns": round(sum(r["turns"] for r in rows) / n, 2),
        "tool_errors": sum(r["tool_errors"] for r in rows),
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
        "avg_latency_s": round(sum(r["latency_s"] for r in rows) / n, 2),
        "by_category": dict(sorted(by_category.items())),
    }
