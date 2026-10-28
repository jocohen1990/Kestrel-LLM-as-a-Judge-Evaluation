"""Agent eval (Phase 4): grade the agent's tool choice, arguments, trajectory and answer.

Usage:
    python agent_eval.py                          # all 27 cases, agent a1 (the default)
    python agent_eval.py --agent a2               # the same cases with prompt a2
    python agent_eval.py --category calculation   # one category
    python agent_eval.py --ids ag013 ag016        # specific cases
    python agent_eval.py --max-turns 3            # tighter step limit
    python agent_eval.py --min-fully-correct 0.85 --min-answer-accuracy 0.9
                                                  # exit 1 below either (a CI gate)

Cost: one model call per agent turn, usually 2-4 per case (about 70-80 calls
for the full set). No judge calls: every check is rule-based.
See evals/agent_eval.py for what's measured and data/agent/README.md for the dataset.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.agent import AGENT_PROMPTS, DEFAULT_AGENT_VERSION, DEFAULT_MAX_TURNS, KestrelAgent
from app.llm import DEFAULT_MODEL, warmup
from app.paths import AGENT_DIR, RESULTS_DIR
from app.tools import TOOLS_VERSION
from evals.agent_eval import CATEGORIES, load_agent_cases, run_case, summarize_agent


def parse_args():
    parser = argparse.ArgumentParser(description="Run the agent evals.")
    parser.add_argument("--cases", default=AGENT_DIR / "agent_v1.jsonl")
    parser.add_argument("--ids", nargs="*", default=None, help="Only these case ids")
    parser.add_argument("--category", choices=sorted(CATEGORIES), default=None)
    parser.add_argument("--model", default=None, help="Agent model")
    parser.add_argument("--agent", default=DEFAULT_AGENT_VERSION, choices=sorted(AGENT_PROMPTS))
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--min-fully-correct", type=float, default=None,
                        help="Fail (exit 1) if the share of fully correct cases is below this")
    parser.add_argument("--min-answer-accuracy", type=float, default=None,
                        help="Fail (exit 1) if the share of correct answers is below this")
    return parser.parse_args()


def fmt(value) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def describe(step: dict) -> str:
    args = ", ".join(f"{k}={v!r}" for k, v in step["args"].items())
    return f"{step['tool']}({args})" + ("  ERROR" if step["is_error"] else "")


def print_report(run: dict) -> None:
    s = run["summary"]
    print(f"\n=== Agent eval: {s['cases']} cases ===")
    print(f"Agent: {run['model']} (prompt {run['agent_version']}, tools {run.get('tools_version', 't1')}, "
          f"max {run['max_turns']} turns)")
    print(f"  Fully correct:     {fmt(s['fully_correct_rate'])}  (right tools, arguments, path AND answer)")
    print(f"  Tool choice:       {fmt(s['tool_choice_accuracy'])}")
    print(f"  Arguments:         {fmt(s['argument_accuracy'])}")
    print(f"  Trajectory:        {fmt(s['trajectory_ok_rate'])}")
    print(f"  Answer:            {fmt(s['answer_accuracy'])}")
    print(f"  Exact refusals:    {fmt(s['exact_refusal_rate'])} of declined no-answer questions")
    print(f"  Avg steps / turns: {s['avg_steps']} / {s['avg_turns']}, tool errors: {s['tool_errors']}")
    print(f"  Tokens:            {s['input_tokens']} in / {s['output_tokens']} out, "
          f"avg {s['avg_latency_s']}s per case")
    for key, label in (("right_answer_wrong_path_ids", "RIGHT ANSWER, WRONG PATH"),
                       ("right_path_wrong_answer_ids", "Right path, wrong answer"),
                       ("wrong_answer_ids", "Wrong answer"),
                       ("wrong_tool_ids", "Wrong tool choice"),
                       ("forbidden_tool_ids", "Used a forbidden tool"),
                       ("first_tool_wrong_ids", "Wrong first tool"),
                       ("wrong_args_ids", "Required step missing"),
                       ("out_of_order_ids", "Calculated before looking up"),
                       ("over_budget_ids", "Over step budget"),
                       ("repeated_call_ids", "Repeated a call"),
                       ("step_limit_ids", "Hit the turn limit"),
                       ("inexact_refusal_ids", "Declined, but not exactly"),
                       ("unsupported_number_ids", "Numbers no tool returned"),
                       ("mentions_internals_ids", "Mentions tools/searches")):
        if s[key]:
            print(f"  {label + ':':<32} {', '.join(s[key])}")
    print("  By category (fully correct / answer right):")
    for category, c in s["by_category"].items():
        print(f"    {category:<14} {c['fully_correct']}/{c['cases']}, {c['answer_ok']}/{c['cases']}")


def main() -> int:
    args = parse_args()
    cases = load_agent_cases(args.cases)
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
    if args.ids:
        cases = [c for c in cases if c["id"] in set(args.ids)]
    if not cases:
        print("No cases match those filters.")
        return 1

    agent = KestrelAgent(model=args.model, agent_version=args.agent, max_turns=args.max_turns)
    print(f"{len(cases)} cases, up to {args.max_turns} model calls each")
    warmup(args.model)

    rows = []
    for case in cases:
        row = run_case(case, agent)
        rows.append(row)
        status = "PASS" if row["fully_correct"] else ("ANSWER OK, PATH WRONG" if row["answer_ok"] else "FAIL")
        print(f"[{status}] {row['id']} {row['category']:<13} {row['num_steps']} steps")
        if not row["fully_correct"]:
            for step in row["steps"]:
                print(f"      {describe(step)}")
            print(f"      A: {row['answer'][:160]!r}")

    model = args.model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
    run = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "agent_version": args.agent,
        "tools_version": TOOLS_VERSION,
        "max_turns": args.max_turns,
        "cases_file": Path(args.cases).name,
        "summary": summarize_agent(rows),
        "cases": rows,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    for name in (f"agent_eval_{stamp}.json", "agent_eval_latest.json"):
        (RESULTS_DIR / name).write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")

    print_report(run)
    print("\nSaved results/agent_eval_latest.json")

    s = run["summary"]
    failed = False
    for value, limit, label in ((s["fully_correct_rate"], args.min_fully_correct, "fully correct rate"),
                                (s["answer_accuracy"], args.min_answer_accuracy, "answer accuracy")):
        if limit is not None and value is not None and value < limit:
            print(f"\nFAIL: {label} {value:.0%} is below the {limit:.0%} minimum")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
