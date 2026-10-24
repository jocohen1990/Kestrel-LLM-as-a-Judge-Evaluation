"""Smoke test for the Phase 4 agent: ask it questions and print every step.

Usage:
    python agent.py                                  # 4 demo questions, one per path
    python agent.py "Who is Kestrel's CFO?"          # your own question(s)
    python agent.py --max-turns 3 "..."

Cost: one model call per turn, usually 2-4 turns per question (about 12 calls
for the demo). No judge calls.

This is NOT the eval: it grades nothing. It shows the trajectory (which tool,
which arguments, what came back) so you can see the agent's behavior before
the graded eval is built on it. The demo questions are deliberately not from
any eval dataset, so they can't be used to tune the agent to the test.
"""
import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.agent import AGENT_PROMPTS, DEFAULT_AGENT_VERSION, DEFAULT_MAX_TURNS, KestrelAgent
from app.llm import DEFAULT_MODEL, warmup
from app.paths import RESULTS_DIR

# One question per path through the graph:
#   lookup only / search only / two lookups + calculator / "not reported" -> refusal
DEMO_QUESTIONS = [
    "How much net income did Kestrel make in the second quarter of 2026?",
    "Does Kestrel carry life insurance on its key executives?",
    "By how many percentage points did Kestrel's gross margin improve from Q1 2025 to Q1 2026?",
    "What were Kestrel's diluted earnings per share for fiscal year 2025?",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Ask the agent questions and print its steps.")
    parser.add_argument("questions", nargs="*", help="Questions (default: 4 demo questions)")
    parser.add_argument("--model", default=None, help="Agent model")
    parser.add_argument("--agent", default=DEFAULT_AGENT_VERSION, choices=sorted(AGENT_PROMPTS))
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    return parser.parse_args()


def short(text: str, width: int = 110) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3] + "..."


def main() -> int:
    args = parse_args()
    questions = args.questions or DEMO_QUESTIONS
    agent = KestrelAgent(model=args.model, agent_version=args.agent, max_turns=args.max_turns)
    warmup(args.model)

    results = []
    for question in questions:
        result = agent.run(question)
        results.append(result)
        print(f"\nQ: {question}")
        for i, step in enumerate(result.steps, start=1):
            arg_text = ", ".join(f"{k}={v!r}" for k, v in step["args"].items())
            flag = "  ERROR" if step["is_error"] else ""
            print(f"  {i}. {step['tool']}({arg_text}){flag}")
            print(f"       -> {short(step['output'])}")
        if not result.steps:
            print("  (no tools used)")
        print(f"A: {result.answer}")
        print(f"   {result.turns} turns, stopped: {result.stopped}, "
              f"{result.input_tokens} tokens in / {result.output_tokens} out, "
              f"{result.latency_s:.1f}s")

    model = args.model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
    run = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "agent_version": args.agent,
        "max_turns": args.max_turns,
        "cases": [asdict(r) for r in results],
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    for name in (f"agent_smoke_{stamp}.json", "agent_smoke_latest.json"):
        (RESULTS_DIR / name).write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved results/agent_smoke_latest.json (full conversations included)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
