"""Jailbreak eval: try to talk the assistant out of its own rules, and check it
still answers legitimate questions that sound risky.

Usage:
    python jailbreak.py                        # 18 attacks + 10 benign questions, default prompt
    python jailbreak.py --prompt v2            # the prompt before any security rules
    python jailbreak.py --kind attack          # only the attacks
    python jailbreak.py --max-asr 0.0 --max-over-refusal 0.2
                                               # exit 1 if any attack works, or more than
                                               # 20% of benign questions are refused

Cost: one assistant call per row (28 by default). No judge calls.
See evals/jailbreak.py for what's measured and data/attacks/README.md for the dataset.
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

from app.llm import DEFAULT_MODEL, sends_temperature, warmup
from app.paths import ATTACKS_DIR, RESULTS_DIR
from app.rag import DEFAULT_PROMPT_VERSION, PROMPTS, RagAssistant
from evals.jailbreak import KINDS, load_jailbreaks, run_case, summarize_jailbreak


def parse_args():
    parser = argparse.ArgumentParser(description="Run the jailbreak evals.")
    parser.add_argument("--cases", default=ATTACKS_DIR / "jailbreak_v1.jsonl")
    parser.add_argument("--ids", nargs="*", default=None, help="Only these case ids")
    parser.add_argument("--kind", choices=sorted(KINDS), default=None)
    parser.add_argument("--model", default=None, help="Assistant model")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT_VERSION, choices=sorted(PROMPTS))
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--max-asr", type=float, default=None,
                        help="Fail (exit 1) if the attack success rate is above this")
    parser.add_argument("--max-over-refusal", type=float, default=None,
                        help="Fail (exit 1) if the share of benign questions refused is above this")
    return parser.parse_args()


def fmt(value) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def print_report(run: dict) -> None:
    s = run["summary"]
    print(f"\n=== Jailbreaks: {s['attacks']} attacks, {s['benign']} benign questions ===")
    print(f"Assistant: {run['model']} (prompt {run['prompt_version']}, k={run['k']})")
    print(f"  Attack success rate:   {fmt(s['attack_success_rate'])} "
          f"({s['attacks_succeeded']}/{s['attacks']}; lower is better)")
    print(f"    asked plainly:       {fmt(s['baseline_success_rate'])}")
    print(f"    with a technique:    {fmt(s['technique_success_rate'])}")
    print(f"  Exact refusals:        {fmt(s['exact_refusal_rate'])} of declined attacks")
    print(f"  Benign answered right: {fmt(s['benign_pass_rate'])}")
    print(f"  Over-refusal rate:     {fmt(s['over_refusal_rate'])} (lower is better)")
    for key, label in (("succeeded_ids", "ATTACKS THAT WORKED"),
                       ("over_refused_ids", "Benign questions refused"),
                       ("benign_failed_ids", "Benign questions answered wrong"),
                       ("benign_evidence_missing_ids", "Benign answer NOT retrieved"),
                       ("inexact_refusal_ids", "Declined, but not exactly"),
                       ("prompt_leak_ids", "System prompt leaked"),
                       ("mentions_internals_ids", "Mentions internals")):
        if s[key]:
            print(f"  {label + ':':<34} {', '.join(s[key])}")
    print("  By topic (attacks succeeded / benign answered):")
    for topic, t in s["by_topic"].items():
        print(f"    {topic:<22} {t['succeeded']}/{t['attacks']} attacks, "
              f"{t['benign_passed']}/{t['benign']} benign")


def main() -> int:
    args = parse_args()
    cases = load_jailbreaks(args.cases)
    if args.kind:
        cases = [c for c in cases if c["kind"] == args.kind]
    if args.ids:
        cases = [c for c in cases if c["id"] in set(args.ids)]
    if not cases:
        print("No cases match those filters.")
        return 1

    assistant = RagAssistant(k=args.k, model=args.model, prompt_version=args.prompt)
    print(f"{len(cases)} assistant calls")
    warmup(args.model)

    rows = []
    for case in cases:
        row = run_case(case, assistant)
        rows.append(row)
        if row["kind"] == "attack":
            status = "ATTACK WORKED" if row["attack_succeeded"] else "blocked      "
        else:
            status = {True: "answered     ", False: "WRONG/REFUSED"}[bool(row["passed"])]
        print(f"[{status}] {row['id']} {row['topic']:<20} {row['technique']}")
        if row["attack_succeeded"] or row["over_refused"]:
            print(f"                {row['answer'][:160]!r}")

    model = args.model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
    run = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "prompt_version": args.prompt,
        "assistant_temperature_sent": sends_temperature(model),
        "cases_file": Path(args.cases).name,
        "k": args.k,
        "summary": summarize_jailbreak(rows),
        "cases": rows,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    for name in (f"jailbreak_{stamp}.json", "jailbreak_latest.json"):
        (RESULTS_DIR / name).write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")

    print_report(run)
    print("\nSaved results/jailbreak_latest.json")

    s = run["summary"]
    failed = False
    if args.max_asr is not None and s["attack_success_rate"] is not None \
            and s["attack_success_rate"] > args.max_asr:
        print(f"\nFAIL: attack success rate {s['attack_success_rate']:.0%} "
              f"is above the {args.max_asr:.0%} limit")
        failed = True
    if args.max_over_refusal is not None and s["over_refusal_rate"] is not None \
            and s["over_refusal_rate"] > args.max_over_refusal:
        print(f"\nFAIL: over-refusal rate {s['over_refusal_rate']:.0%} "
              f"is above the {args.max_over_refusal:.0%} limit")
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
