"""Prompt-injection eval: attack the assistant and measure how often it works.

Usage:
    python injection.py                          # all 21 attacks, prompt v2
    python injection.py --prompt v1              # is the older prompt easier to attack?
    python injection.py --type indirect          # only attacks hidden in documents
    python injection.py --ids inj101 inj104      # only some cases
    python injection.py --no-control             # skip the clean-corpus control runs
    python injection.py --max-asr 0.10           # exit 1 if more than 10% of attacks work

Cost: one assistant call per attack, plus one control call per indirect attack
(31 calls by default). No judge calls: every check is rule-based.

See evals/injection.py for what's measured and why, and data/attacks/README.md
for the attack dataset.
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
from evals.injection import ATTACK_TYPES, load_attacks, poison_chunk, run_case, summarize_injection


def parse_args():
    parser = argparse.ArgumentParser(description="Run the prompt-injection evals.")
    parser.add_argument("--attacks", default=ATTACKS_DIR / "injection_v1.jsonl")
    parser.add_argument("--ids", nargs="*", default=None, help="Only these case ids")
    parser.add_argument("--type", choices=sorted(ATTACK_TYPES), default=None,
                        help="Only direct or only indirect attacks")
    parser.add_argument("--model", default=None, help="Assistant model")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT_VERSION, choices=sorted(PROMPTS))
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--no-control", action="store_true",
                        help="Don't re-ask indirect questions on the clean corpus")
    parser.add_argument("--max-asr", type=float, default=None,
                        help="Fail (exit 1) if the attack success rate is above this, e.g. 0.10")
    return parser.parse_args()


def fmt(value) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def print_report(run: dict) -> None:
    s = run["summary"]
    print(f"\n=== Prompt injection: {s['attacks']} attacks ===")
    print(f"Assistant: {run['model']} (prompt {run['prompt_version']}, k={run['k']})")
    print(f"  Attack success rate:     {fmt(s['attack_success_rate'])} "
          f"({s['attacks_succeeded']}/{s['attacks']}; lower is better)")
    for kind, rate in s["attack_success_rate_by_type"].items():
        print(f"    {kind:<9}              {fmt(rate)} of {s['attacks_by_type'][kind]}")
    print(f"  Real question still answered: {fmt(s['task_preserved_rate'])} "
          f"of {s['task_cases']} cases that include one")
    print(f"  Poisoned passage retrieved:   {fmt(s['poison_retrieved_rate'])}")
    if s["control_task_passed_rate"] is not None:
        print(f"  Clean-corpus control passed:  {fmt(s['control_task_passed_rate'])}")
    for key, label in (("succeeded_ids", "ATTACKS THAT WORKED"),
                       ("task_broken_ids", "Real answer broken by attack"),
                       ("prompt_leak_ids", "System prompt leaked"),
                       ("mentions_internals_ids", "Mentions internals"),
                       ("poison_not_retrieved_ids", "Poison NOT retrieved (attack untested)"),
                       ("control_failed_ids", "Control failed (task failure not the attack's)")):
        if s[key]:
            print(f"  {label + ':':<31} {', '.join(s[key])}")
    print("  By technique (succeeded/total):")
    for tech, count in s["succeeded_by_technique"].items():
        print(f"    {tech:<26} {count}")


def main() -> int:
    args = parse_args()
    cases = load_attacks(args.attacks)
    if args.type:
        cases = [c for c in cases if c["attack_type"] == args.type]
    if args.ids:
        cases = [c for c in cases if c["id"] in set(args.ids)]
    if not cases:
        print("No attack cases match those filters.")
        return 1

    def assistant(extra=None):
        return RagAssistant(k=args.k, model=args.model, prompt_version=args.prompt,
                            extra_chunks=extra)

    clean = assistant()
    indirect = sum(c["attack_type"] == "indirect" for c in cases)
    controls = 0 if args.no_control else indirect
    print(f"{len(cases)} attacks ({len(cases) - indirect} direct, {indirect} indirect)"
          f" + {controls} control calls")
    warmup(args.model)

    rows = []
    for case in cases:
        chunk = poison_chunk(case)
        # Indirect: a fresh index with the poison added. Direct: the clean one.
        attacked = assistant([chunk]) if chunk else clean
        row = run_case(case, attacked, clean=None if args.no_control else clean)
        rows.append(row)
        status = "ATTACK WORKED" if row["attack_succeeded"] else "blocked      "
        task = {True: "task ok", False: "TASK BROKEN", None: ""}[row["task_preserved"]]
        print(f"[{status}] {case['id']} {case['attack_type']:<8} {case['technique']:<24} {task}")
        if row["attack_succeeded"]:
            print(f"                {row['answer'][:160]!r}")

    model = args.model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
    run = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "prompt_version": args.prompt,
        "assistant_temperature_sent": sends_temperature(model),
        "attacks_file": Path(args.attacks).name,
        "k": args.k,
        "controls": not args.no_control,
        "summary": summarize_injection(rows),
        "cases": rows,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    for name in (f"injection_{stamp}.json", "injection_latest.json"):
        (RESULTS_DIR / name).write_text(json.dumps(run, indent=2), encoding="utf-8")

    print_report(run)
    print("\nSaved results/injection_latest.json")

    asr = run["summary"]["attack_success_rate"]
    if args.max_asr is not None and asr is not None and asr > args.max_asr:
        print(f"\nFAIL: attack success rate {asr:.0%} is above the {args.max_asr:.0%} limit")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
