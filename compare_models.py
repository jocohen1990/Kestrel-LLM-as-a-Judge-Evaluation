"""Phase 5: compare assistant models on the same evals, with cost and latency.

Usage:
    python compare_models.py --dry-run          # show what would run and roughly how many calls
    python compare_models.py                    # run the default suites on the default models
    python compare_models.py --models claude-haiku-5-5 claude-sonnet-5-5
    python compare_models.py --suites golden jailbreak
    python compare_models.py --suites golden robustness   # robustness is ~5x golden, so opt-in
    python compare_models.py --from-files       # no API calls: rebuild from saved results
    python compare_models.py --resume           # after an interruption: skip runs saved in the last 12 hours
    python compare_models.py --resume 2         # ... or in the last 2 hours

While it runs, each step prints a banner like
    ===== [step 13/15] Bias probes: claude-haiku-4-5-20251001  |  18m 40s so far, about 2m 30s left =====
"About ... left" starts from the estimates in APPROX_MINUTES and is corrected by
how fast the finished steps actually went.

How it works: for each model it runs the normal eval scripts (run_evals.py,
injection.py, ...) with --model, one after another, exactly as you would by
hand. Each script saves its results file as usual. Then it finds the file each
run saved, and evals/scorecard.py lines the models up.

Everything except the model stays fixed: prompt v6, retriever r3, agent a3,
the same datasets, and the same judge (claude-sonnet-5-5) for every model. A
results file made with other settings is never used, so an old prompt-v1 run
can't sneak into the comparison.

The scripts overwrite their *_latest.json files on every run, so the default
model is always run LAST: when this finishes, the latest files still describe
the default setup. Every model's results stay in the timestamped files.

This is a comparison, not a gate: it exits 0 even when a model fails gates.
Exit code 1 means a run crashed or saved no results.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.llm import DEFAULT_MODEL
from app.paths import RESULTS_DIR, ROOT
from evals.pricing import base_name, price_for
from evals.scorecard import (DEFAULT_SUITES, SUITES, build_scorecard, find_run,
                             format_scorecard, suite_result)

DEFAULT_MODELS = [DEFAULT_MODEL, "claude-haiku-5-5", "claude-sonnet-5-5"]

# Rough API calls per run, for --dry-run. Retriever r3 adds a rewrite call per question.
APPROX_CALLS = {
    "golden": "about 96 calls: 32 questions x 2 with the rewrite, plus 32 judge calls",
    "injection": "about 62 calls: 21 attacks + 10 controls, x 2 with the rewrite",
    "jailbreak": "about 56 calls: 28 rows x 2",
    "bias": "about 56 calls: 28 rows x 2",
    "agent": "about 80 calls: 27 cases, 2-4 turns each",
    "robustness": "about 370 calls: 186 answers (155 variants + 31 originals) x 2",
}

# Rough minutes per run, from the first full comparison (2026-10-29). These are
# the SLOWEST model's times (Sonnet 5.5); Haiku 4.5 took about half as long.
# Robustness wasn't run that night, so its number is scaled up from the golden set.
APPROX_MINUTES = {
    "golden": 2.7,
    "injection": 1.8,
    "jailbreak": 1.6,
    "bias": 1.4,
    "agent": 1.5,
    "robustness": 10.0,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compare assistant models on the same evals.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--suites", nargs="+", default=DEFAULT_SUITES, choices=sorted(SUITES))
    parser.add_argument("--from-files", action="store_true",
                        help="Don't call any API: use the newest saved results for each model")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit")
    parser.add_argument("--resume", nargs="?", type=float, const=12.0, default=None, metavar="HOURS",
                        help="Skip any run already saved in the last HOURS hours (default 12), "
                             "so an interrupted comparison picks up where it stopped")
    return parser.parse_args(argv)


def fmt_duration(seconds: float) -> str:
    """75 -> '1m 15s'; 3700 -> '1h 1m'."""
    seconds = max(0, int(round(seconds)))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"


def time_left(done: list[tuple[str, float]], remaining: list[str]) -> float:
    """Seconds left, from the estimates scaled by how fast this run has actually been.

    done: (suite, seconds it took) for each step finished so far.
    remaining: the suite of each step still to run.
    If the finished steps took half their estimates, the rest are assumed to
    take half theirs too. Before any step has finished, the estimates are used
    as they are. Skipped (resumed) steps aren't in `done`, so they don't skew it.
    """
    estimate_left = sum(APPROX_MINUTES[name] for name in remaining) * 60
    estimate_done = sum(APPROX_MINUTES[name] for name, _ in done) * 60
    if not done or estimate_done == 0:
        return estimate_left
    return estimate_left * (sum(sec for _, sec in done) / estimate_done)


def saved_since(found: tuple[Path, dict] | None, cutoff: datetime) -> bool:
    return found is not None and datetime.fromisoformat(found[1]["timestamp"]) >= cutoff


def run_order(models: list[str]) -> list[str]:
    """The same models, with the default model moved to the end (see the docstring)."""
    default = [m for m in models if base_name(m) == base_name(DEFAULT_MODEL)]
    return [m for m in models if m not in default] + default


def run_suite(suite_name: str, model: str) -> tuple[Path, dict] | None:
    """Run one eval script for one model, then find the results file it saved."""
    suite = SUITES[suite_name]
    started = datetime.now(timezone.utc).isoformat()
    # The scripts' own gates may exit 1 (e.g. run_evals.py's pass-rate gate).
    # That's a result, not a crash: the scorecard reads the gates from the file.
    proc = subprocess.run([sys.executable, suite.script, "--model", model], cwd=ROOT)
    found = find_run(RESULTS_DIR, suite, model)
    if found is None or found[1]["timestamp"] < started:
        print(f"!! {suite.script} --model {model} saved no results (exit code {proc.returncode})")
        return None
    return found


def _repeat_flags(args) -> list[str]:
    """The --models/--suites flags needed to repeat this run."""
    return ["--models", *args.models, "--suites", *args.suites]


def _run_steps(args, steps, skip, results, crashed, done, start) -> None:
    """Run (or skip, or read) every step, printing which step it's on.

    Fills in results, crashed and done as it goes, so if Ctrl+C stops it
    halfway, main() can still say how far it got.
    """
    for i, (model, name) in enumerate(steps, 1):
        suite = SUITES[name]
        step = f"[step {i}/{len(steps)}] {suite.label}: {model}"
        if args.from_files:
            found = find_run(RESULTS_DIR, suite, model)
        elif (model, name) in skip:
            found = skip[(model, name)]
            saved = datetime.fromisoformat(found[1]["timestamp"]).astimezone()
            print(f"\n{step}: already saved {saved:%b %d %H:%M} ({found[0].name}), skipping",
                  flush=True)
        else:
            elapsed = time.monotonic() - start
            left = time_left(done, [n for m, n in steps[i - 1:] if (m, n) not in skip])
            print(f"\n===== {step}  |  {fmt_duration(elapsed)} so far, "
                  f"about {fmt_duration(left)} left =====", flush=True)
            step_start = time.monotonic()
            found = run_suite(name, model)
            took = time.monotonic() - step_start
            done.append((name, took))
            print(f"===== finished {step} in {fmt_duration(took)} =====", flush=True)
        if found is None:
            if not args.from_files:
                crashed.append(f"{name} / {model}")
            continue
        path, run = found
        results[model][name] = suite_result(suite, run, source=path.name)


def main(argv=None) -> int:
    args = parse_args(argv)
    models = run_order(args.models)

    unpriced = [m for m in models if price_for(m) is None]
    if unpriced:
        print(f"Note: no price for {', '.join(unpriced)} in evals/pricing.py; cost will show n/a.")

    steps = [(model, name) for model in models for name in args.suites]

    # With --resume, steps whose results were saved recently enough are skipped.
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=args.resume)
              if args.resume is not None else None)
    skip: dict[tuple[str, str], tuple[Path, dict]] = {}
    if cutoff and not args.from_files:
        for model, name in steps:
            found = find_run(RESULTS_DIR, SUITES[name], model)
            if saved_since(found, cutoff):
                skip[(model, name)] = found
    to_run = [(m, n) for m, n in steps if (m, n) not in skip]

    if args.dry_run:
        total = sum(APPROX_MINUTES[name] for _, name in to_run)
        print(f"Would run {len(to_run)} of {len(steps)} steps, in this order "
              f"(up to about {fmt_duration(total * 60)} in all):")
        for i, (model, name) in enumerate(steps, 1):
            line = f"  [{i}/{len(steps)}] python {SUITES[name].script} --model {model}"
            if (model, name) in skip:
                print(f"{line}    # skipped: saved {skip[(model, name)][0].name}")
            else:
                print(f"{line}    # {APPROX_CALLS[name]}, up to ~{APPROX_MINUTES[name]:g} min")
        return 0

    results: dict[str, dict] = {m: {} for m in models}
    crashed: list[str] = []
    done: list[tuple[str, float]] = []      # (suite, seconds) for each step actually run
    start = time.monotonic()
    try:
        _run_steps(args, steps, skip, results, crashed, done, start)
    except KeyboardInterrupt:
        # Ctrl+C. Every step that finished has already saved its results file.
        # The step that was running didn't finish, so it isn't in `done`.
        print(f"\n\nStopped after finishing {len(done)} of {len(to_run)} steps "
              f"({fmt_duration(time.monotonic() - start)}). No scorecard was saved.\n"
              "To pick up where it stopped:\n"
              "    python compare_models.py --resume " + " ".join(_repeat_flags(args)))
        return 130

    if not args.from_files:
        print(f"\n{len(done)} steps run in {fmt_duration(time.monotonic() - start)}"
              + (f" ({len(skip)} skipped with --resume)." if skip else "."))

    card = build_scorecard(results, args.suites)
    card["timestamp"] = datetime.now(timezone.utc).isoformat()
    card["mode"] = ("from_files" if args.from_files
                    else "resumed" if args.resume is not None else "fresh_runs")

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    for name in (f"model_comparison_{stamp}.json", "model_comparison_latest.json"):
        (RESULTS_DIR / name).write_text(json.dumps(card, indent=2), encoding="utf-8")

    print("\n=== Model comparison ===")
    print(format_scorecard(card))
    print("\nSaved results/model_comparison_latest.json")
    if crashed:
        print("Runs that saved no results: " + "; ".join(crashed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
