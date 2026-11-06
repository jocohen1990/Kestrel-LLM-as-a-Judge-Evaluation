"""Read the traces that traffic.py (or any traced request) recorded (Phase 6). Free: no API calls.

Usage:
    python traces.py                       # summary of the newest file in traces/
    python traces.py traces/traces_<time>.jsonl
    python traces.py --all                 # every file in traces/ together
    python traces.py --show t001           # one request as a tree (traffic id,
                                           #   request id or start of the trace id)
    python traces.py --slowest 5           # the 5 slowest requests, each as a tree
    python traces.py --errors              # every failed request, as a tree
    python traces.py --json                # the summary as JSON
"""
import argparse
import json
import sys
from pathlib import Path

from app.paths import TRACES_DIR
from evals.traces import (find_trace, format_summary, load_traces, render_tree, slowest,
                          summarize, trace_files)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Summarize or inspect recorded traces.")
    p.add_argument("file", nargs="?", help="Trace file (default: newest in traces/)")
    p.add_argument("--all", action="store_true", help="Read every file in traces/")
    p.add_argument("--show", metavar="ID", help="Print one trace as a tree")
    p.add_argument("--slowest", type=int, metavar="N", help="Print the N slowest traces")
    p.add_argument("--errors", action="store_true", help="Print every failed trace")
    p.add_argument("--json", action="store_true", help="Print the summary as JSON")
    return p.parse_args(argv)


def pick_source(args) -> Path | None:
    if args.file:
        return Path(args.file)
    if args.all:
        return TRACES_DIR if trace_files(TRACES_DIR) else None
    files = trace_files(TRACES_DIR)
    return files[-1] if files else None


def main(argv=None) -> int:
    args = parse_args(argv)
    source = pick_source(args)
    if source is None or not source.exists():
        print("No trace files yet. Record some with: python traffic.py --limit 5")
        return 1
    traces = load_traces(source)
    print(f"Read {len(traces)} trace(s) from {source}\n")

    if args.show:
        trace = find_trace(traces, args.show)
        if trace is None:
            print(f"No trace matches {args.show!r}.")
            return 1
        print(render_tree(trace))
        return 0
    if args.slowest:
        print("\n\n".join(render_tree(t) for t in slowest(traces, args.slowest)))
        return 0
    if args.errors:
        failed = [t for t in traces if t["status"] == "error"]
        print("\n\n".join(render_tree(t) for t in failed) if failed else "No failed requests.")
        return 0
    summary = summarize(traces)
    print(json.dumps(summary, indent=2) if args.json else format_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
