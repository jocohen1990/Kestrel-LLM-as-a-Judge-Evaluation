r"""Replay pretend production traffic through the app, recording a trace per request (Phase 6).

Usage:
    python traffic.py                    # the 40 traffic_v1 requests through the RAG assistant
    python traffic.py --app agent        # ... through the agent instead
    python traffic.py --limit 10         # the first 10 only (cheaper)
    python traffic.py --no-content       # store numbers only: no questions, passages or answers
    python traffic.py --model claude-haiku-5-5
    python traffic.py --copy-to C:\Users\you\kestrel-traces   # also write every trace there

Cost: the RAG assistant makes 2 model calls per request (rewrite + answer), so
the full replay is about 80 small calls, a few cents with Haiku 4.5. The agent
makes 2-4 calls per request. No judge calls: nothing is graded.

This is not an eval. It sends unlabeled questions through `app/service.py`,
the same front door a deployed app would have, and writes what happened to
traces/traces_<timestamp>.jsonl, one line per request. Read them with
`python traces.py`.

A request that raises is recorded (its trace has status "error") and the replay
carries on, the way a server keeps serving after one bad request.

--copy-to (or KESTREL_TRACES_COPY_DIR in .env) writes every trace to a second
file in another folder as well, then checks both files for missing requests.
Added after Run 22 lost one trace in a OneDrive-synced folder with no error:
with one copy inside OneDrive and one outside, a line missing from only the
OneDrive copy points at the sync.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app import tracing
from app.agent import AGENT_PROMPTS, DEFAULT_AGENT_VERSION
from app.llm import warmup
from app.paths import TRACES_DIR, TRAFFIC_DIR
from app.rag import DEFAULT_PROMPT_VERSION, DEFAULT_RETRIEVER_VERSION, PROMPTS, RETRIEVER_VERSIONS
from app.service import handle
from evals.traces import format_summary, load_traces, summarize

DEFAULT_TRAFFIC = TRAFFIC_DIR / "traffic_v1.jsonl"


def load_traffic(path=DEFAULT_TRAFFIC) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if line.strip():
                row = json.loads(line)
                for key in ("id", "session", "channel", "question"):
                    if not row.get(key):
                        raise ValueError(f"{path}, line {n}: missing {key!r}")
                rows.append(row)
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate ids")
    return rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Replay traffic through the app and record traces.")
    p.add_argument("--app", choices=["rag", "agent"], default="rag")
    p.add_argument("--traffic", default=str(DEFAULT_TRAFFIC), help="Traffic .jsonl file")
    p.add_argument("--limit", type=int, default=None, help="Only the first N requests")
    p.add_argument("--model", default=None)
    p.add_argument("--prompt", default=DEFAULT_PROMPT_VERSION, choices=sorted(PROMPTS))
    p.add_argument("--retriever", default=DEFAULT_RETRIEVER_VERSION,
                   choices=sorted(RETRIEVER_VERSIONS))
    p.add_argument("--agent", default=DEFAULT_AGENT_VERSION, choices=sorted(AGENT_PROMPTS))
    p.add_argument("--no-content", action="store_true",
                   help="Don't store any text in the traces (numbers only)")
    p.add_argument("--out", default=None, help="Trace file (default traces/traces_<time>.jsonl)")
    p.add_argument("--copy-to", default=os.getenv("KESTREL_TRACES_COPY_DIR") or None, metavar="DIR",
                   help="Also write every trace to a file of the same name in DIR "
                        "(default: KESTREL_TRACES_COPY_DIR, if set)")
    return p.parse_args(argv)


class CopyingExporter:
    """Writes every trace to several files. One failing doesn't stop the others;
    the failures are raised together afterwards, so trace() still warns."""

    def __init__(self, paths):
        self.exporters = [tracing.JsonlExporter(p) for p in paths]

    def export(self, trace: dict) -> None:
        errors = []
        for e in self.exporters:
            try:
                e.export(trace)
            except OSError as exc:
                errors.append(f"{e.path}: {exc}")
        if errors:
            raise OSError("; ".join(errors))


def build_app(args):
    if args.app == "agent":
        from app.agent import KestrelAgent
        return KestrelAgent(model=args.model, agent_version=args.agent)
    from app.rag import RagAssistant
    return RagAssistant(model=args.model, prompt_version=args.prompt,
                        retriever_version=args.retriever)


def replay(rows: list[dict], app, exporter, capture_content: bool = True, echo=print,
           traffic_file: str | None = None) -> dict:
    """Send each request through the service. Returns counts; never raises for
    one bad request."""
    answered, failed = 0, 0
    # The traffic file's name goes on every trace (added in Phase 6 step 3), so a
    # golden-set candidate can say exactly which request it came from.
    extra = {"traffic.file": traffic_file} if traffic_file else {}
    for row in rows:
        start = time.perf_counter()
        try:
            result = handle(row["question"], app, session_id=row["session"],
                            channel=row["channel"], exporter=exporter,
                            capture_content=capture_content,
                            extra={"traffic.case_id": row["id"], **extra})
            answered += 1
            echo(f"{row['id']}  {time.perf_counter() - start:5.2f}s  "
                 f"{' '.join(result.answer.split())[:90]}")
        except Exception as exc:   # the trace has the details
            failed += 1
            echo(f"{row['id']}  FAILED  {type(exc).__name__}: {exc}")
    return {"answered": answered, "failed": failed}


def missing_traces(rows: list[dict], traces: list[dict]) -> list[str]:
    """Requests that were sent but have no trace in the file."""
    traced = {t["attributes"].get("traffic.case_id") for t in traces}
    return [r["id"] for r in rows if r["id"] not in traced]


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = load_traffic(args.traffic)[: args.limit]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(args.out or TRACES_DIR / f"traces_{stamp}.jsonl")
    files = [out] + ([Path(args.copy_to) / out.name] if args.copy_to else [])
    exporter = CopyingExporter(files) if len(files) > 1 else tracing.JsonlExporter(out)

    app = build_app(args)
    warmup(args.model)   # outside any trace, so connection setup isn't in the numbers
    print(f"Replaying {len(rows)} requests through the {args.app} app...\n")
    counts = replay(rows, app, exporter, capture_content=not args.no_content,
                    traffic_file=Path(args.traffic).name)

    print(f"\n{counts['answered']} answered, {counts['failed']} failed. Traces: {out}")
    traces = None
    for path in files:
        loaded = load_traces(path) if path.exists() else []
        traces = loaded if traces is None else traces
        missing = missing_traces(rows, loaded)
        if len(files) > 1:
            print(f"  {path}: {len(rows) - len(missing)} of {len(rows)} traces")
        if missing:
            # Added after Run 22: one request's trace was lost, and the only sign was
            # the summary saying 29 requests where 30 were sent.
            print(f"WARNING: {len(missing)} of {len(rows)} requests have no trace in {path} "
                  f"({', '.join(missing)}). Any monitor result on this file leaves them out.")
    print()
    print(format_summary(summarize(traces)))
    print("\nOne request in detail: python traces.py --show t001")
    return 0


if __name__ == "__main__":
    sys.exit(main())
