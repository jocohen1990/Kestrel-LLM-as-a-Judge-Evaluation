"""Replay pretend production traffic through the app, recording a trace per request (Phase 6).

Usage:
    python traffic.py                    # the 40 traffic_v1 requests through the RAG assistant
    python traffic.py --app agent        # ... through the agent instead
    python traffic.py --limit 10         # the first 10 only (cheaper)
    python traffic.py --no-content       # store numbers only: no questions, passages or answers
    python traffic.py --model claude-haiku-5-5

Cost: the RAG assistant makes 2 model calls per request (rewrite + answer), so
the full replay is about 80 small calls, a few cents with Haiku 4.5. The agent
makes 2-4 calls per request. No judge calls: nothing is graded.

This is not an eval. It sends unlabeled questions through `app/service.py`,
the same front door a deployed app would have, and writes what happened to
traces/traces_<timestamp>.jsonl, one line per request. Read them with
`python traces.py`.

A request that raises is recorded (its trace has status "error") and the replay
carries on, the way a server keeps serving after one bad request.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone

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
    return p.parse_args(argv)


def build_app(args):
    if args.app == "agent":
        from app.agent import KestrelAgent
        return KestrelAgent(model=args.model, agent_version=args.agent)
    from app.rag import RagAssistant
    return RagAssistant(model=args.model, prompt_version=args.prompt,
                        retriever_version=args.retriever)


def replay(rows: list[dict], app, exporter, capture_content: bool = True, echo=print) -> dict:
    """Send each request through the service. Returns counts; never raises for
    one bad request."""
    answered, failed = 0, 0
    for row in rows:
        start = time.perf_counter()
        try:
            result = handle(row["question"], app, session_id=row["session"],
                            channel=row["channel"], exporter=exporter,
                            capture_content=capture_content,
                            extra={"traffic.case_id": row["id"]})
            answered += 1
            echo(f"{row['id']}  {time.perf_counter() - start:5.2f}s  "
                 f"{' '.join(result.answer.split())[:90]}")
        except Exception as exc:   # the trace has the details
            failed += 1
            echo(f"{row['id']}  FAILED  {type(exc).__name__}: {exc}")
    return {"answered": answered, "failed": failed}


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = load_traffic(args.traffic)[: args.limit]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = args.out or TRACES_DIR / f"traces_{stamp}.jsonl"
    exporter = tracing.JsonlExporter(out)

    app = build_app(args)
    warmup(args.model)   # outside any trace, so connection setup isn't in the numbers
    print(f"Replaying {len(rows)} requests through the {args.app} app...\n")
    counts = replay(rows, app, exporter, capture_content=not args.no_content)

    print(f"\n{counts['answered']} answered, {counts['failed']} failed. Traces: {out}\n")
    print(format_summary(summarize(load_traces(out))))
    print("\nOne request in detail: python traces.py --show t001")
    return 0


if __name__ == "__main__":
    sys.exit(main())
