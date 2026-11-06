"""Production-style tracing (Phase 6 step 1).

An eval run knows the right answer to every question. Production doesn't: real
users ask things nobody wrote a test for, and nobody grades the reply. What
production DOES have is a record of every request, and that record is a trace.

A trace is one request, start to finish. It is made of spans: one span per
step, each with a name, a start time, a duration, a status and some
attributes. Spans nest, so a RAG request looks like this:

    kestrel.request                    the whole request (the "root" span)
      rag.answer
        retrieve
          query_rewrite
            llm.chat                   the rewrite call: model, tokens, latency
          bm25.search                  which passages came back, and their scores
        generate
          llm.chat                     the answer call

and an agent request has one span per model turn and one per tool call. Read
top to bottom, a trace answers the questions an eval results file can't: which
step was slow, which step failed, what the model was shown, and what each
request cost.

Why write this instead of installing OpenTelemetry, Langfuse or Phoenix:

  * No new dependencies, so the offline tests still run anywhere.
  * Seeing the whole mechanism (about 250 lines) is the point of a learning project.
  * The attribute names follow OpenTelemetry's GenAI semantic conventions where
    one exists ("gen_ai.request.model", "gen_ai.usage.input_tokens"). Moving to
    a real tracing backend later means writing an exporter, not re-instrumenting
    the app.

How it works:

  * `trace(...)` starts a request and `span(...)` starts a step. Both are `with`
    blocks, so a span always ends, and records the error if its block raised.
  * The current trace and span live in context variables (Python's `contextvars`),
    so a span opened deep inside `app/llm.py` finds its parent without every
    function having to pass it along.
  * OUTSIDE a trace, `span(...)` does nothing and returns a stand-in object.
    That's what keeps every eval exactly as it was: the evals never start a
    trace, so all the spans in the app are skipped.
  * When a trace ends, it goes to the exporter as one JSON object: one line in a
    .jsonl file (`JsonlExporter`), or a list in memory for the tests.
  * Telemetry must never take the app down: if exporting fails, a warning is
    printed and the request's answer is still returned.

Content capture: by default traces store the question, the retrieved passages
and the answer, because the Phase 6 monitors need them (a faithfulness check is
impossible without the passages). Real systems often must not store user text,
so `capture_content=False` (or KESTREL_TRACE_CONTENT=0) keeps every number and
drops every piece of text. Only `set_content` writes text, so that switch
covers all of it.
"""
import json
import os
import sys
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Bumped when the shape of a trace record changes, so old trace files can still
# be read correctly (the same idea as prompt and dataset versions).
TRACE_SCHEMA = "tr1"

# Long text is cut to this many characters. A trace is for diagnosing a request,
# not an archive, and four passages plus an answer fit easily.
MAX_CONTENT_CHARS = 4000

# Span kinds, so a reader can pick out e.g. every model call without knowing names.
KINDS = ("request", "chain", "retrieval", "llm", "agent", "tool", "internal")


def _new_id(n_bytes: int) -> str:
    """Random hex id: 16 bytes for a trace, 8 for a span (OpenTelemetry's sizes)."""
    return uuid.uuid4().hex[: n_bytes * 2]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _clip(value):
    """Cut long text (or each text in a list) to MAX_CONTENT_CHARS."""
    if isinstance(value, str) and len(value) > MAX_CONTENT_CHARS:
        return value[:MAX_CONTENT_CHARS] + f"... [{len(value) - MAX_CONTENT_CHARS} more chars]"
    if isinstance(value, list):
        return [_clip(v) for v in value]
    return value


@dataclass
class Span:
    name: str
    kind: str
    trace_id: str
    span_id: str
    parent_id: str | None
    start_time: str
    duration_ms: float = 0.0
    status: str = "ok"            # "ok" or "error"
    error: str | None = None
    attributes: dict = field(default_factory=dict)
    _start: float = field(default=0.0, repr=False)
    _trace: "Trace | None" = field(default=None, repr=False)

    def set(self, attrs: dict | None = None, **more) -> "Span":
        """Add attributes. Pass a dict for dotted names: span.set({"gen_ai.request.model": m})."""
        self.attributes.update(attrs or {})
        self.attributes.update(more)
        return self

    def set_content(self, key: str, value) -> "Span":
        """Add user or model TEXT, only if this trace captures content."""
        if self._trace is not None and self._trace.capture_content and value is not None:
            self.attributes[key] = _clip(value)
        return self

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "span_id": self.span_id,
                "parent_id": self.parent_id, "start_time": self.start_time,
                "duration_ms": self.duration_ms, "status": self.status,
                "error": self.error, "attributes": self.attributes}


class _NoSpan:
    """What `span()` gives you outside a trace: accepts everything, records nothing."""

    def set(self, attrs=None, **more):
        return self

    def set_content(self, key, value):
        return self


NO_SPAN = _NoSpan()


@dataclass
class Trace:
    trace_id: str
    name: str
    start_time: str
    capture_content: bool = True
    spans: list[Span] = field(default_factory=list)   # in the order they STARTED

    @property
    def root(self) -> Span | None:
        return self.spans[0] if self.spans else None

    def to_dict(self) -> dict:
        root = self.root
        return {
            "schema": TRACE_SCHEMA,
            "trace_id": self.trace_id,
            "name": self.name,
            "start_time": self.start_time,
            "duration_ms": root.duration_ms if root else 0.0,
            "status": "error" if any(s.status == "error" for s in self.spans) else "ok",
            "content_captured": self.capture_content,
            "attributes": dict(root.attributes) if root else {},
            "spans": [s.to_dict() for s in self.spans],
        }


# ---- Exporters: where a finished trace goes ------------------------------------

class NullExporter:
    """The default: traces are built and then dropped."""

    def export(self, trace: dict) -> None:
        pass


class MemoryExporter:
    """Keeps finished traces in a list. For tests."""

    def __init__(self):
        self.traces: list[dict] = []

    def export(self, trace: dict) -> None:
        self.traces.append(trace)


class JsonlExporter:
    """Appends each finished trace to a file as one line of JSON.

    Added after Run 22, which lost one of 30 traces with nothing but a warning on
    stderr: on Windows another program (a sync client such as OneDrive, or an
    antivirus scan) can hold the file for a moment, and the append fails. So a
    failed write is tried again a few times, a little later each time, and the
    exporter counts what it wrote and what it lost, so a caller can say so.
    """

    RETRIES = 4
    RETRY_WAIT_S = 0.1   # then 0.2s, 0.3s

    def __init__(self, path):
        self.path = Path(path)
        self.exported = 0
        self.failed = 0

    def export(self, trace: dict) -> None:
        line = json.dumps(trace, ensure_ascii=False) + "\n"
        for attempt in range(1, self.RETRIES + 1):
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line)
                self.exported += 1
                return
            except OSError:
                if attempt == self.RETRIES:
                    self.failed += 1
                    raise   # trace() turns this into a warning; the request is unaffected
                time.sleep(self.RETRY_WAIT_S * attempt)


_exporter = NullExporter()


def set_exporter(exporter) -> None:
    global _exporter
    _exporter = exporter


def get_exporter():
    return _exporter


# ---- The current trace and span ------------------------------------------------

_current_trace: ContextVar[Trace | None] = ContextVar("kestrel_trace", default=None)
_current_span: ContextVar[Span | None] = ContextVar("kestrel_span", default=None)


def active() -> bool:
    """Is a trace being recorded right now? Lets callers skip work that only a
    trace needs (e.g. collecting retrieval scores)."""
    return _current_trace.get() is not None


def current_span() -> Span | None:
    """The innermost open span. Pass it as `parent=` to a span opened somewhere
    the context variables might not reach (see `span`)."""
    return _current_span.get()


def content_capture_default() -> bool:
    return os.getenv("KESTREL_TRACE_CONTENT", "1").strip().lower() not in ("0", "false", "no")


@contextmanager
def span(name: str, kind: str = "internal", attrs: dict | None = None,
         parent: Span | None = None):
    """One step inside the current trace. Outside a trace, does nothing.

    parent: open this span under a specific span instead of the current one.
    The agent needs it: LangGraph can run a node in a worker thread, and a new
    thread doesn't always see this thread's context variables. Passing the
    parent explicitly (what OpenTelemetry calls propagating context) makes the
    node's spans land in the right trace either way.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown span kind {kind!r}; choose from {KINDS}")
    if not isinstance(parent, Span):
        parent = None
    parent = parent or _current_span.get()
    trace_ = parent._trace if parent is not None else _current_trace.get()
    if trace_ is None:
        yield NO_SPAN
        return
    s = Span(name=name, kind=kind, trace_id=trace_.trace_id, span_id=_new_id(8),
             parent_id=parent.span_id if parent else None, start_time=_now_iso(),
             attributes=dict(attrs or {}), _start=time.perf_counter(), _trace=trace_)
    trace_.spans.append(s)
    trace_token = _current_trace.set(trace_)
    span_token = _current_span.set(s)
    try:
        yield s
    except BaseException as exc:
        s.status = "error"
        s.error = f"{type(exc).__name__}: {exc}"[:500]
        raise
    finally:
        s.duration_ms = round((time.perf_counter() - s._start) * 1000, 3)
        _current_span.reset(span_token)
        _current_trace.reset(trace_token)


def _usage_totals(spans: list[Span]) -> dict:
    """Totals over every model call in the trace, put on the root span so a
    reader can see a request's cost without walking its spans."""
    llm = [s for s in spans if s.kind == "llm"]

    def total(key):
        return sum(s.attributes.get(key) or 0 for s in llm)

    return {
        "llm.calls": len(llm),
        "usage.input_tokens": total("gen_ai.usage.input_tokens"),
        "usage.output_tokens": total("gen_ai.usage.output_tokens"),
        "usage.cache_read_input_tokens": total("gen_ai.usage.cache_read_input_tokens"),
        "usage.cache_creation_input_tokens": total("gen_ai.usage.cache_creation_input_tokens"),
        "llm.models": sorted({s.attributes.get("gen_ai.response.model")
                              or s.attributes.get("gen_ai.request.model") or "" for s in llm} - {""}),
    }


@contextmanager
def trace(name: str = "kestrel.request", attrs: dict | None = None, exporter=None,
          capture_content: bool | None = None):
    """Record one request. The block's body is the root span.

    A trace started inside another trace is just a span of the outer one, so
    calling a traced function from traced code never splits a request in two.
    """
    if _current_trace.get() is not None:
        with span(name, kind="request", attrs=attrs) as s:
            yield s
        return
    if capture_content is None:
        capture_content = content_capture_default()
    t = Trace(trace_id=_new_id(16), name=name, start_time=_now_iso(),
              capture_content=capture_content)
    token = _current_trace.set(t)
    try:
        with span(name, kind="request", attrs=attrs) as root:
            yield root
    finally:
        _current_trace.reset(token)
        if t.root is not None:
            t.root.set(_usage_totals(t.spans))
        try:
            (exporter or _exporter).export(t.to_dict())
        except Exception as exc:   # telemetry must never take the request down
            print(f"warning: trace {t.trace_id} was not exported: {exc}", file=sys.stderr)
