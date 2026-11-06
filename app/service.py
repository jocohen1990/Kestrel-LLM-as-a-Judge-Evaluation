"""The app's front door, as a deployed service would have one (Phase 6).

Every eval calls the assistant directly: `RagAssistant().answer(q)`. A real
deployment puts one function in front of it that every request goes through,
and that is where each request's trace starts. `handle` is that function.

It adds what only the service knows: an id for the request, which user session
and channel it came from, and which release of the app answered it. Those are
what make production traces comparable: "did p95 latency go up after release
X?" needs the release on every trace, and "was this one user's whole session
bad?" needs the session.

It works with either system under test, the RAG assistant (`.answer`) or the
agent (`.run`), and returns whatever that system returns, so a caller can't
tell the difference except that a trace was recorded.
"""
import os
import uuid

from app import tracing


def app_kind(app) -> str:
    if hasattr(app, "run") and hasattr(app, "agent_version"):
        return "agent"
    if hasattr(app, "answer") and hasattr(app, "prompt_version"):
        return "rag"
    raise TypeError(f"don't know how to call {type(app).__name__}: expected a RagAssistant or "
                    "a KestrelAgent")


def app_versions(app) -> dict:
    """Every version that decides this app's behavior, for the trace."""
    if app_kind(app) == "agent":
        from app.tools import TOOLS_VERSION
        return {"kestrel.agent_version": app.agent_version, "kestrel.tools_version": TOOLS_VERSION}
    return {"kestrel.prompt_version": app.prompt_version,
            "kestrel.retriever_version": app.retriever_version}


def handle(question: str, app, *, request_id: str | None = None, session_id: str | None = None,
           channel: str = "api", exporter=None, capture_content: bool | None = None,
           extra: dict | None = None):
    """Answer one request, recording it as one trace.

    extra: more attributes for the root span (traffic.py adds the traffic
    case's id there, so a trace can be matched back to the question that
    produced it).
    """
    kind = app_kind(app)
    attrs = {
        "request.id": request_id or uuid.uuid4().hex[:12],
        "session.id": session_id,
        "request.channel": channel,
        "app.kind": kind,
        # Which build answered. Set KESTREL_RELEASE when deploying; "dev" otherwise.
        "service.version": os.getenv("KESTREL_RELEASE", "dev"),
        "deployment.environment": os.getenv("KESTREL_ENV", "local"),
        **app_versions(app),
        **(extra or {}),
    }
    with tracing.trace("kestrel.request", attrs=attrs, exporter=exporter,
                       capture_content=capture_content) as root:
        root.set_content("input.question", question)
        if kind == "agent":
            result = app.run(question)
        else:
            result = app.answer(question)
        root.set({"output.chars": len(result.answer)})
        root.set_content("output.answer", result.answer)
    return result
