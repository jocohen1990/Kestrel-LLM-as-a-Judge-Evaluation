"""Monitoring production traces without answer keys (Phase 6 step 2).

The evals ask "is this answer right?" and need a reference answer to ask it.
Production has no reference answers. What it does have is a trace of every
request (app/tracing.py), and a lot can be checked on a trace alone:

  * Did the reply follow the app's own rules? Exact refusal sentence, no talk
    of "the provided documents", no lists, no links, no leaked prompt.
  * Did it say something about the documents that should be checked? "The
    documents do not contain X" is a claim about the corpus, and in Run 20 it
    was false twice.
  * Did each step do its job? A query rewrite that answers the user instead of
    writing a search query (Run 20, 4 of 40), an agent that refuses without
    calling a single tool (Run 21, t004, t011, t033), figures no tool returned
    (Run 21, t021).
  * Is every claim backed by what the app retrieved? That's the LLM judge's
    faithfulness verdict, which needs only the passages and the answer. It
    costs a call, so it runs on a sample.

Each check comes from a failure seen in Runs 20-21; CHECKS lists which. That
makes Runs 20-21 the checks' training data: they'll look good on it by
construction. traffic_v2 (written before any check was run on it) is the
held-out test.

Every check has a severity:
  alert   should never happen: a leaked prompt, a link, an error, an agent that
          ran out of turns. One is enough to raise an alert.
  review  goes into the review queue for a person to read. Some are wrong
          often (see Run 22); the review numbers say how often.
  info    counted, never queued. Used for rates (refusals, asking back, low
          retrieval scores) whose change over time matters more than any one case.

Then alert RULES look at the whole window of traces: error rate, p95 latency,
cost per 1,000 requests, refusal rate, and how often each review check fires.

Nothing here changes the app or grades it against an answer key. A flag means
"a person should look", not "wrong".
"""
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from app.agent import NOT_REPORTED_TEMPLATE
from app.rag import REFUSAL, REFUSALS
from app.tools import METRIC_NAMES, PERIODS
from evals.agent_eval import agent_mentions_internals, declined_agent, unsupported_numbers
from evals.injection import leaked_phrases
from evals.metrics import mentions_internals
from evals.traces import percentile, summarize

# m1: Phase 6 step 2, built from Runs 20-21.
# m2: two fixes from the held-out Run 22. A number written with a space between
#     digit groups ("2 850", French style) is read as one number; "do not
#     identify" / "name" count as saying the documents lack something.
# m3: a decimal comma ("412,6", 1-2 digits after the comma) is read as a decimal
#     point, since prompt v7 answers in the user's language.
MONITORS_VERSION = "m3"


# ---- What a check needs from a trace ------------------------------------------------

@dataclass
class TraceView:
    """The parts of one trace the checks read, in one flat object."""
    trace_id: str
    case_id: str          # traffic case id, else the request id
    app: str              # "rag" or "agent"
    status: str
    error: str | None
    duration_ms: float
    content: bool         # were question, passages and answer stored?
    question: str
    answer: str
    rewrite: str | None   # RAG r3 only
    top_score: float | None
    passages: list[str]   # RAG: the retrieved passages. Agent: every tool output
    passage_counts: list[int]   # passages each search returned
    steps: list[dict]     # agent tool calls, in the shape evals/agent_eval.py expects
    stopped: str | None


def _search_passages(output: str) -> int:
    """How many passages one search_documents result holds (they're separated by
    blank lines, each starting with its document title)."""
    return len([p for p in output.split("\n\n") if p.strip()]) if output else 0


def view(trace: dict) -> TraceView:
    a = trace["attributes"]
    spans = trace["spans"]

    def first(name):
        return next((s for s in spans if s["name"] == name), None)

    search, rewrite, run = first("bm25.search"), first("query_rewrite"), first("agent.run")
    tools = [s for s in spans if s["kind"] == "tool"]
    steps = []
    for s in tools:
        args = s["attributes"].get("tool.arguments")
        steps.append({"tool": s["attributes"].get("gen_ai.tool.name") or s["name"].removeprefix("tool."),
                      "args": json.loads(args) if args else {},
                      "output": s["attributes"].get("tool.output", ""),
                      "is_error": bool(s["attributes"].get("tool.is_error"))})
    if a.get("app.kind") == "agent":
        passages = [st["output"] for st in steps if st["output"] and not st["is_error"]]
        counts = [_search_passages(st["output"]) for st in steps
                  if st["tool"] == "search_documents" and not st["is_error"]]
    else:
        sa = search["attributes"] if search else {}
        passages = list(sa.get("retrieval.passages", []))
        counts = [len(sa["retrieval.docs"])] if "retrieval.docs" in sa else []
    error = next((s["error"] for s in spans if s["status"] == "error"), None)
    return TraceView(
        trace_id=trace["trace_id"],
        case_id=a.get("traffic.case_id") or a.get("request.id") or trace["trace_id"][:12],
        app=a.get("app.kind", "?"),
        status=trace["status"],
        error=error,
        duration_ms=trace["duration_ms"],
        content=bool(trace.get("content_captured")),
        question=a.get("input.question", ""),
        answer=a.get("output.answer", ""),
        rewrite=(rewrite["attributes"].get("rewrite.output") if rewrite else None),
        top_score=(search["attributes"].get("retrieval.top_score") if search else None),
        passages=passages,
        passage_counts=counts,
        steps=steps,
        stopped=(run["attributes"].get("agent.stopped") if run else None),
    )


# ---- Text helpers -------------------------------------------------------------------

def _norm(text: str) -> str:
    return " ".join((text or "").replace("’", "'").split())


def without_refusal(answer: str) -> str:
    """The answer with the stock refusal sentence (in any language in REFUSALS)
    taken out. The English sentence itself says "the provided documents", which
    the material rule must not count."""
    text = _norm(answer)
    for sentence in REFUSALS.values():
        text = text.replace(sentence, " ")
    return " ".join(text.split())


def contains_refusal(answer: str) -> bool:
    text = _norm(answer)
    return any(sentence in text for sentence in REFUSALS.values())


def is_exact_refusal(answer: str) -> bool:
    return _norm(answer) in REFUSALS.values()


_NOT_REPORTED_RE = re.compile(
    "^" + re.escape(NOT_REPORTED_TEMPLATE).replace(r"\{metric\}", "(?P<metric>.+?)")
    .replace(r"\{period\}", "(?P<period>.+?)") + "$")


def not_reported_parts(answer: str) -> tuple[str, str] | None:
    """(metric, period) if the answer is shaped like the not-reported sentence."""
    m = _NOT_REPORTED_RE.match(_norm(answer))
    return (m.group("metric"), m.group("period")) if m else None


def is_valid_not_reported(answer: str) -> bool:
    parts = not_reported_parts(answer)
    return bool(parts) and parts[0] in METRIC_NAMES.values() and parts[1] in PERIODS


def first_sentence(answer: str) -> str:
    text = _norm(answer)
    return SENTENCE_RE.split(text, maxsplit=1)[0] if text else ""


def declined(answer: str) -> bool:
    """Did the reply decline, in any wording? Judged on the FIRST sentence: Run 20's
    t023 gave two figures and then said the documents "do not contain" the other
    two, which is a partial answer, not a refusal (says_documents_lack catches it)."""
    return (is_exact_refusal(answer) or is_valid_not_reported(answer)
            or declined_agent(first_sentence(answer)))


# A reply saying the documents lack something. Covers the wordings seen in Runs
# 20-21 ("do not contain", "doesn't state", "don't have access to") and their
# close relatives. The exact refusal sentence is removed first.
ABSENCE_RE = re.compile(
    r"\b(?:doesn't|don't|didn't|does not|do not|did not)\s+(?:contain|state|include|mention|"
    r"provide|specify|say|list|give|disclose|cover|identify|name)\b"
    r"|\bnot\s+(?:provided|mentioned|included|specified|stated|disclosed|available)\b"
    r"|\bno (?:information|data|figures?|details?)\b"
    r"|\b(?:don't|do not) have (?:access|information|data|details)\b",
    re.IGNORECASE)

ASK_BACK_RE = re.compile(
    r"\bwhich (?:period|quarter|year|figure|metric|one)\b|\bcould you (?:clarify|specify)\b"
    r"|\bplease (?:clarify|specify)\b|\bdo you mean\b", re.IGNORECASE)

LIST_RE = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+\S|\*\*")
LINK_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
# "The provided documents", "the available filings": the material, by another name.
MATERIAL_RE = re.compile(r"\b(?:provided|available|given)\s+(?:documents?|filings?|data|information)\b",
                         re.IGNORECASE)
# A space (or a no-break space) between digit groups: "2 850" is 2850, not 2 and 850.
DIGIT_GROUP_RE = re.compile(r"(?<=\d)[ \u00a0\u202f](?=\d{3}(?!\d))")
DECIMAL_COMMA_RE = re.compile(r"(?<=\d),(?=\d{1,2}(?!\d))")


def plain_numbers(text: str) -> str:
    """"2 850" -> "2850", "412,6" -> "412.6". English "3,120" is left alone."""
    return DECIMAL_COMMA_RE.sub(".", DIGIT_GROUP_RE.sub("", text))
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"])")
FIRST_PERSON_RE = re.compile(r"\b(?:I|I'm|I'll|I've|I'd|me|my)\b")
_COUNT_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}
COUNT_RE = re.compile(
    r"\b(two|three|four|five|six|seven|eight|[2-8])\s+(?:[a-z]+\s+){0,2}?"
    r"(risks?|risk factors|factors|products|segments|reasons|items|areas|categories)\b",
    re.IGNORECASE)


def stated_counts(answer: str) -> list[int]:
    """Counts of things the answer says there are: "four risks" -> [4]."""
    out = []
    for word, _ in COUNT_RE.findall(answer or ""):
        out.append(int(word) if word.isdigit() else _COUNT_WORDS[word.lower()])
    return out


def sentences(answer: str) -> int:
    text = _norm(answer)
    return len([s for s in SENTENCE_RE.split(text) if s.strip()]) if text else 0


# ---- The checks ---------------------------------------------------------------------

# Thresholds a check or rule uses. Starting points, not tuned values: each is
# stated with where it came from, and changing one is a monitor version change.
THRESHOLDS = {
    # Run 20's p95 was 1.57s for RAG; Run 21's 3.27s for the agent. About 2x each.
    "slow_ms": {"rag": 3000, "agent": 6000},
    # Run 20: below 6, 9 of 12 refuse-intended requests and 4 of 28 others.
    # Picked after seeing that run, so it's an info-level rate only.
    "low_top_score": 6.0,
    # The rewrite prompt asks for at most 25 words.
    "rewrite_max_words": 25,
    # v6 and a3 ask for one or two sentences.
    "max_sentences": 3,
}


@dataclass(frozen=True)
class Check:
    name: str
    severity: str        # "alert", "review" or "info"
    apps: tuple          # which apps it applies to
    needs_content: bool  # needs the question/answer text in the trace
    what: str            # one line, for reports
    origin: str          # the run and cases it came from


CHECKS = {c.name: c for c in [
    Check("request_error", "alert", ("rag", "agent"), False,
          "The request raised an error", "Phase 6 step 1 design"),
    Check("agent_stopped_early", "alert", ("agent",), False,
          "The agent hit its turn limit or ended with no answer", "Phase 4 (step_limit, empty_answer)"),
    Check("prompt_leak", "alert", ("rag", "agent"), True,
          "The reply contains a sentence from a system prompt", "Phase 3 injection eval"),
    Check("link_in_answer", "alert", ("rag", "agent"), True,
          "The reply contains a link", "Phase 3 injection eval (inj103)"),
    Check("says_documents_lack", "review", ("rag", "agent"), True,
          "The reply says the documents don't contain or state something",
          "Run 20 t023, t035; Run 21 t026"),
    Check("refusal_with_more", "review", ("rag", "agent"), True,
          "The refusal sentence, then more text", "Run 21 t029"),
    Check("not_reported_misused", "review", ("rag", "agent"), True,
          "The not-reported sentence for a metric or period outside the lookup table",
          "Run 21 t030"),
    Check("mentions_internals", "review", ("rag", "agent"), True,
          "The reply talks about its documents, passages, tools or lookups",
          "Run 20 t023; Phase 2 format checks"),
    Check("list_or_bold", "review", ("rag", "agent"), True,
          "The reply uses a bulleted list or bold text", "Run 20 t035; Run 21 t002, t024"),
    Check("count_equals_passages", "review", ("rag", "agent"), True,
          "The reply counts things, and the count equals the passages a search returned",
          "Run 20 t024 (four, k=4); Run 21 t024 (three, SEARCH_K=3)"),
    Check("rewrite_not_a_query", "review", ("rag",), True,
          "The query rewrite answered the user instead of writing a search query",
          "Run 20 t002, t039, t040 (t032's poem line isn't caught)"),
    Check("declined_without_tools", "review", ("agent",), True,
          "The agent declined or asked back without calling any tool", "Run 21 t004, t011, t033"),
    Check("unsupported_numbers", "review", ("agent",), True,
          "A figure in the answer that no tool returned (and isn't in the question)",
          "Run 21 t021; Phase 4 agent eval"),
    Check("tool_error", "review", ("agent",), False,
          "A tool call returned an error", "Phase 4"),
    Check("slow_request", "review", ("rag", "agent"), False,
          "Slower than the latency budget for its app", "Runs 20-21 p95, about 2x"),
    Check("unfaithful", "review", ("rag", "agent"), True,
          "The judge found a claim the passages or tool results don't support (sampled)",
          "Phase 2 judge, faithfulness verdict only"),
    Check("refusal", "info", ("rag", "agent"), True,
          "Declined to answer (any wording)", "rate only"),
    Check("asked_back", "info", ("rag", "agent"), True,
          "Asked the user which period or figure they meant", "Run 20 t035; Run 21 t002, t035"),
    Check("long_answer", "info", ("rag", "agent"), True,
          "More than three sentences", "v6 and a3 ask for one or two"),
    Check("low_top_score", "info", ("rag",), False,
          "The best retrieved passage scored below the low-score line", "Run 20"),
]}


def run_checks(v: TraceView) -> list[dict]:
    """Every check that fires on one trace, as {"check", "severity", "detail"}."""
    flags = []

    def flag(name, detail=""):
        c = CHECKS[name]
        if v.app in c.apps:
            flags.append({"check": name, "severity": c.severity, "detail": detail})

    if v.status == "error":
        flag("request_error", v.error or "")
    if v.stopped in ("step_limit", "empty_answer"):
        flag("agent_stopped_early", v.stopped)
    if v.app == "agent" and any(s["is_error"] for s in v.steps):
        flag("tool_error", ", ".join(s["tool"] for s in v.steps if s["is_error"]))
    if v.duration_ms > THRESHOLDS["slow_ms"].get(v.app, float("inf")):
        flag("slow_request", f"{v.duration_ms / 1000:.2f}s")
    if v.top_score is not None and v.top_score < THRESHOLDS["low_top_score"]:
        flag("low_top_score", f"top score {v.top_score}")

    if v.content and v.status != "error":
        answer, rest = v.answer, without_refusal(v.answer)
        if leaked := leaked_phrases(answer):
            flag("prompt_leak", "; ".join(leaked))
        if LINK_RE.search(answer):
            flag("link_in_answer")
        if m := ABSENCE_RE.search(rest):
            flag("says_documents_lack", f'"{m.group(0)}"')
        if contains_refusal(answer) and rest:
            flag("refusal_with_more", f"{len(rest)} more characters")
        if not_reported_parts(answer) and not is_valid_not_reported(answer):
            flag("not_reported_misused", " / ".join(not_reported_parts(answer)))
        internals = (agent_mentions_internals(rest) if v.app == "agent"
                     else mentions_internals(rest))
        if internals or MATERIAL_RE.search(rest):
            hit = MATERIAL_RE.search(rest)
            flag("mentions_internals", f'"{hit.group(0)}"' if hit else "")
        if LIST_RE.search(answer):
            flag("list_or_bold")
        counts = set(stated_counts(answer)) & set(v.passage_counts)
        if counts:
            flag("count_equals_passages", f"says {sorted(counts)[0]}, a search returned "
                                          f"{sorted(counts)[0]}")
        is_decline = declined(answer)
        asked = answer.rstrip().endswith("?") or bool(ASK_BACK_RE.search(answer))
        if is_decline:
            flag("refusal")
        if asked:
            flag("asked_back")
        if sentences(answer) > THRESHOLDS["max_sentences"]:
            flag("long_answer", f"{sentences(answer)} sentences")
        if v.rewrite is not None and rewrite_not_a_query(v.rewrite):
            flag("rewrite_not_a_query", f'"{v.rewrite[:80]}"')
        if v.app == "agent" and not v.steps and (is_decline or asked):
            flag("declined_without_tools")
        if v.app == "agent" and v.steps and not is_decline:
            if extra := unsupported_numbers(plain_numbers(answer), v.question, v.steps):
                flag("unsupported_numbers", ", ".join(extra))
    return flags


def rewrite_not_a_query(rewrite: str) -> bool:
    """A search query isn't in the first person, isn't a question, and the
    rewrite prompt caps it at 25 words."""
    text = _norm(rewrite)
    return (bool(FIRST_PERSON_RE.search(text)) or text.endswith("?")
            or "please" in text.lower() or len(text.split()) > THRESHOLDS["rewrite_max_words"])


# ---- The judge, on a sample ------------------------------------------------------------

# The judge prompt asks for a reference answer. Production has none, so it gets
# this, and only its faithfulness verdict is used (faithfulness is about the
# passages, not the reference: see Step 3 of the judge's rules).
NO_REFERENCE = ("No reference answer: this is a production request. Judge faithfulness only; "
                "the correct verdict is not used.")


def in_sample(trace_id: str, rate: float) -> bool:
    """Pick the same traces every time for a given rate: a sample that changed
    between runs couldn't be compared."""
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    bucket = int(hashlib.sha256(trace_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < rate


def judgeable(v: TraceView) -> bool:
    """A judge call is only worth making on an answer with claims to check."""
    return (v.content and v.status == "ok" and bool(v.answer) and bool(v.passages)
            and not is_exact_refusal(v.answer) and not is_valid_not_reported(v.answer))


def judge_trace(v: TraceView, judge=None) -> dict:
    """One faithfulness verdict. `judge` is evals.judge.judge_answer unless a test
    passes a fake."""
    if judge is None:
        from evals.judge import judge_answer as judge
    try:
        verdict = judge(v.question, v.passages, NO_REFERENCE, v.answer)
    except Exception as exc:   # a broken grader is recorded, never counted
        return {"case": v.case_id, "error": f"{type(exc).__name__}: {exc}"[:300]}
    return {"case": v.case_id, "faithful": verdict.faithful,
            "unsupported_claims": verdict.unsupported_claims,
            "reasoning": verdict.reasoning,
            "tokens": {"input": verdict.input_tokens, "cached_input": verdict.cached_input_tokens,
                       "output": verdict.output_tokens}}


# ---- A window of traces: flags, queue, alerts ---------------------------------------------

def monitor(traces: list[dict], judge_sample: float = 0.0, judge=None) -> dict:
    """Run every check on every trace, the judge on a sample, then the alert rules."""
    views = [view(t) for t in traces]
    apps = sorted({v.app for v in views})
    per_trace = {v.trace_id: run_checks(v) for v in views}

    verdicts = []
    for v in views:
        if judgeable(v) and in_sample(v.trace_id, judge_sample):
            result = judge_trace(v, judge)
            verdicts.append(result)
            if result.get("faithful") is False:
                claims = "; ".join(result["unsupported_claims"])[:200]
                per_trace[v.trace_id].append({"check": "unfaithful", "severity": "review",
                                              "detail": claims})

    n = len(views)
    checks = {}
    for name, c in CHECKS.items():
        if not set(c.apps) & set(apps):
            continue
        cases = [v.case_id for v in views if any(f["check"] == name for f in per_trace[v.trace_id])]
        eligible = sum(1 for v in views if v.app in c.apps and (v.content or not c.needs_content))
        if name == "unfaithful":
            eligible = sum(1 for r in verdicts if "faithful" in r)
        checks[name] = {"severity": c.severity, "what": c.what, "count": len(cases),
                        "eligible": eligible,
                        "rate": round(len(cases) / eligible, 3) if eligible else None,
                        "cases": cases}

    queue = []
    for v in views:
        serious = [f for f in per_trace[v.trace_id] if f["severity"] in ("alert", "review")]
        if serious:
            queue.append({"case": v.case_id, "trace_id": v.trace_id, "app": v.app,
                          "severity": "alert" if any(f["severity"] == "alert" for f in serious)
                          else "review",
                          "flags": serious, "question": v.question, "answer": v.answer})

    stats = summarize(traces)
    result = {
        "monitors_version": MONITORS_VERSION,
        "apps": apps,
        "requests": n,
        "content_captured": sum(v.content for v in views),
        "stats": {
            "error_rate": stats["error_rate"],
            "p95_latency_ms": stats["latency"]["p95_ms"],
            "cost_per_1000_requests": stats["cost_per_1000_requests"],
            "first_start": stats["first_start"],
            "last_start": stats["last_start"],
        },
        "checks": checks,
        "review_queue": queue,
        "judge": {
            "sample_rate": judge_sample,
            "judged": sum(1 for r in verdicts if "faithful" in r),
            "unfaithful": sum(1 for r in verdicts if r.get("faithful") is False),
            "errors": sum(1 for r in verdicts if "error" in r),
            "verdicts": verdicts,
        },
        "thresholds": THRESHOLDS,
    }
    result["alerts"] = apply_rules(result)
    return result


# Window rules. "alert" means act now; "warn" means look today.
RULES = {
    "min_requests": 20,            # below this, rates aren't judged (one request is 5%)
    "max_error_rate": 0.02,
    "max_p95_latency_ms": {"rag": 3000, "agent": 6000},
    # Run 20: $0.99, Run 21: $3.72 per 1,000 requests. About 2x each.
    "max_cost_per_1000": {"rag": 2.00, "agent": 7.50},
    # Run 20 refused 33% of requests and traffic_v1 was written with 30% to be
    # refused; outside this band, something about the traffic or the app moved.
    "refusal_rate_band": (0.10, 0.50),
    "max_review_check_rate": 0.05,   # any one review check on more than 5% of requests
    "max_unfaithful_rate": 0.05,     # of judged answers (needs 5 or more judged)
    "max_queue_share": 0.20,         # a review queue bigger than this won't get read
}


def apply_rules(result: dict) -> list[dict]:
    alerts = []

    def add(level, rule, message, cases=()):
        alerts.append({"level": level, "rule": rule, "message": message, "cases": list(cases)})

    for name, c in result["checks"].items():
        if c["severity"] == "alert" and c["count"]:
            add("alert", name, f"{c['what']}: {c['count']} request(s)", c["cases"])

    n = result["requests"]
    if n < RULES["min_requests"]:
        add("info", "too_few_requests",
            f"{n} requests: rates aren't judged below {RULES['min_requests']}")
        return alerts

    s = result["stats"]
    apps = result["apps"]
    if (s["error_rate"] or 0) > RULES["max_error_rate"]:
        add("alert", "error_rate", f"Error rate {s['error_rate']:.1%} > {RULES['max_error_rate']:.0%}")
    if len(apps) == 1:   # budgets are per app; a mixed window skips them
        app = apps[0]
        p95, limit = s["p95_latency_ms"], RULES["max_p95_latency_ms"].get(app)
        if p95 is not None and limit and p95 > limit:
            add("warn", "p95_latency", f"p95 latency {p95 / 1000:.2f}s > {limit / 1000:.1f}s")
        cost, budget = s["cost_per_1000_requests"], RULES["max_cost_per_1000"].get(app)
        if cost is not None and budget and cost > budget:
            add("warn", "cost", f"${cost:.2f} per 1,000 requests > ${budget:.2f}")
    refusal = result["checks"].get("refusal", {}).get("rate")
    low, high = RULES["refusal_rate_band"]
    if refusal is not None and not low <= refusal <= high:
        add("warn", "refusal_rate", f"Refusal rate {refusal:.0%} outside {low:.0%}-{high:.0%}")
    for name, c in result["checks"].items():
        if (c["severity"] == "review" and name != "unfaithful" and c["rate"] is not None
                and c["rate"] > RULES["max_review_check_rate"]):
            add("warn", name, f"{c['what']}: {c['count']} of {c['eligible']} ({c['rate']:.0%})",
                c["cases"])
    j = result["judge"]
    if j["judged"] >= 5 and j["unfaithful"] / j["judged"] > RULES["max_unfaithful_rate"]:
        add("warn", "unfaithful", f"Judge: {j['unfaithful']} of {j['judged']} sampled answers "
                                  "had an unsupported claim",
            result["checks"]["unfaithful"]["cases"])
    share = len(result["review_queue"]) / n
    if share > RULES["max_queue_share"]:
        add("warn", "queue_size", f"Review queue holds {share:.0%} of requests "
                                  f"(> {RULES['max_queue_share']:.0%})")
    return alerts


# ---- Checking the checks against a human review -------------------------------------------

def load_reviews(path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    for r in rows:
        for key in ("app", "case_id", "verdict"):
            if key not in r:
                raise ValueError(f"{path}: review row missing {key!r}: {r}")
        if r["verdict"] not in ("wrong", "rule_break", "pipeline", "ok"):
            raise ValueError(f"{path}: unknown verdict {r['verdict']!r}")
    return rows


def score_against_reviews(result: dict, reviews: list[dict], traffic_ids: list[str]) -> dict:
    """How well the queue matches a person's reading of the same requests.

    reviews lists every request the reviewer found a problem with (verdict
    wrong, rule_break or pipeline) for each app; any other request in the
    traffic counts as reviewed and fine. Precision: of the queued requests, how
    many had a problem. Recall: of the problems, how many were queued.
    """
    out = {}
    for app in result["apps"]:
        problems = {r["case_id"]: r for r in reviews if r["app"] == app and r["verdict"] != "ok"}
        queued = {q["case"] for q in result["review_queue"] if q["app"] == app}
        hit = sorted(queued & set(problems))
        out[app] = {
            "reviewed": len(traffic_ids),
            "problems": len(problems),
            "queued": len(queued),
            "queued_with_problem": len(hit),
            "precision": round(len(hit) / len(queued), 3) if queued else None,
            "recall": round(len(hit) / len(problems), 3) if problems else None,
            "missed": sorted(set(problems) - queued),
            "false_alarms": sorted(queued - set(problems)),
            "by_verdict": {
                verdict: {"total": sum(1 for p in problems.values() if p["verdict"] == verdict),
                          "queued": sum(1 for c, p in problems.items()
                                        if p["verdict"] == verdict and c in queued)}
                for verdict in ("wrong", "rule_break", "pipeline")},
        }
    return out


# ---- Text output ------------------------------------------------------------------------

def format_report(result: dict, scores: dict | None = None) -> str:
    lines = [f"Monitors {result['monitors_version']}: {result['requests']} requests "
             f"({', '.join(result['apps'])}), {result['content_captured']} with content"]
    s = result["stats"]
    cost = s["cost_per_1000_requests"]
    lines.append(f"Error rate {(s['error_rate'] or 0):.1%}, p95 latency "
                 f"{(s['p95_latency_ms'] or 0) / 1000:.2f}s, "
                 + (f"${cost:.2f} per 1,000 requests" if cost is not None else "cost n/a"))
    lines.append("")
    levels = {"alert": "ALERT", "warn": "WARN ", "info": "info "}
    if result["alerts"]:
        for a in result["alerts"]:
            cases = f"  [{', '.join(a['cases'][:8])}{', ...' if len(a['cases']) > 8 else ''}]" \
                if a["cases"] else ""
            lines.append(f"{levels[a['level']]} {a['message']}{cases}")
    else:
        lines.append("No alerts.")
    lines.append("")
    lines.append(f"{'Check':<24} {'severity':<8} {'count':>5}  cases")
    for name, c in result["checks"].items():
        if c["count"] or c["severity"] != "info":
            cases = ", ".join(c["cases"][:10]) + (", ..." if len(c["cases"]) > 10 else "")
            lines.append(f"{name:<24} {c['severity']:<8} {c['count']:>5}  {cases}")
    j = result["judge"]
    if j["sample_rate"]:
        lines.append(f"\nJudge (faithfulness, {j['sample_rate']:.0%} sample): {j['judged']} judged, "
                     f"{j['unfaithful']} unfaithful, {j['errors']} error(s)")
    lines.append(f"\nReview queue: {len(result['review_queue'])} of {result['requests']} requests")
    if scores:
        lines.append("")
        for app, sc in scores.items():
            p = f"{sc['precision']:.0%}" if sc["precision"] is not None else "n/a"
            r = f"{sc['recall']:.0%}" if sc["recall"] is not None else "n/a"
            lines.append(f"Against the human review ({app}): {sc['queued_with_problem']} of "
                         f"{sc['queued']} queued requests had a problem (precision {p}); "
                         f"{sc['queued_with_problem']} of {sc['problems']} problems were queued "
                         f"(recall {r})")
            bv = sc["by_verdict"]
            lines.append("  by verdict: " + ", ".join(f"{k} {v['queued']}/{v['total']}"
                                                       for k, v in bv.items() if v["total"]))
            if sc["missed"]:
                lines.append(f"  missed: {', '.join(sc['missed'])}")
            if sc["false_alarms"]:
                lines.append(f"  queued, no problem found: {', '.join(sc['false_alarms'])}")
    return "\n".join(lines)


def format_queue(result: dict, width: int = 110) -> str:
    out = []
    for q in result["review_queue"]:
        out.append(f"{q['case']}  [{q['severity']}]  " + ", ".join(f["check"] for f in q["flags"]))
        if q["question"]:
            out.append(f"   Q: {_norm(q['question'])[:width]}")
        if q["answer"]:
            out.append(f"   A: {_norm(q['answer'])[:width]}")
        for f in q["flags"]:
            if f["detail"]:
                out.append(f"   - {f['check']}: {f['detail'][:width]}")
    return "\n".join(out) if out else "The review queue is empty."
