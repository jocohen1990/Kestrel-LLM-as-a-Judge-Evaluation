# Traffic dataset card

| Field | Value |
|---|---|
| Current version | `traffic_v1.jsonl` (40 requests in 16 sessions) |
| Held-out set | `traffic_v2.jsonl` (30 requests in 10 sessions, `u001`-`u030`). Written after the Phase 6 monitors were built and before any monitor was run on it, so it tests them the way `paraphrases_v2` tested the retriever fix |
| Review | `reviews_v1.jsonl`: one person's reading of every Run 20 (RAG) and Run 21 (agent) reply on traffic_v1 that had a problem |
| Review (held-out) | `reviews_v2.jsonl`: the same for Run 22 (both apps on traffic_v2) |
| Used by | `python traffic.py` (Phase 6): replays the requests through the app as if they were production traffic, and records a trace for each |
| Author | Hand-written for Phase 6, after Run 19 |
| Overlap with eval data | None: no question here is word for word a question in any eval dataset (`tests/test_traffic.py` checks) |

## What it's for

Every other dataset in this project is a test set: each question comes with the right answer, and
a grader checks the reply. This one is **pretend production traffic**. It's what users might send
to the deployed assistant on an ordinary day, and in production no one grades the replies. The
questions are replayed through `app/service.py` (the app's front door), which records a trace for
each request. Those traces are what the Phase 6 monitors will read.

So the dataset has **no expected answers**, on purpose. A monitor that needs the right answer
can't run in production.

## Mix

| Kind | Requests | What it is |
|---|---|---|
| `in_scope` | 17 | Ordinary questions the documents answer, some casual or lowercase |
| `multi_step` | 6 | Need two figures and arithmetic, or several facts in one answer |
| `out_of_scope` | 8 | Reasonable questions the documents can't answer (market cap, Q4 2025, the board) or that aren't about Kestrel's filings at all (weather, a poem, investment advice) |
| `noisy` | 4 | Spanish, typos, one-word ("revenue"), shorthand ("kestrel q2 rev??") |
| `attack` | 4 | Direct attempts to get the instructions or the raw documents, or to drop the rules |
| `follow_up` | 1 | t002, "And how much of that was profit?", only makes sense after t001. The app has no memory between requests, so this is a real production pattern no eval covers |

## Fields

| Field | Meaning |
|---|---|
| `id` | `t001` ... `t040` |
| `session` | A pretend user session. Requests in one session are sent one after another |
| `channel` | Where the request came from: `web`, `api` or `slack`. Recorded on the trace |
| `question` | What the user typed |
| `kind` | The writer's description of the request (table above) |
| `intended` | What the writer expects a good reply to do: `answer`, `refuse` or `either` |

`kind` and `intended` are **not labels the monitors may use**. They're there so the monitors
themselves can be checked later: a monitor that flags "probably out of scope" should flag the
`out_of_scope` requests and leave the others alone, and these fields are how to count that. The
same idea as mutation testing the judge (Phase 2): know the answer, then see if the checker finds it.
`intended` is one person's judgment, not a graded answer key, and nothing is scored against it in
step 1.

## Rules for adding requests

- Write new wording. Don't copy a question from an eval dataset: the test fails if you do, because
  traffic that's also a test case would tell the monitors nothing new.
- Keep the mix roughly like real use: mostly ordinary questions, a minority of odd ones.
- A new set of requests is a new version (`traffic_v2.jsonl`). Don't edit v1, so traces made from
  it stay comparable.

## traffic_v2

The same mix as v1 at a smaller size: 13 ordinary questions, 5 multi-step, 6 out of scope (including
"Who founded Kestrel?", which the documents don't name, and an arithmetic question that isn't about
Kestrel), 3 noisy (French, shorthand, typos), 2 attacks and a follow-up. New wording throughout, and
new sessions (`s21`-`s30`). A test checks it shares no question with v1 or any eval dataset.

It was written knowing what the monitors check, which is a risk: it's easy to write requests a
known check will catch. To limit that, every request was written to the v1 mix and kinds, none was
written to trip a particular check, and none was changed after the monitors were run.

## reviews_v1 and reviews_v2

One row per request with a problem, per app. Any request in that traffic file that isn't listed was
read and found fine. `verdict` is one of:

| Verdict | Meaning |
|---|---|
| `wrong` | The reply is wrong, or says something false (about Kestrel or about the documents) |
| `rule_break` | True, but against the app's prompt rules (exact refusal, no lists, only what was asked) |
| `pipeline` | The reply is fine, but a step behind it misbehaved (a rewrite that isn't a query, arithmetic without the calculator) |

`note` says why. `python monitor.py --reviews data/traffic/reviews_v1.jsonl` scores the monitors'
review queue against it. It's one person's judgment, made by reading each reply with its trace's
passages and tool results open; another reader could call some `rule_break`s differently (t002 and
t035 for the agent, for example).
