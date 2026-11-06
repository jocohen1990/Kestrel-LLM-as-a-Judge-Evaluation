# Traffic dataset card

| Field | Value |
|---|---|
| Current version | `traffic_v1.jsonl` (40 requests in 16 sessions) |
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
