# Agent dataset card

| Field | Value |
|---|---|
| Current version | `agent_v1.jsonl` (27 cases, 8 categories) |
| System under test | The Phase 4 agent (`app/agent.py`, prompt a1) and its three tools (`app/tools.py`) |
| Corpus | The same 4 Kestrel documents as the golden set (`data/corpus/`), never edited |
| Runner | `python agent_eval.py` (grading in `evals/agent_eval.py`) |
| Author | Hand-written |

## Why a separate dataset

The golden set says what the right **answer** is. An agent eval also needs the right **path**:
which tool, with which arguments, in what order, in how many steps. So each case here lists
`required_steps` alongside the expected answer. The questions are new, not copied from the golden
set or the smoke-test demo questions in `agent.py`, so the agent can't have been tuned to them.

## Categories

| Category | Cases | What it tests | The right path |
|---|---|---|---|
| `lookup` | ag001–ag005 | One reported figure | `get_financial_metric` first, one call |
| `search` | ag006–ag010 | One fact the metric tool doesn't cover | `search_documents` first. ag010 (employee count) sounds like a metric but isn't one |
| `calculation` | ag011–ag016 | A figure that has to be computed | Two lookups, then `calculator`. ag013 and ag016 aren't stated anywhere in the filings |
| `mixed` | ag017–ag018 | Search and calculate | Search for the inputs (and/or look one up), then `calculator` |
| `not_reported` | ag019–ag021 | A figure the filings don't give for that period | Look it up, get "Not reported", decline rather than estimate. The exact decline (prompt a2 on) is "Kestrel did not report \<metric\> for \<period\>." |
| `guidance` | ag022–ag023 | Forecasts, which the metric tool deliberately excludes | `search_documents` for the outlook |
| `false_premise` | ag024 | "Why did revenue fall?" (it rose 19%) | Check the figure, correct the premise |
| `out_of_scope` | ag025–ag027 | Nothing to find, or a request to misuse a tool | Decline. ag026 asks for the calculator by name for a non-Kestrel sum: `calculator` is forbidden |

## Fields

| Field | Meaning |
|---|---|
| `required_steps` | Steps a correct run must contain (see below). Order is checked only for the calculator: it must come after its inputs |
| `first_tool` | The tool prompt a1 says to reach for first, or `null` when more than one start is reasonable |
| `forbidden_tools` | Tools that must not be called at all |
| `max_steps` | Step budget: the shortest correct path plus room for one retry or reworded search |
| `expected_numbers`, `must_include`, `must_include_any` | Answer checks, the same rule grader as the golden set |
| `notes` | Why the case exists |

Required steps are graded on **outcome**, so any reasonable way of getting there passes:

- `{"tool": "get_financial_metric", "args": {...}}`: a call with exactly these arguments (both are fixed lists, so exact match is fair).
- `{"tool": "search_documents", "evidence": "..."}`: a search whose result contains this text. The query itself isn't graded.
- `{"tool": "calculator", "result": [...]}`: a calculation that gives one of these values, so `11.4/116*100` and `11.4/116` both pass.
- `{"any_of": [...]}`: either step is fine (Q2 revenue from the metric tool or from a search).

## How the cases are checked

`tests/test_agent_eval.py` checks every case against the real tools and corpus before any API
money is spent: every metric figure exists (or is really "Not reported"), every evidence string
is in a filing, every calculator target matches hand-written arithmetic, and a perfect trajectory
built with the real tools gets full marks on every case.

## Calculator probes (`calc_probes_v1.jsonl`)

A 6-case diagnostic, not a benchmark, built after Run 15. ag017 ("how much of the credit
facility was undrawn?") skipped the calculator in 4 of 4 runs under both a1 and a2, while ag014
started using it under a2. Instead of guessing at more prompt wording, each probe changes one
thing about ag017, so which probes use the calculator points at the cause.

| Probe | What changes from ag017 | Numbers | Where they come from |
|---|---|---|---|
| cp01 | Nothing: ag017 word for word (control) | 150 − 40 | One paragraph |
| cp02 | Wording: asks for "size minus amount drawn" instead of "undrawn" | 150 − 40 | One paragraph |
| cp03 | Harder numbers | 2,850 − 1,240 | One paragraph (or the metric tool for 2,850) |
| cp04 | Two sources | 64 − 25 | Two paragraphs, two documents |
| cp05 | Not a "what's left over" question: the width of a range | 480 − 470 | One paragraph |
| cp06 | The numbers are in the question, not in a tool result | 150 − 40 | The user |

How to read the results, against the control (cp01 skipping):

- **cp02 uses the calculator**: the word "undrawn" makes the answer feel like a fact to read,
  not a sum to do.
- **cp03 uses it, cp05 doesn't**: the model skips easy round numbers, whatever the wording.
- **cp04 uses it, cp05 doesn't**: numbers that arrive together in one paragraph are read as one fact.
- **cp06 uses it**: numbers inside a tool result are treated differently from the user's numbers.
- **Everything skips**: a2's calculator rule doesn't reach search results at all.

The probes overlap (cp03 changes the numbers *and* keeps one paragraph), so a pattern is a lead
for the next prompt, not proof. Run each probe more than once:
`python agent_eval.py --agent a2 --cases data/agent/calc_probes_v1.jsonl`.

Each probe has a `probe` field saying what it changes. `tests/test_calc_probes.py` checks them
like the main set (real evidence, hand-written arithmetic, perfect runs get full marks), plus two
checks of its own: cp01 must stay identical to ag017, and no filing may state any probe's answer,
since reading a stated figure would be correct.

## Known limits

- 27 cases: one case is about 4 percentage points.
- `max_steps` is a judgment call. An agent that double-checks a not-reported figure with a search
  (which prompt a1 allows) uses 2 of 3.
- The answer checks are the golden set's rule grader, with its known gaps (0.5% number tolerance).
- `unsupported_numbers` and `mentions_internals` are heuristics, reported but not gated.
- The exact-refusal check for `not_reported` cases builds the expected sentence from the case's
  `get_financial_metric` step, so every `not_reported` case must have one (the loader enforces it).
