# Golden dataset card

| Field | Value |
|---|---|
| Current version | `golden_v3.jsonl` (32 cases): the CI default |
| Next version | `golden_v4.jsonl` (40 cases): v3 plus 8 cases from production traffic. Not the default yet |
| Previous versions | `golden_v2.jsonl` (32 cases), `golden_v1.jsonl` (20 cases), both kept unchanged |
| Corpus | 4 documents for the fictional company Kestrel Freight Analytics (`data/corpus/`) |
| Author | Hand-written |

## Why a fictional company?

Every fact exists only in this corpus, so the model can't answer from its training
data. A correct answer proves the RAG pipeline worked; a confident answer to an
out-of-scope question is unambiguously a hallucination.

## Category mix (v2 and v3)

| Category | Count | What it tests |
|---|---|---|
| `numeric` | 15 | Exact figures, including 2 distractor cases where a similar number sits in the same passage |
| `factual` | 4 | Names and places |
| `multi_hop` | 1 | Combining facts from two documents |
| `calculation` | 3 | Arithmetic over figures in the documents (margin, sum, difference) |
| `comparison` | 2 | Change between two periods, across documents |
| `false_premise` | 2 | The question assumes something untrue; the model should correct it, not play along |
| `out_of_scope` | 4 | Must refuse. v2 adds 2 near-misses that *look* answerable (Q3 2026, FY2024 net income) |
| `ambiguous` | 1 | Underspecified question; a good answer names the period or asks which one |

v4 adds 8 cases (g033–g040), one per production request: 3 `multi_fact` (a new category: several facts
in one question), 2 `factual`, 1 `numeric`, 1 `ambiguous`, 1 `out_of_scope`.

## Schema

| Field | Meaning |
|---|---|
| `id` | Stable identifier, never reused |
| `category` | One of the categories above |
| `question` | What gets sent to the assistant |
| `expected_answer` | Human-readable reference answer, given to the LLM judge |
| `expected_numbers` | Every number the answer must contain |
| `must_include` | Every phrase the answer must contain (case-insensitive) |
| `must_include_any` | *(v2)* At least one of these phrases must appear. Used where several wordings are valid, e.g. "rose" / "grew" / "increased" |
| `source_docs` | Files the retriever must fetch; empty for out-of-scope and ambiguous |
| `evidence` | *(v3)* Short strings copied exactly from the corpus that must appear in the retrieved **passages**, e.g. `"116.0"`. Empty for out-of-scope and ambiguous. A test checks every string exists in its source document |
| `origin` | *(v4, optional)* For a case promoted from production: `traffic_id`, `traffic_file`, `run` and `found_by` (which monitors, comparison or review flagged it). The question is the user's words, unchanged; a test checks it matches the traffic file |
| `language` | *(v4, optional, default `en`)* The question's language. When it isn't English, the number grader reads local formats first ("2 850", "412,6"), as the robustness eval does for translated variants |

## Versioning rules

- Never edit a published version in place. Copy to a new version and log the change below.
- Results files record which dataset version produced them, so scores are only compared
  within the same version.

## Changelog

- **v4**: v3 unchanged, plus 8 cases promoted from production traffic (Phase 6 step 3): 7 after Run 23 and g040 after Run 24. The Run 23 seven are
  the five requests the Run 20–22 reviews found the RAG assistant wrong on, the Spanish question
  the agent refused (t033), and u022.

  | Case | From | Why |
  |---|---|---|
  | g033 | t023 | Four facts in one question; the search missed the headcount and customer paragraph and the reply said the documents don't have them |
  | g034 | t024 | "Which risks": the filing has six; the reply said four, the number of passages retrieved |
  | g035 | t033 | The CEO question in Spanish (`language: es`); the agent refused it without a tool call |
  | g036 | t035 | The one-word question "revenue"; the reply said the annual report doesn't state revenue |
  | g037 | u001 | "Head office": typo correction turned it into "had officer" and the reply refused |
  | g038 | u025 | Customer count in French (`language: fr`); the right passage came back and the reply refused |
  | g039 | u022 | "Who founded Kestrel?": not in the documents, and the reply added a sentence after the refusal |
  | g040 | u018 | Added after Run 24: CEO, CFO and headquarters city. Run 22 got it right; retriever r4 lost the headquarters paragraph and the reply refused |

  Each of these went wrong in production for at least one app, so the current default setup may
  well fail some of them in the evals: that's why v4 isn't the CI default yet. New category
  `multi_fact`, new optional fields `origin` and `language`.

- **v3**: v2 plus an `evidence` field on every answerable case. Questions and answers are
  unchanged. Added after run 2, where the document-level retrieval check reported 100% while
  two answers failed because the right paragraph wasn't retrieved.
- **v2**: v1 kept as-is, plus 12 harder cases (g021–g032) added after run 1 scored 100%
  on v1, a sign the set was too easy. New field `must_include_any`.
- **v1**: initial 20 cases.
