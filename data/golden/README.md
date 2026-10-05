# Golden dataset card

| Field | Value |
|---|---|
| Current version | `golden_v3.jsonl` (32 cases) |
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

## Versioning rules

- Never edit a published version in place. Copy to a new version and log the change below.
- Results files record which dataset version produced them, so scores are only compared
  within the same version.

## Changelog

- **v3**: v2 plus an `evidence` field on every answerable case. Questions and answers are
  unchanged. Added after run 2, where the document-level retrieval check reported 100% while
  two answers failed because the right paragraph wasn't retrieved.
- **v2**: v1 kept as-is, plus 12 harder cases (g021–g032) added after run 1 scored 100%
  on v1, a sign the set was too easy. New field `must_include_any`.
- **v1**: initial 20 cases.
