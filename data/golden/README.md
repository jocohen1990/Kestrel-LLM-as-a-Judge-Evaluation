# Golden dataset card

| Field | Value |
|---|---|
| Current version | `golden_v1.jsonl` |
| Cases | 20 |
| Corpus | 4 documents for the fictional company Kestrel Freight Analytics (`data/corpus/`) |
| Author | Hand-written |

## Why a fictional company?

Every fact exists only in this corpus, so the model can't answer from its training
data. A correct answer proves the RAG pipeline worked; a confident answer to an
out-of-scope question is unambiguously a hallucination.

## Category mix (v1)

| Category | Count | What it tests |
|---|---|---|
| `numeric` | 13 | Exact figures: revenue, margins, counts, ranges |
| `factual` | 4 | Names and places |
| `multi_hop` | 1 | Combining facts from two documents |
| `out_of_scope` | 2 | The model must refuse instead of inventing an answer |

## Schema

| Field | Meaning |
|---|---|
| `id` | Stable identifier, never reused |
| `category` | One of the four categories above |
| `question` | What gets sent to the assistant |
| `expected_answer` | Human-readable reference answer (used by the Phase 2 judge) |
| `expected_numbers` | Every number the answer must contain |
| `must_include` | Every phrase the answer must contain (case-insensitive) |
| `source_docs` | Files the retriever must fetch; empty for out-of-scope |

## Versioning rules

- Never edit a published version in place. Copy to `golden_v2.jsonl` and log the change below.
- Results files record which dataset version produced them, so scores are only compared
  within the same version.

## Changelog

- **v1**: initial 20 cases.
