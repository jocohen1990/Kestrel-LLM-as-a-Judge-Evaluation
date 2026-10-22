# Robustness dataset card

| Field | Value |
|---|---|
| Current version | `paraphrases_v1.jsonl` (62 hand-written variants: 31 reworded, 31 casual) |
| Source questions | The 31 golden v3 cases except g032 (ambiguous) |
| Generated variants | `typo`, `typos3`, `lower`, made in code (`evals/robustness.py`) from a fixed seed per case |
| Runner | `python robustness.py` (`--retrieval-only` checks retrieval for free) |
| Author | Hand-written |

## Styles

| Style | How it's made | Example (g006) |
|---|---|---|
| `original` | The golden question, asked in the same run as the baseline | What was Kestrel's revenue in Q1 2026? |
| `reworded` | Hand-written. Same meaning, different vocabulary | How much revenue did Kestrel report for the first quarter of 2026? |
| `casual` | Hand-written. Short and informal, correctly spelled | Q1 2026 sales for Kestrel? |
| `typo` | One typo (swap, drop, double letter, or neighboring key) in one word | What was Kestrel's revebue in Q1 2026? |
| `typos3` | The same, in up to three words | |
| `lower` | Lowercase, no apostrophes, no `?` or commas | what was kestrels revenue in q1 2026 |

## Rules for writing variants

- **Same question, same right answer.** Each variant is graded against its original case's
  `expected_numbers` and `must_include`, so a variant must not change what's being asked.
  Periods and years are always kept ("Q1 2026" can become "the first quarter of 2026", never
  "Q1").
- **Use words people actually use**, including ones the documents don't: "sales" for revenue,
  "head office" for headquarters, "revolver" for revolving credit facility, "Feb" for February.
  That vocabulary gap is part of what's being tested.
- **Out-of-scope questions stay out of scope.** A reworded version of a question the documents
  can't answer must still be refused.
- **Generated typos never touch numbers or years**, and skip "Kestrel", which appears in every
  passage and so carries no search weight. A typo there tested nothing (8 of 27 single typos
  landed on it before this rule).

## Schema

| Field | Meaning |
|---|---|
| `id` | `<source_id>-<style>`, never reused |
| `source_id` | The golden case this varies. Its expected answer is used for grading |
| `style` | `reworded` or `casual` (generated styles aren't stored) |
| `question` | What gets sent to the assistant |

A test checks every eligible golden case has exactly one variant of each hand-written style,
no variant is identical to its original, and every generated variant differs from its original.

## Versioning rules

Same as the golden set: never edit a published version in place. The typo generator is part of
the dataset too: changing it changes the questions, so any change to it is logged below.

## Changelog

- **v1**: 62 hand-written variants. Typo generator v1 (company name excluded from typos).
