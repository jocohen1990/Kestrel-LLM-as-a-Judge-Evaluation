# LLM Evaluation Framework

An end-to-end evaluation suite for an LLM-powered financial Q&A assistant, built the way
a QA engineer tests software: versioned test data, graders that are themselves tested,
measurable quality thresholds, and a CI pipeline that blocks regressions.

> **Status:** Phases 1–2 of 6 complete; Run 3 (prompt v2) pending. See the [roadmap](#roadmap).

## What's being tested

A retrieval-augmented (RAG) assistant that answers questions about **Kestrel Freight
Analytics**, a fictional company with four documents: an annual report, two quarterly
earnings releases, and a risk-factors filing.

The company is invented on purpose. The model can't know any of these facts from its
training data, so every correct answer proves the pipeline worked, and every confident
answer to an out-of-scope question is a measurable hallucination.

```
question ─> BM25 retriever ─> top-k passages ─> assistant LLM ─> answer
                 │                                                 │
   retrieval (doc + passage)                ┌──────────────────────┼─────────────────────┐
                                            ▼                      ▼                     ▼
                                    rule-based graders       LLM judge            format checks
                                   (numbers, phrases,   (faithfulness,       (exact refusals,
                                    refusals)            correctness)          no "the context")
                                                                   │
                                            judge validated by mutation testing + human labels
```

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest -v                          # 97 offline tests: free, no API key needed

cp .env.example .env               # then add your ANTHROPIC_API_KEY
python run_evals.py                # golden v3, prompt v2, LLM judge
python run_evals.py --prompt v1    # same, with the original prompt (for before/after)
python run_evals.py --no-judge     # rule-based graders only (cheaper)
```

### Validating the judge

```bash
python calibrate.py mutate                       # plant errors, see how many the judge catches
python calibrate.py export results/latest.json results/mutations_latest.json
# fill in human_correct / human_faithful (y/n) in the CSV, without opening the .key.json
python calibrate.py score data/labels/labels_<timestamp>.csv
```

The sheet opens in Excel or Google Sheets and can be saved in any CSV format Excel offers.
Rows are shuffled, so labeling the first 50 gives a representative sample.

## Tests vs. evals

| | `pytest` (tests/) | `run_evals.py` |
|---|---|---|
| Checks | That **our code** works | How well **the model** performs |
| Cost | Free, offline | API calls |
| Pass criteria | 100% | Thresholds: rule pass rate ≥ 85%, judge faithfulness ≥ 90% |
| Runs | Every push | Every push, as a CI quality gate |

## Metrics

| Metric | Grader | What it catches |
|---|---|---|
| Rule pass rate | Rules | Expected numbers and phrases missing; refusing answerable questions; answering unanswerable ones |
| Judge correctness | LLM judge | Wrong answers phrased in ways rules can't parse; mishandled false premises and ambiguity |
| Judge faithfulness | LLM judge | **Any** claim not supported by the retrieved passages, even when the main answer is right |
| Rule/judge agreement | Both | Where the two graders disagree, one of them is wrong; those cases get reviewed |
| Exact refusals | Rules | Instruction-following: refusing with exactly the required wording, nothing extra |
| Mentions internals | Rules | Answers that expose the pipeline to users ("According to the context…") |
| Retrieval (document) | Rules | The retriever missed a document the question needs |
| Retrieval (passage) | Rules | The retriever missed the **paragraph** holding the answer, even if it found the right document. Uses the `evidence` strings in golden v3 |
| Latency avg / p95, tokens | — | Performance and cost (a warm-up call keeps connection setup out of the numbers) |

## Validating the judge

An LLM judge is a model too, so it gets tested before it's trusted:

1. **Mutation testing** (`calibrate.py mutate`). Planted errors with known correct verdicts:
   the asked-for figure swapped (`number_swap`: incorrect and unfaithful), a supporting figure
   swapped (`detail_swap`: unfaithful), a fabricated claim appended to a correct answer
   (`false_claim`: unfaithful), and a confident made-up answer replacing a correct refusal
   (`fabrication`). Reports how many each grader catches. The appended-claim mutation is
   invisible to the rule grader by design.
2. **Human calibration** (`calibrate.py export` / `score`). Blind labeling: the sheet hides the
   judge's verdicts and shuffles planted-error answers in with real ones, so the labeler isn't
   anchored. Reports accuracy, Cohen's kappa (agreement corrected for chance), and which way
   the judge errs: too lenient or too strict.

The judge defaults to a different, stronger model than the assistant, to avoid a model
grading its own outputs. If a judge reply isn't valid JSON, it's retried once; a reply that
still fails is recorded as a judge error, never counted as a pass or fail.

## Golden dataset

32 versioned cases across 8 categories, including calculations, cross-document comparisons,
false premises, distractor numbers, near-miss out-of-scope questions, and an ambiguous question.
v3 adds `evidence` strings for passage-level retrieval checks. Earlier versions stay frozen.
See the [dataset card](data/golden/README.md).

## Project structure

```
app/                  The system under test
  retriever.py        BM25 keyword search, written from scratch
  llm.py              Single wrapper for every model call
  rag.py              Retrieve -> prompt -> answer (prompts v1 and v2, both kept)
evals/                The evaluation framework
  dataset.py          Loads and validates the golden dataset
  metrics.py          Rule-based graders
  judge.py            LLM-as-judge
  runner.py           Runs one case through every grader
  mutations.py        Planted errors for testing the judge
  calibration.py      Blind labeling export, Cohen's kappa, confusion matrix
  report.py           Scorecard aggregation
data/
  corpus/             The four Kestrel documents
  golden/             golden_v1, golden_v2 (frozen), golden_v3, dataset card
  labels/             Human labeling sheets
tests/                97 offline tests
run_evals.py          Eval runner and quality gates
calibrate.py          Judge validation commands
.github/workflows/    CI pipeline
```

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | RAG app, golden dataset v1, rule-based correctness, CI pipeline | ✅ Done |
| 2 | Golden v2, LLM-as-judge (faithfulness, correctness), format checks, mutation testing, human calibration | ✅ Done |
| 3 | Safety and robustness: prompt injection (including injections hidden inside documents), jailbreaks, bias probes, paraphrase and typo robustness, repeat-run consistency | ⬜ |
| 4 | Agent evaluation: LangGraph agent with tools, graded on tool choice, arguments, and trajectory | ⬜ |
| 5 | Model comparison scorecard and a results dashboard | ⬜ |
| 6 | Production-style tracing and monitoring | ⬜ |

## Findings

### Run 1: golden v1, rule-based graders, claude-haiku-4-5

- **20/20 passed, 100% retrieval hit rate.** Treated as a signal that the test set was too
  easy rather than proof of quality, which led to golden v2.
- **Unchecked claims.** Answers often added figures nobody asked for (g001 added the
  year-over-year growth rate). The rule grader couldn't verify them, which motivated the
  faithfulness judge.
- **Instruction drift.** One of two refusals (g020) added commentary after the required
  exact refusal sentence.
- **Pipeline leakage.** g005 began "According to the context…", wording that means
  nothing to an end user.
- **Cold start.** The first call took 2.4s against a p95 of 1.2s for the rest, now excluded
  via a warm-up call.

### Run 2: golden v2, prompt v1, LLM judge

Assistant claude-haiku-4-5, judge claude-sonnet-5-5. Rule pass rate 29/32 (91%), judge
correctness 97%, judge faithfulness 97%, rule/judge agreement 94%.

- **The retrieval metric was hiding failures.** It reported 100%, but it only checked that the
  right *document* came back. For g022 (first-half revenue), the retriever returned the Q2
  document's guidance paragraph instead of its results paragraph. The model then correctly
  said it couldn't find Q2 revenue, and **both graders blamed the model** for a retriever
  failure. g032 failed the same way: the paragraph with FY2025 revenue never reached the model.
  → Added a passage-level check (golden v3). True rate: 26/27 at k=4; g022 needs k=6.
- **The refusal sentence became an opener.** 6 of 32 answers started with "I don't know based on
  the provided documents", including 2 that then answered (g022, g026). Only 1 of 4 out-of-scope
  refusals used the exact wording. → Prompt v2.
- **Pipeline leakage persisted:** 7 of 32 answers mentioned "the context". → Prompt v2.
- **The judge caught a hallucination inside a correct refusal.** g030 refused properly but
  claimed the documents contain "Q1 2026 guidance". They don't. No rule could see this.
- **Likely judge leniency on g032.** Asked "What was Kestrel's revenue?", the model answered with
  *guidance*, and the judge accepted it. Flagged for the human calibration round.
- **Graders disagreed on g026.** The model corrected the false premise (judge: correct) but
  opened with the refusal sentence (rules: fail). Both have a point; the answer contradicts itself.

**Mutation check:**

| Mutation | Judge | Rule grader |
|---|---|---|
| False claim appended | 28/28 | 3/28 |
| Number swapped | 20/25 as scored; **25/25** flagged unfaithful | 21/25 |
| Refusal replaced by fabrication | 4/4 | 4/4 |

- The rule grader's 3 false-claim "catches" were answers that already failed before the
  mutation, so it caught **0 of 25** planted claims on passing answers. That gap is why the
  judge exists.
- The 5 number-swap "misses" were bugs in the mutation generator, not the judge: it expected
  "incorrect" even when the swapped figure wasn't the one asked about, and it sometimes changed
  period labels ("Q1" to "Q2") instead of figures. The judge flagged all 5 as unfaithful.
  → Split into `number_swap` and `detail_swap`; period labels are skipped.
- 1 of 58 judge replies was malformed JSON. → The judge now retries once.

### Changes after Run 2

| Change | Why |
|---|---|
| Golden v3 with `evidence` strings; passage-level retrieval metric | Document-level hit rate hid two retrieval failures |
| Prompt v2 (v1 kept) | Refusal used as an opener; "the context" leaking into answers; no guidance for false premises or ambiguity |
| `detail_swap` mutation type; period labels never swapped | 5 false "misses" in the mutation check |
| Number pattern fixed (`2026,` no longer read as one number) | Found while fixing the mutation generator |
| Judge retries once on malformed JSON | 1 judge error in 58 calls |
| Labeling sheet handles Excel's CSV formats | Excel's "CSV UTF-8" adds a hidden marker that broke the reader |

### Run 3: prompt v1 vs. v2 on golden v3

*Pending.*
