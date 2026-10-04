# LLM Evaluation Framework

An end-to-end evaluation suite for an LLM-powered financial Q&A assistant, built the way
a QA engineer tests software: versioned test data, graders that are themselves tested,
measurable quality thresholds, and a CI pipeline that blocks regressions.

> **Status:** Phases 1–2 of 6 complete. See the [roadmap](#roadmap).

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
         retrieval hit rate                 ┌──────────────────────┼─────────────────────┐
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

pytest -v                          # 72 offline tests: free, no API key needed

cp .env.example .env               # then add your ANTHROPIC_API_KEY
python run_evals.py                # golden v2 + LLM judge
python run_evals.py --no-judge     # rule-based graders only (cheaper)
```

### Validating the judge

```bash
python calibrate.py mutate                       # plant errors, see how many the judge catches
python calibrate.py export results/latest.json results/mutations_latest.json
# fill in human_correct / human_faithful (y/n) in the CSV, without opening the .key.json
python calibrate.py score data/labels/labels_<timestamp>.csv
```

For about 50 labels, export from two runs (for example, run `run_evals.py` once with the
default assistant model and once with `--model <another model>`) plus the mutation results.

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
| Retrieval hit rate | Rules | The retriever missed a document the question needs |
| Latency avg / p95, tokens | — | Performance and cost (a warm-up call keeps connection setup out of the numbers) |

## Validating the judge

An LLM judge is a model too, so it gets tested before it's trusted:

1. **Mutation testing** (`calibrate.py mutate`). Planted errors with known correct verdicts:
   a key number swapped, a fabricated claim appended to a correct answer, a confident
   made-up answer replacing a correct refusal. Reports how many each grader catches. The
   appended-claim mutation is invisible to the rule grader by design.
2. **Human calibration** (`calibrate.py export` / `score`). Blind labeling: the sheet hides the
   judge's verdicts and shuffles planted-error answers in with real ones, so the labeler isn't
   anchored. Reports accuracy, Cohen's kappa (agreement corrected for chance), and which way
   the judge errs: too lenient or too strict.

The judge defaults to a different, stronger model than the assistant, to avoid a model
grading its own outputs.

## Golden dataset

32 versioned cases across 8 categories, including calculations, cross-document comparisons,
false premises, distractor numbers, near-miss out-of-scope questions, and an ambiguous question.
See the [dataset card](data/golden/README.md).

## Project structure

```
app/                  The system under test
  retriever.py        BM25 keyword search, written from scratch
  llm.py              Single wrapper for every model call
  rag.py              Retrieve -> prompt -> answer (prompt is versioned)
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
  golden/             golden_v1 (frozen), golden_v2, dataset card
  labels/             Human labeling sheets
tests/                72 offline tests
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

### Run 2: golden v2 with the judge

*Pending.*
