# LLM Evaluation Framework

An end-to-end evaluation suite for an LLM-powered financial Q&A assistant, built the way
a QA engineer tests software: versioned test data, tested graders, measurable quality
thresholds, and a CI pipeline that blocks regressions.

> **Status:** Phase 1 of 6 complete. See the [roadmap](#roadmap).

## What's being tested

A retrieval-augmented (RAG) assistant that answers questions about **Kestrel Freight
Analytics**, a fictional company with four documents: an annual report, two quarterly
earnings releases, and a risk-factors filing.

The company is invented on purpose. The model can't know any of these facts from its
training data, so every correct answer proves the pipeline worked, and every confident
answer to an out-of-scope question is a measurable hallucination.

```
question ──> BM25 retriever ──> top-k passages ──> LLM ──> answer ──> graders ──> scorecard
                  │                                                      │
          retrieval hit rate                                 correctness, refusals
```

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest -v                          # offline tests: free, no API key needed

cp .env.example .env               # then add your ANTHROPIC_API_KEY
python run_evals.py                # runs the golden set against the real model
```

Results are written to `results/latest.json` plus a timestamped history file.

## Tests vs. evals

| | `pytest` (tests/) | `run_evals.py` |
|---|---|---|
| Checks | That **our code** works | How well **the model** performs |
| Cost | Free, offline | API calls |
| Pass criteria | 100% | Pass rate above a threshold (default 85%) |
| Runs | Every push | Every push, as a CI quality gate |

The graders themselves are unit-tested (`tests/test_metrics.py`). If a grader is wrong,
every score it produces is wrong.

## Current metrics (Phase 1)

| Metric | How it's measured |
|---|---|
| Correctness | Every expected number and phrase appears in the answer (unit-aware: "$412.6 million" and "$412,600,000" both match) |
| Refusal accuracy | Out-of-scope questions must be refused; answerable ones must not be |
| Retrieval hit rate | The retriever fetched every source document the question needs. Currently 18/18 at k≥2 |
| Latency | Average and p95 per question |
| Token usage | Input and output tokens per run |

## Project structure

```
app/                  The system under test
  retriever.py        BM25 keyword search, written from scratch
  llm.py              Single wrapper for every model call
  rag.py              Retrieve -> prompt -> answer
evals/                The evaluation framework
  dataset.py          Loads and validates the golden dataset
  metrics.py          Graders
  report.py           Scorecard aggregation
data/
  corpus/             The four Kestrel documents
  golden/             Versioned golden dataset + dataset card
tests/                Offline test suite (27 tests)
run_evals.py          Eval runner and quality gate
.github/workflows/    CI pipeline
```

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | RAG app, golden dataset v1, rule-based correctness, CI pipeline | ✅ Done |
| 2 | Faithfulness and answer relevance via **LLM-as-judge**, plus judge calibration against 50 hand-labeled cases | ⬜ |
| 3 | Safety and robustness: prompt injection (including injections hidden inside documents), jailbreaks, bias probes, paraphrase and typo robustness, repeat-run consistency | ⬜ |
| 4 | Agent evaluation: LangGraph agent with tools, graded on tool choice, arguments, and trajectory | ⬜ |
| 5 | Model comparison scorecard and a results dashboard | ⬜ |
| 6 | Production-style tracing and monitoring | ⬜ |

## Findings

*To be written up as phases complete.*
