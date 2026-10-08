# LLM Evaluation Framework

An end-to-end evaluation suite for an LLM-powered financial Q&A assistant, built the way
a QA engineer tests software: versioned test data, graders that are themselves tested,
measurable quality thresholds, and a CI pipeline that blocks regressions.

> **Status:** Phases 1–2 of 6 complete, through Run 3 and the first human calibration round. See the [roadmap](#roadmap).

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

pytest -v                          # 104 offline tests: free, no API key needed

cp .env.example .env               # then add your ANTHROPIC_API_KEY
python run_evals.py                # golden v3, prompt v2, LLM judge
python run_evals.py --prompt v1    # same, with the original prompt (for before/after)
python run_evals.py --no-judge     # rule-based graders only (cheaper)
```

### Validating the judge

```bash
python calibrate.py mutate                       # plant errors, see how many the judge catches
python calibrate.py export results/latest.json results/mutations_latest.json
# label following data/labels/LABELING_GUIDE.md, without opening the .key.json
python calibrate.py score data/labels/labels_<timestamp>.csv
```

The sheet opens in Excel or Google Sheets and can be saved in any CSV format Excel offers.
Rows are shuffled, so labeling the first 50 gives a representative sample. The
[labeling guide](data/labels/LABELING_GUIDE.md) defines both labels with worked examples.

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
   judge's verdicts and shuffles planted-error answers in with real ones, keeping versions of the
   same question at least 10 rows apart so the labeler judges each one on its own. A written
   [labeling guide](data/labels/LABELING_GUIDE.md) keeps the labels consistent. Reports accuracy,
   Cohen's kappa (agreement corrected for chance), and which way the judge errs: too lenient or
   too strict.

The judge defaults to a different, stronger model than the assistant, to avoid a model
grading its own outputs. If a judge reply isn't valid JSON, it's retried once; a reply that
still fails is recorded as a judge error, never counted as a pass or fail, and is saved in the
results file with its error message.

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
  labels/             Labeling guide and human labeling sheets
tests/                104 offline tests
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

Same assistant (claude-haiku-4-5), judge (claude-sonnet-5-5) and 32 questions; only the prompt changes.

| Metric | Prompt v1 | Prompt v2 |
|---|---|---|
| Rule pass rate | 29/32 (91%) | **31/32 (97%)** |
| Judge correctness | 30/32 | **31/32** |
| Judge faithfulness | 31/32 | **32/32** |
| Exact refusals | 1/4 | **4/4** |
| Mentions "the context" | 7/32 | **1/32** |
| Retrieval (passage) | 26/27 | 26/27 |
| Avg / p95 latency | 0.74s / 1.01s | 0.69s / 1.04s |
| Tokens in / out | 13.2k / 1.4k | **16.7k** / 1.2k |

- **v2 fixed every prompt-level failure from Run 2.** The false premise (g026) is now corrected
  without the refusal opener, the ambiguous question (g032) now asks which period is meant, and
  the hallucinated detail inside a refusal (g030) is gone.
- **The one remaining failure is retrieval, not the prompt.** g022 still refuses because the Q2
  results paragraph isn't retrieved at k=4. That answer also accounts for v2's only refusal with
  extra commentary and its only "reference material" mention.
- **The cost:** v2's longer instructions add 27% more input tokens per question.
- **Caveat:** v2 was written against failures seen in these same 32 questions, and the gap
  between the prompts is 2 questions. New, unseen questions are needed to show it generalizes.

**Run-to-run variation (same prompt v1, same questions, Run 2 vs. Run 3):**

- The judge gave **opposite verdicts on an identical answer.** g032's answer text was the same
  in both runs; the judge marked it correct in Run 2 and incorrect in Run 3, at temperature 0.
  It's the same borderline case the human calibration flagged.
- The assistant's wording differed on 3 of 32 questions (g001, g014, g031), also at temperature 0.
  The verdicts didn't change, but "temperature 0" doesn't mean "repeatable".
- → Repeat-run consistency testing moves into Phase 3, for both the assistant and the judge.

**Mutation check, rerun with the fixed scoring:**

| Mutation | Judge | Rule grader |
|---|---|---|
| Asked-for number swapped | **19/19** | 19/19 |
| Supporting figure swapped | **6/6** | 1/6 |
| False claim appended | **26/26** | 1/26 |
| Refusal replaced by fabrication | **4/4** | 4/4 |

The judge caught all 55 planted errors. Both rule-grader "catches" outside number swaps and
fabrications are g022, which already failed before mutation. Two false-claim rows (g010, g023)
are missing because the judge returned unusable replies twice in a row; those failures were
being skipped silently. → They're now saved and counted as judge errors.

### Human calibration, round 1

50 blind labels (real answers mixed with planted errors) compared with the judge's verdicts.

| Label | Agreement | Cohen's kappa | Judge too lenient | Judge too strict |
|---|---|---|---|---|
| Correct | 92% | **0.83** (strong) | 3 | 1 |
| Faithful | 84% | **0.59** (moderate) | 5 | 3 |

Every disagreement was reviewed against the written rules:

- **Faithful: the judge applied the rules correctly in all 8 disagreements.** The labels were
  inconsistent in three ways:
  - accurate extra details marked unfaithful (2 rows);
  - a made-up detail marked faithful, in one case despite a note pointing it out (3 rows);
  - "faithful" used to mean "a bad answer" when every claim was actually supported (3 rows).
- **Correct:** 3 of the 4 disagreements were label errors. In the fourth (g032, answering
  "what was revenue?" with guidance), the human's stricter call is the better one. The judge
  prompt doesn't say that a forecast isn't an actual result, and this is the same case the judge
  flip-flopped on.
- **Why:** the labeling rules lived in chat messages, and the rule for extra details was
  clarified partway through labeling, so earlier and later rows followed different standards.
- **Takeaway:** the main limit on trusting the judge was the clarity of the labeling rules, not
  the judge. Kappa is reported as measured; the disputed rows were not relabeled after the
  judge's verdicts were seen, since that would inflate agreement.

### Changes after Run 3

| Change | Why |
|---|---|
| Written [labeling guide](data/labels/LABELING_GUIDE.md) with worked examples | Inconsistent labels capped faithfulness kappa at 0.59 |
| Export keeps versions of the same question at least 10 rows apart | Adjacent versions let the labeler compare instead of judging each row |
| Judge failures saved in mutation results and counted per type | 2 mutation rows went missing without a trace |

### Known gaps

- The judge prompt doesn't state that guidance isn't an actual result; the labeling guide does.
  Until a judge-v2 prompt adds it, ambiguous questions like g032 will keep disagreeing.
- Ambiguous questions have no `evidence` strings, so their retrieval isn't measured. g032's
  answers say 2025 revenue isn't in the documents because that paragraph was never retrieved.
- 32 questions is small: one question is about 3 percentage points.
