# Labeling guide

How to fill in `human_correct` and `human_faithful` in a labeling sheet from
`python calibrate.py export`. Read this before every labeling round, and keep it
open while you label.

Why it exists: in round 1, agreement with the judge on faithfulness was 84%
(Cohen's kappa 0.59). Reviewing the 8 disagreements showed the judge had applied
the rules correctly in all of them. The labels were inconsistent, mostly because
the two columns got mixed up and the rule for extra details changed partway
through. Written rules applied the same way on every row fix that.

## The two columns check different things

| Column | Compare the answer against | Question to ask |
|---|---|---|
| `human_correct` | `reference_answer` | Did it get the thing that was asked right? |
| `human_faithful` | `passages` | Is every claim backed by the passages? |

Keep them separate. An answer can be correct but unfaithful (right answer plus one
made-up fact), or faithful but incorrect (refuses because the passages are missing
the answer).

## Step 1: `human_faithful`

1. Read the answer **to the end** and split it into individual claims. Planted
   errors are often a single sentence added at the end.
2. Check each claim against the `passages` cell only. Ignore what you know is in
   the full documents: the model only saw these passages.
3. Every claim supported → `y`. Even one unsupported claim → `n`.

Things that count as claims:
- **Extra details**, even ones nobody asked for. Supported extra detail is fine (`y`);
  unsupported extra detail is not (`n`).
- **Where a fact came from.** "According to the Q2 earnings release…" is a claim. If
  the passage containing that fact is from the Q1 release, it's unsupported.
- **Statements about the documents.** "The documents don't give Q3 revenue" is
  supported if the passages indeed don't. "The documents include Q1 guidance" is
  unsupported if they don't.
- **Arithmetic** is supported when it follows from figures in the passages.

Things that do **not** make an answer unfaithful:
- It's wordy, awkward, badly ordered, or answers the wrong question. Those are
  problems with correctness or style, not faithfulness.
- A plain refusal ("I don't know based on the provided documents.") makes no claims → `y`.

## Step 2: `human_correct`

1. Decide what the question actually asked.
2. Compare the answer's response to *that* with the `reference_answer`. Same
   meaning → `y`. Extra correct detail doesn't change this.
3. Special cases:

| The reference answer says | `y` when the answer… |
|---|---|
| A figure or fact | gives that figure or fact |
| "Refusal: not in the documents" | refuses, with or without a short explanation |
| "False premise" | says the premise is wrong and gives the correct facts |
| "Ambiguous" | asks which period is meant, **or** names a period and gives that period's actual figure. A forecast or guidance is not an actual result, so answering "what was revenue?" with guidance is `n`. |

An extra claim only makes `correct` = `n` if it changes or contradicts the answer to
the question asked (for example, "Kestrel doesn't pay a dividend. The board approved
a $0.12 dividend.").

## Worked examples

These use an invented company, Harlow Ridge Logistics, so they don't give away any
answers in the real sheet.

**1. Supported extra detail**
- Question: What was Harlow Ridge's Q3 revenue?
- Passages: "…Q3 revenue was $52.0 million, up 8% from $48.1 million a year earlier."
- Reference: $52.0 million
- Answer: "Q3 revenue was $52.0 million, up 8% year over year."
- **correct `y`, faithful `y`.** The growth rate wasn't asked for, but it's in the passages.

**2. Unsupported extra detail**
- Same question and passages.
- Answer: "Q3 revenue was $52.0 million. The company also opened a warehouse in Tulsa."
- **correct `y`, faithful `n`.** The revenue is right; the warehouse is in no passage.

**3. Refusing when the passages are missing the answer**
- Question: What was combined revenue for Q3 and Q4?
- Passages contain Q3 revenue and the Q4 *forecast*, but not actual Q4 revenue.
- Reference: $107.5 million
- Answer: "I don't know based on the provided documents."
- **correct `n`, faithful `y`.** It didn't give the expected answer, but it made no
  false claims. The real problem is retrieval, which the passage-level metric tracks.

**4. Wrong source**
- Question: When did Harlow Ridge open its Denver hub?
- Passages: "[Q1 2026 Earnings Release] …opened its Denver hub in March 2026."
- Answer: "According to the Q2 2026 earnings release, the Denver hub opened in March 2026."
- **correct `y`, faithful `n`.** The date is right; the attribution is not.

**5. A good refusal with a bad explanation**
- Question: What was Q4 revenue? (Not in the documents.)
- Reference: Refusal
- Answer: "I don't know based on the provided documents. The passages give Q1 guidance
  but not Q4 results." (The passages contain no Q1 guidance.)
- **correct `y`, faithful `n`.** It refused, as it should, but its explanation
  contains a false claim.

**6. A false premise corrected, but awkwardly**
- Question: Why did revenue fall in Q3?
- Reference: False premise; revenue rose 8%.
- Answer: "I don't know based on the provided documents. Revenue actually rose 8% in Q3."
- **correct `y`, faithful `y`.** It corrects the premise, and every claim is supported.
  The contradictory opening is a style problem: put it in `notes`, not in either label.

**7. Ambiguous question answered with a forecast**
- Question: What was Harlow Ridge's revenue?
- Reference: Ambiguous; should name the period or ask which one.
- Answer: "Full-year 2026 revenue is expected to be $210–$220 million."
- **correct `n`, faithful `y`.** A forecast isn't revenue, and every claim is supported.

## Habits that keep labels consistent

- **Judge each row on its own.** Don't compare it with other versions of the same
  question. The export keeps them at least 10 rows apart, but you may still remember
  an earlier one.
- **Use `notes` for anything that isn't correctness or faithfulness**, such as style,
  awkward wording or unnecessary detail, and for any row where you hesitated.
- **Don't change the rules midway.** If a new kind of case comes up, write down how you
  decided it and apply that from then on. Add it to this guide afterward.
- **Don't open the `.key.json` file** until every row you plan to label is done.
- **Label in one or two sittings**, so your standard doesn't drift.
