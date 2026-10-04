"""Judge calibration: how often does the judge agree with a human?

An LLM judge is itself a model, so it can be wrong. Before trusting its
scores you measure it against human labels, the same way you'd validate a
new test tool against known results before relying on it.

Two numbers matter:
  * Accuracy: the share of cases where judge and human agree. Easy to
    read, but misleading when almost everything passes: a judge that
    ALWAYS says "pass" scores 95% on a set that's 95% passes.
  * Cohen's kappa: agreement AFTER removing what you'd expect by chance.
    1.0 = perfect, 0 = no better than chance. Rough guide:
    above 0.8 strong, 0.6 to 0.8 substantial, below 0.6 don't trust the judge yet.

Blind labeling: the export file deliberately leaves out the judge's
verdicts, and mixes planted-error answers in with real ones in random order.
If you could see the judge's opinion while labeling, you'd be nudged toward
agreeing with it (anchoring bias), and the agreement score would be inflated.
"""
import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

LABEL_COLUMNS = [
    "label_id", "question", "passages", "reference_answer", "answer",
    "human_correct", "human_faithful", "notes",
]

TRUE_WORDS = {"y", "yes", "true", "1", "pass", "p"}
FALSE_WORDS = {"n", "no", "false", "0", "fail", "f"}


def parse_label(text: str) -> bool | None:
    """Read a human's label. Blank means "not labeled yet" and is skipped."""
    word = (text or "").strip().lower()
    if not word:
        return None
    if word in TRUE_WORDS:
        return True
    if word in FALSE_WORDS:
        return False
    raise ValueError(f"can't read label {text!r}; use y or n")


def export_for_labeling(rows: list[dict], csv_path: Path, seed: int = 7) -> Path:
    """Write a blind labeling sheet plus a separate answer key.

    rows: result rows that have judge verdicts (from run_evals or the
    mutation check). Returns the path of the answer-key file.
    """
    usable = [r for r in rows if r.get("judge_correct") is not None]
    random.Random(seed).shuffle(usable)   # fixed seed = same order every time

    key = {}
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        for i, row in enumerate(usable, start=1):
            label_id = f"L{i:03d}"   # opaque id: doesn't reveal case or mutation
            writer.writerow({
                "label_id": label_id,
                "question": row["question"],
                "passages": "\n---\n".join(row.get("contexts", [])),
                "reference_answer": row["expected_answer"],
                "answer": row["answer"],
                "human_correct": "",
                "human_faithful": "",
                "notes": "",
            })
            key[label_id] = {
                "case_id": row["id"],
                "mutation": row.get("mutation"),
                "judge_correct": row["judge_correct"],
                "judge_faithful": row["judge_faithful"],
                "judge_reasoning": row.get("judge_reasoning"),
            }

    key_path = csv_path.with_suffix(".key.json")
    key_path.write_text(json.dumps(key, indent=2), encoding="utf-8")
    return key_path


@dataclass
class Agreement:
    n: int
    accuracy: float | None
    kappa: float | None
    # Confusion matrix, treating "pass" (True) as the positive class.
    both_pass: int = 0
    both_fail: int = 0
    judge_pass_human_fail: int = 0    # judge too lenient: missed a real problem
    judge_fail_human_pass: int = 0    # judge too strict: false alarm
    disagreements: list[str] = field(default_factory=list)


def cohen_kappa(judge: list[bool], human: list[bool]) -> float | None:
    """Agreement corrected for chance. None when it's mathematically undefined
    (both raters gave the same single label to every case)."""
    n = len(judge)
    if n == 0:
        return None
    observed = sum(j == h for j, h in zip(judge, human)) / n
    judge_yes, human_yes = sum(judge) / n, sum(human) / n
    expected = judge_yes * human_yes + (1 - judge_yes) * (1 - human_yes)
    if expected == 1:
        return None
    return round((observed - expected) / (1 - expected), 3)


def agreement(pairs: list[tuple[str, bool, bool]]) -> Agreement:
    """pairs: (label_id, judge_verdict, human_verdict) for one criterion."""
    judge = [j for _, j, _ in pairs]
    human = [h for _, _, h in pairs]
    result = Agreement(
        n=len(pairs),
        accuracy=round(sum(j == h for j, h in zip(judge, human)) / len(pairs), 3) if pairs else None,
        kappa=cohen_kappa(judge, human),
    )
    for label_id, j, h in pairs:
        if j and h:
            result.both_pass += 1
        elif not j and not h:
            result.both_fail += 1
        elif j and not h:
            result.judge_pass_human_fail += 1
            result.disagreements.append(label_id)
        else:
            result.judge_fail_human_pass += 1
            result.disagreements.append(label_id)
    return result


def score_labels(csv_path: Path, key_path: Path | None = None) -> dict:
    """Compare the human labels in the CSV against the judge's verdicts."""
    csv_path = Path(csv_path)
    key_path = Path(key_path) if key_path else csv_path.with_suffix(".key.json")
    key = json.loads(key_path.read_text(encoding="utf-8"))

    pairs = {"correct": [], "faithful": []}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            judged = key[row["label_id"]]
            for criterion in pairs:
                human = parse_label(row[f"human_{criterion}"])
                if human is not None:
                    pairs[criterion].append((row["label_id"], judged[f"judge_{criterion}"], human))

    return {criterion: agreement(p) for criterion, p in pairs.items()}
