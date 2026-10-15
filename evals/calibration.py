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

Spacing: one question can appear up to 3 times (the real answer plus planted
versions). If two versions sit side by side, the labeler compares them instead
of judging each one against its passages, which makes planted errors easier
to spot than they are for the judge. The export keeps versions of the same
question at least MIN_GAP rows apart. (Round 1 used a plain shuffle and had
versions of the same question in adjacent rows.)

How to label: see data/labels/LABELING_GUIDE.md.

Re-judging: human labels don't depend on the judge, so one labeled sheet can
score any number of judge versions. rejudge_label_rows() runs a new judge over
the same rows and writes a second answer key; scoring both keys against the
same labels gives a fair before/after for a judge prompt change.
"""
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from evals.judge import JudgeError

# Excel adds and expects a hidden "byte order mark" (BOM) at the start of
# UTF-8 CSV files. Writing with "utf-8-sig" adds it, so Excel shows special
# characters correctly.
WRITE_ENCODING = "utf-8-sig"

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


MIN_GAP = 10   # rows between versions of the same question


def _one_ordering(groups: dict, rng: random.Random, min_gap: int, total: int) -> list[dict]:
    """One attempt: build the order a row at a time, avoiding recent questions."""
    groups = {g: list(members) for g, members in groups.items()}
    for members in groups.values():
        rng.shuffle(members)              # random order of versions within a question
    names = sorted(groups)                # sorted so the result depends only on the seed
    order, last_used = [], {}
    while len(order) < total:
        left = [g for g in names if groups[g]]
        allowed = [g for g in left if len(order) - last_used.get(g, -min_gap) >= min_gap]
        if allowed:
            # Favor questions with more rows left, so they don't pile up at the end.
            weights = [len(groups[g]) ** 2 for g in allowed]
            pick = rng.choices(allowed, weights=weights)[0]
        else:
            pick = min(left, key=lambda g: last_used.get(g, -min_gap))
        order.append(groups[pick].pop())
        last_used[pick] = len(order) - 1
    return order


def spread_out(rows: list[dict], rng: random.Random, min_gap: int = MIN_GAP,
               group_key: str = "id", attempts: int = 200) -> list[dict]:
    """Shuffle rows so rows sharing a question are at least min_gap rows apart.

    Each attempt picks rows one at a time, choosing a random question that
    hasn't appeared in the last min_gap rows. An attempt can get cornered
    near the end, when only a few questions are left, so it tries again with
    a fresh random order. It returns the first attempt that keeps the gap, or
    the best attempt if none does (possible with very few questions).
    """
    groups = defaultdict(list)
    for row in rows:
        groups[row[group_key]].append(row)

    best, best_gap = list(rows), -1
    for _ in range(attempts):
        order = _one_ordering(groups, rng, min_gap, len(rows))
        gap = min_distance(order, group_key)
        if gap is None or gap >= min_gap:
            return order
        if gap > best_gap:
            best, best_gap = order, gap
    return best


def min_distance(rows: list[dict], group_key: str = "id") -> int | None:
    """Smallest row distance between two rows of the same question (None if no repeats)."""
    last, best = {}, None
    for i, row in enumerate(rows):
        g = row[group_key]
        if g in last:
            gap = i - last[g]
            best = gap if best is None else min(best, gap)
        last[g] = i
    return best


def export_for_labeling(rows: list[dict], csv_path: Path, seed: int = 7) -> Path:
    """Write a blind labeling sheet plus a separate answer key.

    rows: result rows that have judge verdicts (from run_evals or the
    mutation check). Returns the path of the answer-key file.
    """
    usable = [r for r in rows if r.get("judge_correct") is not None]
    usable = spread_out(usable, random.Random(seed))   # fixed seed = same order every time

    key = {}
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding=WRITE_ENCODING) as f:
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


def read_label_rows(csv_path: Path) -> list[dict]:
    """Read the labeling sheet, whichever way Excel saved it.

    Excel's "CSV UTF-8" adds a BOM; without handling it, the first column is
    read as "\ufefflabel_id" and every lookup fails. "utf-8-sig" strips it
    (and works without one too). Excel's plain "CSV (Comma delimited)" saves
    in the Windows-1252 encoding instead, so that's the fallback.
    """
    raw = Path(csv_path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")
    rows = list(csv.DictReader(text.splitlines(keepends=True)))
    if rows and "label_id" not in rows[0]:
        raise ValueError(f"no label_id column in {csv_path}; was the header row changed?")
    return rows


PASSAGE_SEPARATOR = "\n---\n"   # how export_for_labeling joins passages in one cell


def split_passages(cell: str) -> list[str]:
    """Undo the join from export_for_labeling. Excel may save Windows line
    endings (\r\n), so those are normalized first."""
    text = (cell or "").replace("\r\n", "\n")
    return [p for p in text.split(PASSAGE_SEPARATOR) if p.strip()]


def rejudge_label_rows(csv_path: Path, judge_fn, key_path: Path | None = None,
                       out_key_path: Path | None = None, **judge_kwargs) -> Path:
    """Grade every row of a labeling sheet again with judge_fn; write a new key.

    The new key keeps each row's case_id and mutation from the original key,
    so the two keys line up row for row. Rows with no human label are
    skipped: grading them costs API calls and can't change any score.
    A judge failure is saved as
    judge_error with null verdicts; score_labels() skips those rows.
    judge_kwargs (model, prompt_version, ...) are passed to judge_fn.
    """
    csv_path = Path(csv_path)
    key_path = Path(key_path) if key_path else csv_path.with_suffix(".key.json")
    old_key = json.loads(key_path.read_text(encoding="utf-8"))
    new_key = {}
    for row in read_label_rows(csv_path):
        label_id = row["label_id"].strip()
        if not label_id:
            continue
        if not (row.get("human_correct", "").strip() or row.get("human_faithful", "").strip()):
            continue
        entry = {"case_id": old_key[label_id]["case_id"],
                 "mutation": old_key[label_id].get("mutation")}
        try:
            verdict = judge_fn(row["question"], split_passages(row["passages"]),
                               row["reference_answer"], row["answer"], **judge_kwargs)
            entry.update({"judge_correct": verdict.correct, "judge_faithful": verdict.faithful,
                          "judge_reasoning": verdict.reasoning, "judge_error": None})
        except JudgeError as exc:
            entry.update({"judge_correct": None, "judge_faithful": None,
                          "judge_reasoning": None, "judge_error": str(exc)})
        new_key[label_id] = entry

    out = Path(out_key_path) if out_key_path else csv_path.with_suffix(".rejudged.key.json")
    out.write_text(json.dumps(new_key, indent=2), encoding="utf-8")
    return out


def changed_verdicts(old_key: dict, new_key: dict) -> list[dict]:
    """Rows where the two judges disagree on either verdict."""
    changes = []
    for label_id, new in new_key.items():
        old = old_key.get(label_id, {})
        for criterion in ("correct", "faithful"):
            before, after = old.get(f"judge_{criterion}"), new.get(f"judge_{criterion}")
            if before is not None and after is not None and before != after:
                changes.append({"label_id": label_id, "case_id": new.get("case_id"),
                                "criterion": criterion, "before": before, "after": after})
    return changes


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
    for row in read_label_rows(csv_path):
        label_id = row["label_id"].strip()
        if not label_id:          # Excel sometimes keeps empty trailing rows
            continue
        judged = key[label_id]
        for criterion in pairs:
            if judged.get(f"judge_{criterion}") is None:   # judge failed on this row
                continue
            try:
                human = parse_label(row.get(f"human_{criterion}", ""))
            except ValueError as exc:
                raise ValueError(f"{label_id} human_{criterion}: {exc}") from exc
            if human is not None:
                pairs[criterion].append((label_id, judged[f"judge_{criterion}"], human))

    return {criterion: agreement(p) for criterion, p in pairs.items()}
