"""LLM-as-judge: a second model grades the assistant's answers.

Why a judge? The rule-based graders in metrics.py check that the expected
number appears. They can't tell whether the REST of the answer is true. In
run 1, g001 added "an 18% increase from $349.7 million". That happened to be
correct, but "a 25% increase" would have passed too. The judge reads the
retrieved passages and checks every claim.

Two verdicts per answer, both pass/fail:
  * faithful: every factual claim is supported by the retrieved passages
              (catches hallucination, even when the main answer is right)
  * correct:  the answer matches the reference answer in meaning
              (catches wrong answers phrased in ways the rules can't parse)

Pass/fail beats a 1-10 score here because (a) judges are more consistent at
binary decisions and (b) binary labels make calibration simple: you can
directly count how often the judge agrees with a human.

Known judge weaknesses (worth stating in your write-up):
  * Self-preference: a model may grade its own outputs leniently, so the
    judge defaults to a DIFFERENT, stronger model than the assistant.
  * The judge can be wrong. That's why calibrate.py measures it against
    human labels, and mutations.py checks it catches planted errors.
"""
import json
import os
import re
from dataclasses import dataclass, field

from app.llm import generate

DEFAULT_JUDGE_MODEL = "claude-sonnet-5-5"
JUDGE_PROMPT_VERSION = "judge-v1"

JUDGE_SYSTEM = """You are a strict evaluator of answers produced by a document Q&A assistant.
You will receive a question, the passages the assistant was given, a reference answer
written by a human, and the assistant's answer.

Step 1. Break the assistant's answer into individual factual claims.
Step 2. For each claim, decide whether the passages support it. A claim is supported
        only if the passages state it or it follows from simple arithmetic on figures
        in the passages. General knowledge does NOT count as support.
Step 3. Decide two verdicts:
  - faithful: true only if EVERY claim is supported. A refusal ("I don't know...")
    makes no factual claims and is faithful.
  - correct: true if the answer means the same as the reference answer. Extra
    supported detail is fine. If the reference says the question should be refused,
    correct means the assistant refused. If the reference says the question has a
    false premise, correct means the assistant corrected the premise. If the
    reference says the question is ambiguous, correct means the assistant either
    asked which period was meant or clearly stated which period it answered for.

Respond with ONLY a JSON object, no other text:
{"claims": [{"claim": "...", "supported": true}],
 "faithful": true,
 "correct": true,
 "reasoning": "one or two sentences"}"""


class JudgeError(ValueError):
    """The judge's reply couldn't be turned into a verdict."""


@dataclass
class Verdict:
    faithful: bool
    correct: bool
    reasoning: str
    claims: list[dict] = field(default_factory=list)

    @property
    def unsupported_claims(self) -> list[str]:
        return [c.get("claim", "") for c in self.claims if not c.get("supported", False)]


def build_judge_prompt(question: str, contexts: list[str], reference: str, answer: str) -> str:
    passages = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(contexts, start=1))
    return (
        f"QUESTION:\n{question}\n\n"
        f"PASSAGES GIVEN TO THE ASSISTANT:\n{passages}\n\n"
        f"REFERENCE ANSWER:\n{reference}\n\n"
        f"ASSISTANT'S ANSWER:\n{answer}"
    )


def parse_judge_output(text: str) -> Verdict:
    """Turn the judge's reply into a Verdict.

    Models sometimes wrap JSON in ```json fences or add a sentence before it,
    so we pull out the first {...} block instead of trusting the whole reply.
    Anything malformed raises JudgeError rather than silently counting as a
    pass or fail; a broken grader must never look like a real result.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise JudgeError(f"no JSON object in judge reply: {text[:200]!r}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise JudgeError(f"invalid JSON from judge: {exc}; reply ends with {text[-120:]!r}") from exc

    for key in ("faithful", "correct"):
        if not isinstance(data.get(key), bool):
            raise JudgeError(f"judge reply is missing a true/false {key!r}")
    claims = data.get("claims", [])
    if not isinstance(claims, list):
        raise JudgeError("'claims' must be a list")

    return Verdict(
        faithful=data["faithful"],
        correct=data["correct"],
        reasoning=str(data.get("reasoning", "")),
        claims=claims,
    )


def judge_answer(question: str, contexts: list[str], reference: str, answer: str,
                 model: str | None = None, attempts: int = 2) -> Verdict:
    """Ask the judge model for a verdict on one answer.

    If the reply can't be parsed, ask again (once, by default). In the first
    mutation run, 1 of 58 judge replies was broken JSON; a single retry is
    the usual fix for an occasional formatting slip. If every attempt fails,
    the error is raised and recorded, never counted as a pass or fail.
    """
    model = model or os.getenv("JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    prompt = build_judge_prompt(question, contexts, reference, answer)
    last_error = None
    for _ in range(attempts):
        response = generate(JUDGE_SYSTEM, prompt, model=model, max_tokens=1200, temperature=0.0)
        try:
            return parse_judge_output(response.text)
        except JudgeError as exc:
            last_error = exc
    raise JudgeError(f"no valid verdict after {attempts} attempts; last error: {last_error}")
