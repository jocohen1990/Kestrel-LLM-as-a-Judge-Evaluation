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

How the verdict comes back (output_mode):
  * "tool" (default): the judge must call a submit_verdict tool whose input
    schema IS the verdict format, with strict mode on. The API guarantees the
    shape, so there's no JSON text to parse. Added after Run 4, where 4 of 160
    free-text replies were broken JSON (all on the arithmetic questions g023
    and g024, and each failed its retry too).
  * "text": the original approach. The prompt asks for "ONLY a JSON object"
    and parse_judge_output() digs it out of the reply. Kept so the two modes
    can be compared on the same answers.

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

from app.llm import StructuredOutputError, generate, generate_structured

DEFAULT_JUDGE_MODEL = "claude-sonnet-5-5"

# Like the assistant prompts in app/rag.py, judge prompts are versioned and
# never edited in place. A results file records which version graded it, so a
# score change can be traced to the judge rather than the model.
_STEPS = """You are a strict evaluator of answers produced by a document Q&A assistant.
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
    asked which period was meant or clearly stated which period it answered for."""

_REPLY_FORMAT = """

Respond with ONLY a JSON object, no other text:
{"claims": [{"claim": "...", "supported": true}],
 "faithful": true,
 "correct": true,
 "reasoning": "one or two sentences"}"""

# How the judge should hand back its verdict in "tool" mode. The schema
# below defines the fields, so the prompt only has to name the tool.
_TOOL_FORMAT = """

Record your claims and verdicts by calling the submit_verdict tool."""

# The grading rules, by version. The reply-format paragraph is added by
# judge_system_prompt(), depending on the output mode.
JUDGE_RULES = {
    # judge-v1: used for Runs 2 and 3, the mutation checks, and human
    # calibration round 1.
    "judge-v1": _STEPS,
    # judge-v2: adds one rule, from calibration round 1. Asked "What was
    # Kestrel's revenue?" (g032), the assistant answered with GUIDANCE, a
    # forecast. The human labeled it incorrect; the judge called it correct in
    # Run 2 and incorrect in Run 3 on the identical answer. v1 never said a
    # forecast isn't an actual result, so the judge had to guess. The labeling
    # guide already had this rule; now the judge has it too.
    "judge-v2": _STEPS + """

Rule for forecasts. Guidance, forecasts, outlooks and targets are NOT actual results.
When the question asks for an actual result (what revenue, earnings or margin was),
an answer whose only figure for it is guidance or a forecast is NOT correct, even if
it labels the figure as guidance. For an ambiguous question, correct means the
assistant asked which period was meant, or named a period and gave that period's
ACTUAL figure. When the question itself asks for guidance or what the company
expected, guidance is the right answer.""",
}

DEFAULT_JUDGE_PROMPT = "judge-v2"

OUTPUT_MODES = ("tool", "text")
DEFAULT_OUTPUT_MODE = "tool"

TOOL_NAME = "submit_verdict"
TOOL_DESCRIPTION = "Submit the claim-by-claim check and the two verdicts for the assistant's answer."

# The verdict format as a JSON schema. Strict mode requires
# additionalProperties: false on every object and every field listed in
# required. Field order matches the text format: claims first, so the judge
# checks each claim before deciding.
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "supported": {"type": "boolean"},
                },
                "required": ["claim", "supported"],
                "additionalProperties": False,
            },
        },
        "faithful": {"type": "boolean"},
        "correct": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
    "required": ["claims", "faithful", "correct", "reasoning"],
    "additionalProperties": False,
}


def judge_system_prompt(prompt_version: str = DEFAULT_JUDGE_PROMPT,
                        output_mode: str = DEFAULT_OUTPUT_MODE) -> str:
    """The full judge instructions: grading rules + how to return the verdict."""
    if prompt_version not in JUDGE_RULES:
        raise ValueError(f"unknown judge prompt {prompt_version!r}; "
                         f"choose from {sorted(JUDGE_RULES)}")
    if output_mode not in OUTPUT_MODES:
        raise ValueError(f"unknown output mode {output_mode!r}; choose from {OUTPUT_MODES}")
    return JUDGE_RULES[prompt_version] + (_TOOL_FORMAT if output_mode == "tool" else _REPLY_FORMAT)


class JudgeError(ValueError):
    """The judge's reply couldn't be turned into a verdict."""


@dataclass
class Verdict:
    faithful: bool
    correct: bool
    reasoning: str
    claims: list[dict] = field(default_factory=list)
    # Token usage of the call that produced this verdict (0 when unknown,
    # e.g. a fake judge in tests). input_tokens includes cached tokens.
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0

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


def verdict_from_data(data: dict) -> Verdict:
    """Check the verdict fields and build a Verdict. Used by both output modes.

    Strict tool mode already guarantees these types, but checking costs
    nothing and keeps a misconfigured schema from passing silently.
    """
    if not isinstance(data, dict):
        raise JudgeError("judge verdict is not a JSON object")
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


def parse_judge_output(text: str) -> Verdict:
    """Turn a free-text judge reply ("text" mode) into a Verdict.

    Models sometimes wrap JSON in ```json fences or add a sentence before it,
    so we pull out the {...} block instead of trusting the whole reply.
    Anything malformed raises JudgeError rather than silently counting as a
    pass or fail; a broken grader must never look like a real result. The
    error keeps the WHOLE reply: in Run 4 it kept only the last 120
    characters, which wasn't enough to see what went wrong.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise JudgeError(f"no JSON object in judge reply: {text!r}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise JudgeError(f"invalid JSON from judge: {exc}; full reply: {text!r}") from exc
    return verdict_from_data(data)


def _with_usage(verdict: Verdict, response) -> Verdict:
    verdict.input_tokens = response.input_tokens
    verdict.cached_input_tokens = response.cache_read_input_tokens
    verdict.output_tokens = response.output_tokens
    return verdict


def _ask_once(system: str, prompt: str, model: str, output_mode: str) -> Verdict:
    """One judge call. The system prompt (and in tool mode, the tool) is the
    same for every answer, so it's marked as a cacheable prefix: after the
    first call, later calls within a few minutes read it from the cache at a
    fraction of the input price. Verdicts are unaffected; caching changes the
    bill, not the model's input."""
    if output_mode == "tool":
        try:
            response = generate_structured(system, prompt, VERDICT_SCHEMA, TOOL_NAME,
                                           TOOL_DESCRIPTION, model=model, max_tokens=1200,
                                           temperature=0.0, cache_system=True)
        except StructuredOutputError as exc:
            raise JudgeError(str(exc)) from exc
        return _with_usage(verdict_from_data(response.data), response)
    response = generate(system, prompt, model=model, max_tokens=1200, temperature=0.0,
                        cache_system=True)
    return _with_usage(parse_judge_output(response.text), response)


def judge_answer(question: str, contexts: list[str], reference: str, answer: str,
                 model: str | None = None, attempts: int = 2,
                 prompt_version: str = DEFAULT_JUDGE_PROMPT,
                 output_mode: str = DEFAULT_OUTPUT_MODE) -> Verdict:
    """Ask the judge model for a verdict on one answer.

    If no usable verdict comes back, ask again (once, by default). If every
    attempt fails, the error is raised and recorded, never counted as a pass
    or fail.

    prompt_version picks the grading rules (see JUDGE_RULES), so two versions
    can grade the same answers and be compared. output_mode picks how the
    verdict comes back (see the top of this file).
    """
    system = judge_system_prompt(prompt_version, output_mode)   # validates both settings
    model = model or os.getenv("JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    prompt = build_judge_prompt(question, contexts, reference, answer)
    last_error = None
    for _ in range(attempts):
        try:
            return _ask_once(system, prompt, model, output_mode)
        except JudgeError as exc:
            last_error = exc
    raise JudgeError(f"no valid verdict after {attempts} attempts; last error: {last_error}")
