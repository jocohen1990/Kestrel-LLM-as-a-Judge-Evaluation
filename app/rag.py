"""The system under test: a retrieval-augmented (RAG) Q&A assistant.

Flow:  question -> retriever finds passages -> passages + question go to the
LLM -> LLM answers using only those passages.

Prompts are versioned. Each version is kept (never edited in place), so any
two can be compared on the same dataset:

    python run_evals.py --prompt v1
    python run_evals.py --prompt v2
"""
from dataclasses import dataclass

from app.llm import LLMResponse, generate
from app.paths import CORPUS_DIR
from app.retriever import Retriever, load_chunks

REFUSAL = "I don't know based on the provided documents."

PROMPTS = {
    # v1: the original prompt. Run 2 showed three problems with it:
    #   * the refusal sentence was used as an opener, even before real
    #     answers (6 of 32 answers started with it; 1 of 4 refusals was exact)
    #   * answers said "according to the context" (7 of 32)
    #   * no guidance for false premises or ambiguous questions
    "v1": {
        "system": f"""You are a financial research assistant for Kestrel Freight Analytics.
Answer the user's question using ONLY the context passages provided.

Rules:
- If the answer is not in the context, reply exactly: "{REFUSAL}"
- Do not guess or use outside knowledge.
- Keep answers to one or two sentences.
- Include the specific figures from the context when relevant.
""",
        "header": "Context passages:",
        "show_file_names": True,
    },
    # v2: one rule per run 2 failure. The material is labeled "reference
    # material" with document titles instead of file names, so the model has
    # no "context" or "passage [2]" wording to repeat back to the user.
    "v2": {
        "system": f"""You are a financial research assistant answering questions about Kestrel Freight Analytics.
Answer using ONLY the reference material provided with each question. Do not use outside knowledge or guess.

How to answer:
1. If the reference material answers the question, answer it in one or two sentences with the specific figures.
2. If the question assumes something the material contradicts (for example, it asks why revenue fell when it rose), say the premise is incorrect and give the correct figures.
3. If the question doesn't say which period it means, either name the period you are answering for or give the figure for each period in the material.
4. If the material does not contain the answer, reply with exactly this sentence and nothing else: "{REFUSAL}"

Rules:
- Use the sentence in step 4 only when you are not answering. Never put it before an answer.
- Only state facts that appear in the reference material. Don't add facts that weren't asked for.
- The user cannot see the reference material. Never mention "context", "passages", "the material" or passage numbers. You may name a document, such as "the Q2 2026 earnings release".
""",
        "header": "Reference material:",
        "show_file_names": False,
    },
}

# The prompt used when none is specified. A run's results file records which
# version produced it, so a score change can always be traced to a prompt.
DEFAULT_PROMPT_VERSION = "v2"


@dataclass
class RagAnswer:
    question: str
    answer: str
    sources: list[str]    # which files the retrieved passages came from
    contexts: list[str]   # the passages themselves (the judge grades against these)
    llm: LLMResponse


class RagAssistant:
    def __init__(self, corpus_dir=CORPUS_DIR, k: int = 4, model: str | None = None,
                 prompt_version: str = DEFAULT_PROMPT_VERSION):
        if prompt_version not in PROMPTS:
            raise ValueError(f"unknown prompt version {prompt_version!r}; "
                             f"choose from {sorted(PROMPTS)}")
        self.retriever = Retriever(load_chunks(corpus_dir))
        self.k = k
        self.model = model
        self.prompt_version = prompt_version
        self.prompt = PROMPTS[prompt_version]

    def build_prompt(self, question: str, chunks) -> str:
        """Lay out the retrieved passages, then append the question."""
        if self.prompt["show_file_names"]:
            parts = [f"[{i}] (source: {c.doc})\n{c.text}" for i, c in enumerate(chunks, start=1)]
        else:
            # Each chunk already starts with its document title.
            parts = [c.text for c in chunks]
        material = "\n\n".join(parts)
        return f"{self.prompt['header']}\n{material}\n\nQuestion: {question}"

    def answer(self, question: str) -> RagAnswer:
        chunks = self.retriever.search(question, k=self.k)
        response = generate(self.prompt["system"], self.build_prompt(question, chunks),
                            model=self.model)
        return RagAnswer(
            question=question,
            answer=response.text.strip(),
            sources=sorted({c.doc for c in chunks}),
            contexts=[c.text for c in chunks],
            llm=response,
        )
