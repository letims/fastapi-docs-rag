import re
import time
from dataclasses import dataclass, field

from app.providers.base import LLM
from app.rag.retrieve import Hit

NO_ANSWER = "INSUFFICIENT_CONTEXT"

SYSTEM = f"""You answer questions about FastAPI using ONLY the numbered context passages.
Rules:
- Cite every claim with the passage number in square brackets, e.g. [2].
- Never use outside knowledge, even if you are sure of it.
- If the passages do not contain the answer, reply with exactly: {NO_ANSWER}
  followed by one sentence saying what information is missing.
- If the passages answer only part of the question, answer that part, then add a line starting
  with "Not found:" listing what is missing."""


def build_prompt(question: str, hits: list[Hit]) -> str:
    """Format retrieved chunks and the question into the user prompt.

    Args:
        question: The user's question.
        hits: Retrieved chunks, in rank order. Each is numbered from 1; those numbers are what
            the LLM cites as `[n]`.

    Returns:
        The prompt text: numbered context passages (with source and heading) then the question.
    """
    ctx = "\n\n".join(f"[{i}] ({h.source} :: {h.heading})\n{h.content}" for i, h in enumerate(hits, 1))
    return f"Context passages:\n\n{ctx}\n\nQuestion: {question}"


@dataclass
class Answer:
    """The result of answering one question, with enough detail to evaluate and trace it."""

    question: str
    text: str
    abstained: bool
    citations: list[Hit]  # only passages actually cited in the answer
    retrieved: list[Hit]
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    trace: list[dict] = field(default_factory=list)


async def generate_answer(llm: LLM, question: str, hits: list[Hit]) -> Answer:
    """Ask the LLM to answer from the retrieved chunks only, and parse its citations.

    The system prompt tells the model to cite passages as `[n]` and to reply
    `INSUFFICIENT_CONTEXT` when the passages do not contain the answer. Citation numbers outside
    1..len(hits) are ignored, so a hallucinated `[9]` never produces a citation.

    Args:
        llm: Provider used to generate the answer.
        question: The user's question.
        hits: Retrieved chunks to use as context.

    Returns:
        An `Answer`. `abstained` is true if the text starts with `INSUFFICIENT_CONTEXT`;
        `citations` holds only the chunks actually cited; `latency_s` covers the LLM call only.
    """
    t0 = time.perf_counter()
    res = await llm.generate(build_prompt(question, hits), system=SYSTEM)
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", res.text) if 1 <= int(n) <= len(hits)})
    return Answer(
        question=question,
        text=res.text.strip(),
        abstained=res.text.strip().startswith(NO_ANSWER),
        citations=[hits[i - 1] for i in cited],
        retrieved=hits,
        input_tokens=res.input_tokens,
        output_tokens=res.output_tokens,
        latency_s=time.perf_counter() - t0,
    )
