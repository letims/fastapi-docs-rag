import re
import time
from dataclasses import dataclass, field

from app.providers.base import LLM
from app.rag.retrieve import Hit

NO_ANSWER = "INSUFFICIENT_CONTEXT"

# Models do not always use the exact `[2]` form we ask for. Accept the common variants so a
# correct citation is not scored as missing: full-width brackets `【2】`, lists `[2, 5]` and `[2,5]`.
_CITATION = re.compile(r"[\[【]\s*(\d+(?:\s*[,，、]\s*\d+)*)\s*[\]】]")


def parse_citations(text: str, n_hits: int) -> list[int]:
    """Extract the passage numbers an answer cites, tolerating common bracket variants.

    Args:
        text: The model's answer.
        n_hits: Number of passages given to the model; numbers outside 1..n_hits are dropped.

    Returns:
        Sorted, de-duplicated passage numbers (1-based).
    """
    nums = {int(n) for m in _CITATION.findall(text) for n in re.findall(r"\d+", m)}
    return sorted(n for n in nums if 1 <= n <= n_hits)


SYSTEM = f"""You answer questions about FastAPI using ONLY the numbered context passages.
Rules:
- Cite every claim with the passage number in plain ASCII square brackets, one number per
  bracket, e.g. [2] or [2][5]. Never use other bracket styles, ranges or parentheses for citations.
- Every answer that is not {NO_ANSWER} must contain at least one citation, including partial ones.
- Never use outside knowledge, even if you are sure of it.
- Passages include code examples; read them, they often answer the question directly.
- If the passages do not contain the answer, reply with exactly: {NO_ANSWER}
  followed by one sentence saying what information is missing.
- If the passages answer only part of the question, answer that part (with citations), then add a
  line starting with "Not found:" listing what is missing. Use "Not found:" only for things that
  are truly absent from every passage."""


def build_prompt(question: str, hits: list[Hit]) -> str:
    """Format retrieved chunks and the question into the user prompt.

    Args:
        question: The user's question.
        hits: Retrieved chunks, in rank order. Each is numbered from 1; those numbers are what
            the LLM cites as `[n]`.

    Returns:
        The prompt text: numbered context passages (with source and heading) then the question.
    """
    ctx = "\n\n".join(
        f"[{i}] ({h.source} :: {h.heading})\n{h.content}" for i, h in enumerate(hits, 1)
    )
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
    cached: bool = False  # answer came from the disk cache (cost no API request)
    trace: list[dict] = field(default_factory=list)


async def generate_answer(llm: LLM, question: str, hits: list[Hit]) -> Answer:
    """Ask the LLM to answer from the retrieved chunks only, and parse its citations.

    The system prompt tells the model to cite passages as `[n]` and to reply
    `INSUFFICIENT_CONTEXT` when the passages do not contain the answer. Citation numbers outside
    1..len(hits) are ignored (see `parse_citations` for the accepted bracket forms), so a hallucinated `[9]` never produces a citation.

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
    cited = parse_citations(res.text, len(hits))
    return Answer(
        question=question,
        text=res.text.strip(),
        abstained=res.text.strip().startswith(NO_ANSWER),
        citations=[hits[i - 1] for i in cited],
        retrieved=hits,
        input_tokens=res.input_tokens,
        output_tokens=res.output_tokens,
        latency_s=time.perf_counter() - t0,
        cached=res.cached,
    )
