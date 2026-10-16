"""Chat client for any OpenAI-compatible `/chat/completions` API.

Groq, GitHub Models, OpenRouter, Together, a local vLLM or Ollama all speak this; only the base
URL, key and model name differ.
"""

import httpx

from .base import LLMResult
from .http import post_json

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class OpenAICompatLLM:
    """`LLM` implementation over `POST {base_url}/chat/completions`."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = GROQ_BASE_URL,
        *,
        client: httpx.AsyncClient | None = None,
    ):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.client = client or httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=120.0
        )

    async def generate(
        self, prompt: str, *, system: str = "", json: bool = False, temperature: float = 0.0
    ) -> LLMResult:
        """Send one chat request and return the first choice; `json=True` asks for a JSON object."""
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        payload: dict = {"model": self.model, "messages": messages, "temperature": temperature}
        if json:
            payload["response_format"] = {"type": "json_object"}
        body = await post_json(self.client, f"{self.base_url}/chat/completions", payload)
        usage = body.get("usage") or {}
        return LLMResult(
            text=(body["choices"][0]["message"].get("content") or ""),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )
