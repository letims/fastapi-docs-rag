"""Shared pieces for adapters that talk plain HTTP+JSON (Voyage, Groq, any OpenAI-compatible API)."""

import httpx

from app.retry import with_retries


def is_rate_limited(e: BaseException) -> bool:
    """True for an HTTP 429. Not retried by the adapters: `app.quota.Governor` owns 429 handling."""
    return isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429


def is_transient(e: BaseException) -> bool:
    """True for errors worth a backoff retry: 5xx responses and network-level failures."""
    if isinstance(e, httpx.HTTPStatusError):
        return e.response.status_code >= 500
    return isinstance(e, httpx.TransportError)


async def post_json(client: httpx.AsyncClient, url: str, payload: dict) -> dict:
    """POST `payload` as JSON and return the decoded body, retrying transient failures only.

    Raises:
        httpx.HTTPStatusError: for any non-2xx that is not retried (4xx, including 429), or when
            transient retries run out. The error text carries the response body, which is where
            these APIs explain what was wrong.
    """

    async def call() -> dict:
        resp = await client.post(url, json=payload)
        if resp.is_error:
            raise httpx.HTTPStatusError(
                f"{resp.status_code} from {url}: {resp.text[:500]}",
                request=resp.request,
                response=resp,
            )
        return resp.json()

    return await with_retries(call, retry_on=is_transient)
