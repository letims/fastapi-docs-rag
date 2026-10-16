"""One place to configure logging for every entry point (worker, API, evals, scripts).

Levels used across the project:
  INFO     decisions and milestones: what was skipped or re-embedded and why, retries and backoff,
           retrieval results, each step of the agent loop, quota thresholds, progress.
  DEBUG    per-item detail: every claimed job, embedding batches, cache hits and misses, fusion
           overlap. Turn it on with `LOG_LEVEL=DEBUG` when you want to see everything.
  WARNING  something is degraded or will be retried: rate limits, failed jobs, unusable grader reply.

Never log secrets: API keys live only in `.env` and are not part of any log line. Question text is
logged (first 60 characters) at INFO; use `LOG_LEVEL=WARNING` where that is not acceptable.
"""

import logging

from app.config import get_settings

FORMAT = "%(asctime)s %(levelname)-7s %(name)s %(message)s"
_QUIET = ("httpx", "httpcore")  # one INFO line per HTTP request drowns everything else


def setup_logging(level: str | None = None) -> None:
    """Configure the root logger once; `LOG_LEVEL` (default INFO) sets the threshold.

    Safe to call more than once and when a handler already exists (tests, uvicorn): in that case
    only the level is applied.

    Args:
        level: Overrides `LOG_LEVEL`, e.g. "DEBUG".
    """
    name = (level or get_settings().log_level).upper()
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(name)
    for noisy in _QUIET:
        logging.getLogger(noisy).setLevel(logging.WARNING)
