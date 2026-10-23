"""Sentry is opt-in: no DSN means no SDK init, a DSN means a privacy-safe init."""

import sentry_sdk

from app.config import get_settings
from app.observability import setup_sentry


def test_sentry_stays_off_without_a_dsn(monkeypatch):
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    calls = []
    monkeypatch.setattr(sentry_sdk, "init", lambda **kw: calls.append(kw))
    assert setup_sentry() is False
    assert calls == []


def test_sentry_init_uses_settings_and_keeps_content_out(monkeypatch):
    monkeypatch.setenv("SENTRY_DSN", "https://key@example.ingest.sentry.io/1")
    monkeypatch.setenv("SENTRY_ENVIRONMENT", "staging")
    monkeypatch.setenv("SENTRY_TRACES_SAMPLE_RATE", "0.5")
    get_settings.cache_clear()
    calls = []
    monkeypatch.setattr(sentry_sdk, "init", lambda **kw: calls.append(kw))
    assert setup_sentry() is True
    (kw,) = calls
    assert kw["dsn"] == "https://key@example.ingest.sentry.io/1"
    assert kw["environment"] == "staging"
    assert kw["traces_sample_rate"] == 0.5
    assert kw["send_default_pii"] is False
    assert kw["include_local_variables"] is False
    assert kw["max_request_body_size"] == "never"
