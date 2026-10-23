"""Error and trace reporting to Sentry. Off unless `SENTRY_DSN` is set, so tests and offline dev
runs never send events.

Privacy follows the logging rules: no prompts, document text or request bodies leave the process.
"""

import logging

import sentry_sdk

from app.config import get_settings

log = logging.getLogger(__name__)


def setup_sentry() -> bool:
    """Initialise the Sentry SDK from settings. Returns whether it was enabled.

    Call it before the FastAPI app is created so the framework integrations can hook in.
    """
    s = get_settings()
    if not s.sentry_dsn:
        log.debug("sentry disabled: SENTRY_DSN is not set")
        return False
    sentry_sdk.init(
        dsn=s.sentry_dsn,
        environment=s.sentry_environment,
        traces_sample_rate=s.sentry_traces_sample_rate,
        send_default_pii=False,
        include_local_variables=False,  # locals can hold prompts and document text
        max_request_body_size="never",
    )
    log.info(
        "sentry enabled: environment=%s traces_sample_rate=%s",
        s.sentry_environment,
        s.sentry_traces_sample_rate,
    )
    return True
