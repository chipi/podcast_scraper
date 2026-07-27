"""o11y P2: the API surface initialises OTEL (was pipeline-CLI-only), so API errors/events
carry a trace_id."""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.mark.unit
def test_init_api_otel_delegates_to_init_otel():
    from podcast_scraper.server import app

    with patch("podcast_scraper.utils.otel_init.init_otel") as m:
        app._init_api_otel()
    m.assert_called_once()


@pytest.mark.unit
def test_init_api_otel_never_raises_on_failure():
    from podcast_scraper.server import app

    with patch("podcast_scraper.utils.otel_init.init_otel", side_effect=RuntimeError("x")):
        app._init_api_otel()  # swallowed — API startup must not break on tracing
