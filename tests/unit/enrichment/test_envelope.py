"""Unit tests for ``enrichment.envelope``."""

from __future__ import annotations

import json

import pytest

from podcast_scraper.enrichment.envelope import (
    build_envelope,
    EnvelopePayload,
    EnvelopeShapeError,
    utc_iso_now,
    validate_envelope,
)
from podcast_scraper.enrichment.protocol import (
    EnricherResult,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_TIMEOUT,
)


def _build_minimal_ok_envelope() -> dict:
    return build_envelope(
        result=EnricherResult(status=STATUS_OK, data={"x": 1}, duration_ms=42),
        enricher_id="topic_cooccurrence",
        enricher_version="1.0.0",
        schema_version="1.0",
    ).to_dict()


# ---------------------------------------------------------------------------
# utc_iso_now
# ---------------------------------------------------------------------------


def test_utc_iso_now_returns_z_suffixed_string() -> None:
    s = utc_iso_now()
    # YYYY-MM-DDTHH:MM:SSZ
    assert len(s) == 20
    assert s.endswith("Z")
    assert s[4] == "-" and s[7] == "-"
    assert s[10] == "T"


# ---------------------------------------------------------------------------
# build_envelope
# ---------------------------------------------------------------------------


def test_build_envelope_ok_status_includes_data() -> None:
    env = build_envelope(
        result=EnricherResult(status=STATUS_OK, data={"hello": "world"}),
        enricher_id="x",
        enricher_version="1.0.0",
        schema_version="1.0",
    )
    assert env.derived is True
    assert env.status == STATUS_OK
    assert env.data == {"hello": "world"}
    assert env.error is None


def test_build_envelope_failed_status_drops_data() -> None:
    """Even if EnricherResult somehow has data, failed envelopes carry None."""
    env = build_envelope(
        result=EnricherResult(
            status=STATUS_FAILED,
            error="boom",
            error_class="RuntimeError",
        ),
        enricher_id="x",
        enricher_version="1.0.0",
        schema_version="1.0",
    )
    assert env.status == STATUS_FAILED
    assert env.data is None
    assert env.error == "boom"
    assert env.error_class == "RuntimeError"


def test_build_envelope_uses_provided_computed_at() -> None:
    env = build_envelope(
        result=EnricherResult(status=STATUS_OK, data={}),
        enricher_id="x",
        enricher_version="1.0.0",
        schema_version="1.0",
        computed_at="2026-06-26T15:01:42Z",
    )
    assert env.computed_at == "2026-06-26T15:01:42Z"


def test_build_envelope_round_trips_through_to_dict() -> None:
    env = build_envelope(
        result=EnricherResult(
            status=STATUS_OK,
            data={"k": [1, 2, 3]},
            retry_count=2,
            circuit_state="closed",
            duration_ms=100,
            records_written=3,
        ),
        enricher_id="topic_cooccurrence",
        enricher_version="1.0.0",
        schema_version="2.0",
    )
    d = env.to_dict()
    # All envelope fields surface in the dict (no lossy transformation).
    assert d["derived"] is True
    assert d["enricher_id"] == "topic_cooccurrence"
    assert d["schema_version"] == "2.0"
    assert d["status"] == STATUS_OK
    assert d["data"] == {"k": [1, 2, 3]}
    assert d["retry_count"] == 2
    assert d["circuit_state"] == "closed"
    assert d["duration_ms"] == 100
    assert d["records_written"] == 3
    # JSON-serializable.
    assert json.loads(json.dumps(d)) == d


# ---------------------------------------------------------------------------
# validate_envelope
# ---------------------------------------------------------------------------


def test_validate_envelope_accepts_a_built_envelope() -> None:
    validate_envelope(_build_minimal_ok_envelope())


def test_validate_envelope_rejects_missing_required_keys() -> None:
    env = _build_minimal_ok_envelope()
    del env["derived"]
    with pytest.raises(EnvelopeShapeError, match="missing required key: 'derived'"):
        validate_envelope(env)


def test_validate_envelope_rejects_derived_false() -> None:
    env = _build_minimal_ok_envelope()
    env["derived"] = False
    with pytest.raises(EnvelopeShapeError, match="derived must be True"):
        validate_envelope(env)


def test_validate_envelope_rejects_unknown_status() -> None:
    env = _build_minimal_ok_envelope()
    env["status"] = "weird"
    with pytest.raises(EnvelopeShapeError, match="status must be one of"):
        validate_envelope(env)


def test_validate_envelope_rejects_ok_status_with_no_data() -> None:
    env = _build_minimal_ok_envelope()
    env["data"] = None
    with pytest.raises(EnvelopeShapeError, match="non-None data"):
        validate_envelope(env)


@pytest.mark.parametrize("field", ["enricher_id", "enricher_version", "schema_version"])
def test_validate_envelope_rejects_empty_string_fields(field: str) -> None:
    env = _build_minimal_ok_envelope()
    env[field] = ""
    with pytest.raises(EnvelopeShapeError, match=f"{field}"):
        validate_envelope(env)


def test_validate_envelope_accepts_non_ok_with_null_data() -> None:
    env = build_envelope(
        result=EnricherResult(status=STATUS_TIMEOUT, error="hit hard timeout"),
        enricher_id="x",
        enricher_version="1.0.0",
        schema_version="1.0",
    ).to_dict()
    validate_envelope(env)


# ---------------------------------------------------------------------------
# EnvelopePayload identity
# ---------------------------------------------------------------------------


def test_envelope_payload_is_frozen() -> None:
    env = build_envelope(
        result=EnricherResult(status=STATUS_OK, data={}),
        enricher_id="x",
        enricher_version="1.0.0",
        schema_version="1.0",
    )
    assert isinstance(env, EnvelopePayload)
    with pytest.raises(Exception):
        env.derived = False  # type: ignore[misc]
