"""Contract test for the app<->infra delivery seam (RFC-110 / ADR-145).

The ``DeliveryEnvelope`` is the ONLY thing the learning-player server (which produces
envelopes) and the infra delivery service (#1412, which consumes them) share. This test
pins the contract: the committed JSON Schema + golden fixtures are the single source of
truth both sides validate against. The infra repo mirrors these assertions against the
same fixtures so the two tracks cannot drift.

Invariants asserted beyond raw schema validity:
- **Carry the graph** (the moat rule): every digest item carries ``graph_refs`` + a
  ``deep_link`` — never a flat clip.
- **Bridge-only audio**: no source-audio field anywhere in an envelope (PRD-035 Principle 4).
- **Idempotency**: a non-empty ``id`` the delivery service dedupes on.
- **Channel/recipient consistency**: email carries a verified address; push carries a
  subscription.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCHEMA_PATH = _REPO_ROOT / "docs" / "api" / "delivery-envelope.schema.json"
_FIXTURE_DIR = _REPO_ROOT / "tests" / "fixtures" / "delivery"

_GOLDEN_FILES = sorted(_FIXTURE_DIR.glob("*.golden.json"))

# Any of these keys appearing anywhere in an envelope would mean we are shipping source
# audio — forbidden (bridge-only). Kept as a denylist so a future field addition is caught.
_FORBIDDEN_AUDIO_KEYS = {
    "audio",
    "audio_url",
    "audio_uri",
    "enclosure",
    "enclosure_url",
    "media_url",
}


def _load(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def schema() -> dict:
    return _load(_SCHEMA_PATH)


def test_schema_is_itself_valid(schema: dict) -> None:
    # A malformed schema would make every downstream validation meaningless.
    Draft202012Validator.check_schema(schema)


def test_golden_fixtures_exist() -> None:
    # Guard against a silently-empty glob turning the parametrized tests into no-ops.
    assert _GOLDEN_FILES, f"no *.golden.json fixtures under {_FIXTURE_DIR}"
    names = {p.name for p in _GOLDEN_FILES}
    assert "your-week-digest.v1.golden.json" in names
    assert "resurface-nudge.v1.golden.json" in names


@pytest.mark.parametrize("golden_path", _GOLDEN_FILES, ids=lambda p: p.name)
def test_golden_validates_against_schema(schema: dict, golden_path: Path) -> None:
    envelope = _load(golden_path)
    errors = sorted(Draft202012Validator(schema).iter_errors(envelope), key=str)
    assert not errors, "\n".join(f"{list(e.path)}: {e.message}" for e in errors)


@pytest.mark.parametrize("golden_path", _GOLDEN_FILES, ids=lambda p: p.name)
def test_envelope_is_idempotent_and_versioned(golden_path: Path) -> None:
    env = _load(golden_path)
    assert env["schema_version"] == "1"
    assert isinstance(env["id"], str) and env["id"], "envelope id (dedupe key) must be non-empty"


@pytest.mark.parametrize("golden_path", _GOLDEN_FILES, ids=lambda p: p.name)
def test_envelope_carries_ttl(golden_path: Path) -> None:
    # Seam v1.1: expires_at bounds delivery so a homelab-down window can't flush stale digests.
    env = _load(golden_path)
    assert isinstance(env.get("expires_at"), str) and env["expires_at"] > env["created_at"]


@pytest.mark.parametrize("golden_path", _GOLDEN_FILES, ids=lambda p: p.name)
def test_channel_recipient_consistency(golden_path: Path) -> None:
    env = _load(golden_path)
    recipient = env["recipient"]
    if env["channel"] == "email":
        assert recipient.get("email"), "email channel needs an address"
        # Never enqueue email to an unverified address.
        assert recipient.get("email_verified") is True
    elif env["channel"] == "push":
        assert recipient.get("push_subscription"), "push channel needs a subscription"


def _walk(node: object):
    """Yield every (key, value) pair reachable in a nested dict/list structure."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield key, value
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


@pytest.mark.parametrize("golden_path", _GOLDEN_FILES, ids=lambda p: p.name)
def test_no_source_audio_anywhere(golden_path: Path) -> None:
    env = _load(golden_path)
    offending = {k for k, _ in _walk(env) if k in _FORBIDDEN_AUDIO_KEYS}
    assert not offending, f"bridge-only audio violated: {offending} present in {golden_path.name}"


def _digest_items(env: dict):
    template = env["template"]
    if template == "your-week-digest.v1":
        for section in env["payload"]["sections"]:
            yield from section["items"]
    elif template == "resurface-nudge.v1":
        yield env["payload"]["lead"]


@pytest.mark.parametrize("golden_path", _GOLDEN_FILES, ids=lambda p: p.name)
def test_every_item_carries_the_graph(golden_path: Path) -> None:
    # The moat rule: an outbound item is a graph node (deep_link + graph_refs), not a flat clip.
    env = _load(golden_path)
    items = list(_digest_items(env))
    assert items, "an envelope with no deliverable items should not have been enqueued"
    for item in items:
        assert item.get("deep_link"), f"item missing deep_link: {item}"
        refs = item.get("graph_refs") or []
        assert refs, f"item carries no graph_refs (flat clip): {item.get('episode_slug')}"
        for ref in refs:
            assert ref["kind"] in {"person", "topic"}
            assert ref["id"].startswith(("person:", "topic:"))


@pytest.mark.parametrize("golden_path", _GOLDEN_FILES, ids=lambda p: p.name)
def test_auto_pick_source_is_marked(golden_path: Path) -> None:
    # Auto-extracted picks (FR3) must be distinguishable from user captures at the contract level.
    env = _load(golden_path)
    for item in _digest_items(env):
        if "source" in item:
            assert item["source"] in {"user", "auto"}
