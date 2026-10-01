"""The persisted sign-in access policy (#2190).

The property that matters most is the boring one: an instance that has never used the admin
endpoint must behave EXACTLY as it did when the policy came only from env. Everything else here is
about the two ways this feature could do real damage — failing open, and locking the operator out.
"""

from __future__ import annotations

import json
from pathlib import Path

from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_access_store import (
    effective_policy,
    load_policy,
    policy_from_dict,
    policy_to_dict,
    save_policy,
)

ENV_POLICY = AccessPolicy(
    mode="allowlist",
    allowed_emails=frozenset({"env@example.com"}),
    allowed_domains=frozenset({"env.test"}),
)


def _policy(*emails: str, mode: str = "allowlist", domains: tuple[str, ...] = ()) -> AccessPolicy:
    return AccessPolicy(
        mode=mode, allowed_emails=frozenset(emails), allowed_domains=frozenset(domains)
    )


# --- the fallback contract -------------------------------------------------------------------


def test_absent_file_falls_back_to_env(tmp_path: Path) -> None:
    """The whole-feature safety property: no file == previous behaviour, unchanged."""
    assert load_policy(tmp_path) is None
    assert effective_policy(tmp_path, ENV_POLICY) is ENV_POLICY


def test_absent_data_dir_falls_back_to_env() -> None:
    """An instance with no data dir configured still enforces the env policy."""
    assert load_policy(None) is None
    assert effective_policy(None, ENV_POLICY) is ENV_POLICY


def test_persisted_policy_wins_over_env(tmp_path: Path) -> None:
    save_policy(tmp_path, _policy("file@example.com"))
    effective = effective_policy(tmp_path, ENV_POLICY)
    assert effective is not None
    assert effective.is_allowed("file@example.com")
    # The env address is GONE, not merged — revoking has to be possible without a deploy.
    assert not effective.is_allowed("env@example.com")


# --- malformed input must not fail open, and must not crash ------------------------------------


def test_unparseable_file_falls_back_to_env_rather_than_denying_everyone(tmp_path: Path) -> None:
    (tmp_path / "access_policy.json").write_text("{not json", encoding="utf-8")
    assert load_policy(tmp_path) is None
    assert effective_policy(tmp_path, ENV_POLICY) is ENV_POLICY


def test_non_object_json_falls_back(tmp_path: Path) -> None:
    (tmp_path / "access_policy.json").write_text("[1, 2, 3]", encoding="utf-8")
    assert load_policy(tmp_path) is None


def test_unknown_mode_denies_rather_than_opens() -> None:
    """A corrupted mode must degrade to allowlist. Failing OPEN here would expose the platform."""
    assert policy_from_dict({"mode": "sideways", "allowed_emails": []}).mode == "allowlist"
    assert policy_from_dict({"mode": 42}).mode == "allowlist"
    assert policy_from_dict({}).mode == "allowlist"


def test_mode_open_survives_the_round_trip(tmp_path: Path) -> None:
    save_policy(tmp_path, _policy(mode="open"))
    loaded = load_policy(tmp_path)
    assert loaded is not None and loaded.mode == "open"
    assert loaded.is_allowed("anyone@anywhere.test")


def test_junk_entries_are_dropped_not_fatal() -> None:
    policy = policy_from_dict(
        {"mode": "allowlist", "allowed_emails": ["  A@B.com ", "", None, 7, "a@b.com"]}
    )
    assert policy.allowed_emails == frozenset({"a@b.com"})


# --- normalisation ------------------------------------------------------------------------------


def test_addresses_are_normalised_on_save(tmp_path: Path) -> None:
    save_policy(tmp_path, policy_from_dict({"allowed_emails": ["  Tester@Example.COM  "]}))
    loaded = load_policy(tmp_path)
    assert loaded is not None
    assert loaded.is_allowed("tester@example.com")
    assert loaded.is_allowed("TESTER@EXAMPLE.COM")


def test_domain_rule_still_applies(tmp_path: Path) -> None:
    save_policy(tmp_path, _policy(domains=("beta.test",)))
    loaded = load_policy(tmp_path)
    assert loaded is not None
    assert loaded.is_allowed("someone@beta.test")
    assert not loaded.is_allowed("someone@other.test")


def test_file_is_stable_and_sorted(tmp_path: Path) -> None:
    """Written twice from the same set, the bytes match — so the file diffs cleanly."""
    save_policy(tmp_path, _policy("c@x.test", "a@x.test", "b@x.test"))
    first = (tmp_path / "access_policy.json").read_text(encoding="utf-8")
    save_policy(tmp_path, _policy("b@x.test", "c@x.test", "a@x.test"))
    assert (tmp_path / "access_policy.json").read_text(encoding="utf-8") == first
    assert json.loads(first)["allowed_emails"] == ["a@x.test", "b@x.test", "c@x.test"]


def test_to_dict_from_dict_round_trip() -> None:
    original = _policy("a@x.test", mode="allowlist", domains=("x.test",))
    assert policy_from_dict(policy_to_dict(original)) == original


def test_save_creates_the_data_dir(tmp_path: Path) -> None:
    nested = tmp_path / "does" / "not" / "exist"
    save_policy(nested, _policy("a@x.test"))
    assert (nested / "access_policy.json").is_file()
