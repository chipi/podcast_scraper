"""Unit tests for sign-in access policy (#1064)."""

from __future__ import annotations

from podcast_scraper.server.app_access import AccessPolicy, policy_from_env


def test_open_allows_anyone() -> None:
    policy = AccessPolicy("open", frozenset(), frozenset())
    assert policy.is_allowed("a@x.com")
    assert policy.is_allowed("")


def test_allowlist_by_email_is_case_insensitive() -> None:
    policy = AccessPolicy("allowlist", frozenset({"jane@x.com"}), frozenset())
    assert policy.is_allowed("Jane@X.com")
    assert not policy.is_allowed("bob@x.com")
    assert not policy.is_allowed("")


def test_allowlist_by_domain() -> None:
    policy = AccessPolicy("allowlist", frozenset(), frozenset({"x.com"}))
    assert policy.is_allowed("anyone@x.com")
    assert not policy.is_allowed("anyone@y.com")


def test_policy_from_env(monkeypatch) -> None:
    monkeypatch.setenv("APP_SIGNUP_MODE", "allowlist")
    monkeypatch.setenv("APP_ALLOWED_EMAILS", "A@x.com, b@y.com")
    monkeypatch.setenv("APP_ALLOWED_DOMAINS", "z.com")
    policy = policy_from_env()
    assert policy.mode == "allowlist"
    assert policy.is_allowed("a@x.com")
    assert policy.is_allowed("anyone@z.com")
    assert not policy.is_allowed("c@q.com")


def test_policy_from_env_defaults_to_deny(monkeypatch) -> None:
    for var in ("APP_SIGNUP_MODE", "APP_ALLOWED_EMAILS", "APP_ALLOWED_DOMAINS"):
        monkeypatch.delenv(var, raising=False)
    policy = policy_from_env()
    assert policy.mode == "allowlist"
    assert not policy.is_allowed("anyone@x.com")  # default-deny until configured
