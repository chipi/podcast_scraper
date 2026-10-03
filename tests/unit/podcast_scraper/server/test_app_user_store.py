"""Unit tests for the file-based per-user identity store (#1063)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from podcast_scraper.server.app_user_store import (
    create_user,
    delete_user,
    get_or_create_user,
    get_user,
    list_users,
    new_analytics_id,
    set_disabled,
    set_role,
    user_id_for,
)


def test_user_id_is_deterministic_and_prefixed() -> None:
    assert user_id_for("google", "sub1") == user_id_for("google", "sub1")
    assert user_id_for("google", "sub1") != user_id_for("google", "sub2")
    assert user_id_for("google", "sub1") != user_id_for("other", "sub1")
    assert user_id_for("google", "sub1").startswith("u_")


def test_get_or_create_is_idempotent(tmp_path: Path) -> None:
    u1 = get_or_create_user(tmp_path, provider="google", subject="s1", email="a@x.com", name="A")
    # second call with different email returns the SAME user, unchanged
    u2 = get_or_create_user(tmp_path, provider="google", subject="s1", email="b@x.com", name="B")
    assert u1.user_id == u2.user_id
    assert u2.email == "a@x.com"
    loaded = get_user(tmp_path, u1.user_id)
    assert loaded is not None and loaded.email == "a@x.com" and loaded.name == "A"


def test_username_is_derived_from_email_local_part(tmp_path: Path) -> None:
    u = get_or_create_user(
        tmp_path, provider="google", subject="s1", email="Jane.Doe@x.com", name="Jane"
    )
    assert u.username == "jane_doe"  # lowercased, non-alnum → underscore
    # Persisted + immutable: a later login (even with a different email) keeps the first handle.
    again = get_or_create_user(
        tmp_path, provider="google", subject="s1", email="other@x.com", name="X"
    )
    assert again.username == "jane_doe"
    loaded = get_user(tmp_path, u.user_id)
    assert loaded is not None and loaded.username == "jane_doe"


def test_username_dedupes_with_a_numeric_suffix(tmp_path: Path) -> None:
    a = get_or_create_user(tmp_path, provider="google", subject="s1", email="sam@x.com", name="Sam")
    b = get_or_create_user(tmp_path, provider="google", subject="s2", email="sam@y.com", name="Sam")
    c = get_or_create_user(tmp_path, provider="google", subject="s3", email="sam@z.com", name="Sam")
    assert a.username == "sam"
    assert {b.username, c.username} == {"sam2", "sam3"}


def test_username_falls_back_when_seed_is_unusable(tmp_path: Path) -> None:
    u = get_or_create_user(tmp_path, provider="google", subject="s1", email="!!!@x.com", name="")
    assert u.username == "user"


def test_oauth_image_is_captured_at_creation_and_persisted(tmp_path: Path) -> None:
    u = get_or_create_user(
        tmp_path,
        provider="google",
        subject="s1",
        email="a@x.com",
        name="A",
        image="https://cdn/pic.jpg",
    )
    assert u.image == "https://cdn/pic.jpg"
    loaded = get_user(tmp_path, u.user_id)
    assert loaded is not None and loaded.image == "https://cdn/pic.jpg"


def test_no_image_is_none_not_empty(tmp_path: Path) -> None:
    u = get_or_create_user(tmp_path, provider="google", subject="s1", email="a@x.com", name="A")
    assert u.image is None


def test_create_user_also_mints_a_handle(tmp_path: Path) -> None:
    # advisor M4: the admin/seed creation path mints a handle too, deduped against existing users.
    a = get_or_create_user(tmp_path, provider="google", subject="s1", email="dana@x.com", name="D")
    b = create_user(
        tmp_path, provider="stub", subject="s2", email="dana@y.com", name="D", role="listener"
    )
    assert a.username == "dana"
    assert b.username == "dana2"  # deduped against the existing 'dana'


def test_get_user_missing(tmp_path: Path) -> None:
    assert get_user(tmp_path, "u_does_not_exist") is None


def test_set_disabled_and_list_users(tmp_path: Path) -> None:
    u1 = get_or_create_user(tmp_path, provider="google", subject="s1", email="a@x.com", name="A")
    u2 = get_or_create_user(tmp_path, provider="google", subject="s2", email="b@x.com", name="B")
    assert set_disabled(tmp_path, u1.user_id, True) is True
    reloaded = get_user(tmp_path, u1.user_id)
    assert reloaded is not None and reloaded.disabled is True
    assert set_disabled(tmp_path, "u_does_not_exist", True) is False
    assert {u.user_id for u in list_users(tmp_path)} == {u1.user_id, u2.user_id}


def test_delete_user(tmp_path: Path) -> None:
    user = get_or_create_user(tmp_path, provider="google", subject="s1", email="a@x.com", name="A")
    assert delete_user(tmp_path, user.user_id) is True
    assert get_user(tmp_path, user.user_id) is None
    assert delete_user(tmp_path, user.user_id) is False


def test_path_traversal_user_ids_are_rejected_not_found(tmp_path: Path) -> None:
    """A non-conforming user_id (``..`` / ``/`` / non-hex) must never touch the
    filesystem: lookups/mutations return the graceful not-found result and a file
    outside the users/ tree is never read or removed. Real ids are always
    ``user_id_for`` = ``u_`` + 24 hex; anything else is treated as unknown."""
    victim = tmp_path / "secret.txt"
    victim.write_text("keep me", encoding="utf-8")

    for bad in ("../secret", "u_../../secret", "u_/etc/passwd", "u_NOTHEX", "", "u_short"):
        assert get_user(tmp_path, bad) is None
        assert set_disabled(tmp_path, bad, True) is False
        assert set_role(tmp_path, bad, "admin") is False
        assert delete_user(tmp_path, bad) is False

    assert victim.read_text(encoding="utf-8") == "keep me"


def test_valid_shape_but_absent_user_id_is_not_found(tmp_path: Path) -> None:
    """A well-formed but non-existent id is a clean not-found (distinct from rejected)."""
    assert get_user(tmp_path, "u_" + "0" * 24) is None
    assert delete_user(tmp_path, "u_" + "1" * 24) is False
    assert set_disabled(tmp_path, "u_" + "2" * 24, True) is False
    assert set_role(tmp_path, "u_" + "3" * 24, "admin") is False


# --- analytics_id (#2265) -------------------------------------------------------------------


def test_analytics_id_is_minted_at_creation_and_is_a_uuid4(tmp_path: Path) -> None:
    user = get_or_create_user(tmp_path, provider="google", subject="s1", email="a@x.com", name="A")
    assert user.analytics_id
    # A real UUID, not a truncated hash or a slug.
    parsed = uuid.UUID(user.analytics_id)
    assert parsed.version == 4
    # Survives a reload — it is persisted, not computed per call.
    assert get_user(tmp_path, user.user_id).analytics_id == user.analytics_id  # type: ignore[union-attr]


def test_analytics_id_is_not_derived_from_the_identity(tmp_path: Path) -> None:
    """The whole point of the field: it must not be re-derivable from who the person is.

    If it were ``user_id``, the email, or any digest of them, anyone holding an analytics export
    plus a guessed identity could confirm the match. Two accounts with the same name and adjacent
    subjects must get unrelated ids.
    """
    # A realistic, multi-character identity on purpose. An earlier version of this test used
    # ``a@x.com``, and "a" appears in nearly every UUID's hex — the assertion passed or failed on
    # luck rather than on whether the id was derived.
    a = get_or_create_user(
        tmp_path, provider="google", subject="s1", email="jordan.lee@x.com", name="Jordan Lee"
    )
    b = get_or_create_user(
        tmp_path, provider="google", subject="s2", email="jordan.lee@x.com", name="Jordan Lee"
    )
    assert a.analytics_id != b.analytics_id
    for user in (a, b):
        assert user.analytics_id != user.user_id
        assert user.user_id.removeprefix("u_") not in user.analytics_id
        assert "jordan" not in user.analytics_id.lower()
        assert "lee" not in user.analytics_id.lower()
        assert user.username not in user.analytics_id


def test_analytics_id_is_stable_across_sign_ins(tmp_path: Path) -> None:
    """It is promised never to rotate: a rotated id splits one participant across two reports."""
    first = get_or_create_user(tmp_path, provider="google", subject="s1", email="a@x.com", name="A")
    for _ in range(3):
        again = get_or_create_user(
            tmp_path, provider="google", subject="s1", email="a@x.com", name="A"
        )
        assert again.analytics_id == first.analytics_id


def test_analytics_id_is_backfilled_for_a_profile_written_before_the_field(
    tmp_path: Path,
) -> None:
    """Every account that exists today predates the field, including the operator's own.

    Minting only at creation would leave them permanently unmeasurable, so the next sign-in fills
    one in. Simulated by stripping the key from the profile on disk, which is exactly the shape of
    an older profile.
    """
    user = get_or_create_user(tmp_path, provider="google", subject="s1", email="a@x.com", name="A")
    path = tmp_path / "users" / user.user_id / "profile.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    del doc["analytics_id"]
    path.write_text(json.dumps(doc), encoding="utf-8")

    # Reading it back shows the gap...
    assert get_user(tmp_path, user.user_id).analytics_id == ""  # type: ignore[union-attr]

    # ...and signing in closes it, persistently.
    back = get_or_create_user(tmp_path, provider="google", subject="s1", email="a@x.com", name="A")
    assert back.analytics_id
    assert get_user(tmp_path, user.user_id).analytics_id == back.analytics_id  # type: ignore[union-attr]

    # And the backfilled id is then itself stable.
    again = get_or_create_user(tmp_path, provider="google", subject="s1", email="a@x.com", name="A")
    assert again.analytics_id == back.analytics_id


def test_backfill_does_not_block_sign_in_when_the_write_fails(
    tmp_path: Path, monkeypatch: object
) -> None:
    """Analytics must never be the reason someone cannot sign in."""
    user = get_or_create_user(tmp_path, provider="google", subject="s1", email="a@x.com", name="A")
    path = tmp_path / "users" / user.user_id / "profile.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    del doc["analytics_id"]
    path.write_text(json.dumps(doc), encoding="utf-8")

    import podcast_scraper.server.app_user_store as store

    def boom(*_a: object, **_k: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(store, "_write_profile", boom)  # type: ignore[attr-defined]
    back = get_or_create_user(tmp_path, provider="google", subject="s1", email="a@x.com", name="A")
    # Sign-in still succeeds; the id is simply still missing, to be filled next time.
    assert back.user_id == user.user_id
    assert back.analytics_id == ""


def test_created_users_also_get_one(tmp_path: Path) -> None:
    """``create_user`` is the admin / seed path, and those accounts are measured too."""
    user = create_user(
        tmp_path, provider="google", subject="s9", email="c@x.com", name="C", role="listener"
    )
    assert uuid.UUID(user.analytics_id).version == 4


def test_new_analytics_id_does_not_repeat() -> None:
    assert len({new_analytics_id() for _ in range(50)}) == 50
