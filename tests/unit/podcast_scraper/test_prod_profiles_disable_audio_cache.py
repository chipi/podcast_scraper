"""Prod profiles must not keep a LOCAL raw-audio cache (#947 / #1199 follow-up).

The GUID-addressed audio cache and the cold archive are the same archive with two transports —
``audio_cache.py`` says the local and remote backends "store by the same sharded GUID key" and are
byte-compatible. So with ``audio_storage_backend: remote``, a local cache is a second copy of bytes
already held in cold, on the disk that copy is filling.

WHY THIS IS A TEST AND NOT JUST A PROFILE EDIT. The cache has **no pruning mechanism** — no TTL, no
max-size, no retention setting exists anywhere in the codebase — so it is append-only by
construction and grows without bound. Nothing else reclaims it either: the end-of-run eviction and
the ``sweep-prod-audio`` workflow both operate on ``media/``, which was 2.96 GB on prod while the
cache was **47.28 GB across 656 files**. A dry-run sweep on 2026-09-07 examined 44 files and would
have evicted 0 — it cannot see the cache at all.

Measured growth was ~6 GB on an active day against 64 GB free, i.e. ~10 days of runway, and it was
still growing after ``audio_storage_backend: remote`` had been deployed. Deleting the profile line
therefore does not fail loudly; it silently resumes a disk leak that only shows up as a full disk
days later. That is what this test exists to catch.

``audio_cache_enabled: false`` is the unconditional off switch: ``resolve_cache_root()`` returns
``None`` on it BEFORE ``audio_cache_in_corpus`` or any backend logic is consulted — which matters
because prod's corpus operator YAML sets ``audio_cache_in_corpus: true``, and that alone would put
the cache back inside the corpus.

Development profiles deliberately keep the cache: re-fetching audio on every reprocess is the real
cost there, and a dev disk is not a shared production resource.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

#: Profiles that run against the production corpus, where cold storage is the archive of record.
PROD_PROFILES = ("prod_dgx_full", "cloud_balanced")

_PROFILE_DIR = Path(__file__).resolve().parents[3] / "config" / "profiles"


def _load(name: str) -> dict:
    path = _PROFILE_DIR / f"{name}.yaml"
    assert path.is_file(), f"profile not found: {path}"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


@pytest.mark.parametrize("name", PROD_PROFILES)
def test_prod_profile_disables_the_local_audio_cache(name: str) -> None:
    """THE REGRESSION GUARD: removing this line resumes a ~6 GB/day unbounded disk leak."""
    cfg = _load(name)
    assert cfg.get("audio_cache_enabled") is False, (
        f"{name}.yaml must set `audio_cache_enabled: false`. Without it the GUID audio cache "
        "writes a second local copy of audio already stored in cold, and NOTHING prunes it: "
        "no TTL, no max-size, and the eviction/sweep path only covers media/. Measured 47.28 GB "
        "on prod before this was disabled."
    )


@pytest.mark.parametrize("name", PROD_PROFILES)
def test_prod_profile_still_uses_cold_storage(name: str) -> None:
    """Disabling the cache is only safe BECAUSE cold storage holds the archive.

    If a profile ever drops the remote backend while keeping the cache disabled, reprocessing
    loses its audio source entirely — a worse failure than the disk leak. The two settings are a
    pair and must be changed together.
    """
    cfg = _load(name)
    assert cfg.get("audio_storage_backend") == "remote", (
        f"{name}.yaml disables the local audio cache, so `audio_storage_backend: remote` is what "
        "keeps a retrievable copy. Do not change one without the other."
    )


def test_a_development_profile_keeps_the_cache() -> None:
    """Pins the intent: this is a prod-only restriction, not a global one.

    Asserted over whichever dev-ish profiles exist rather than a fixed name, so renaming one does
    not turn this into a false pass.
    """
    candidates = [
        p.stem
        for p in _PROFILE_DIR.glob("*.yaml")
        if p.stem in {"dev", "local", "local_dgx_full", "local_dgx_balanced"}
    ]
    assert candidates, "expected at least one development profile to exist"
    for name in candidates:
        assert _load(name).get("audio_cache_enabled") is not False, (
            f"{name}.yaml should NOT disable the audio cache — re-fetching audio on every local "
            "reprocess is the cost this cache exists to remove."
        )
