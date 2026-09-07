"""A dev twin profile must stay identical to its prod counterpart except for storage.

WHAT A DEV TWIN IS FOR. ``dev_dgx_full`` and ``dev_cloud_balanced`` exist so a developer can run
the EXACT production pipeline locally — same ASR, same models, same fallback ladders, same
thresholds — and have the result mean something about prod. The moment the two drift, the dev
profile is validating a pipeline nobody runs, which is worse than having no dev profile at all
because it looks like coverage.

WHY THIS TEST IS NEEDED. Two mechanisms already exist and NEITHER closes this:

* The registry preset is derived with ``dataclasses.replace(prod_preset, name=...)``, so the
  ROUTING cannot drift. But presets have no storage fields and do not describe most YAML keys.
* ``test_profile_yaml_registry_drift`` only checks a YAML against its preset, and it SKIPS both
  twins — "no overlay expectations for profile 'dev_dgx_full'" — exactly as it skips
  ``prod_dgx_full``. Opting in is a separate piece of work.

So the YAML-to-YAML relationship is unguarded, and that is the one a human edit breaks: someone
tunes ``summary_model`` or ``dgx_diarize_request_timeout_sec`` in the prod profile, the twin keeps
the old value, and nothing anywhere notices.

THE ONE INTENDED DIFFERENCE is where the raw-audio archive lives. Prod ships it to the Hetzner cold
box; dev keeps it on the local disk, because a dev box has no Hetzner credentials and its disk is
not shared. Everything else must match, and ``ALLOWED_DIVERGENCE`` is deliberately small so that
adding to it is a visible decision rather than a silent one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

pytestmark = pytest.mark.unit

_PROFILE_DIR = Path(__file__).resolve().parents[3] / "config" / "profiles"

#: dev twin -> the prod profile it must track.
TWINS = {
    "dev_dgx_full": "prod_dgx_full",
    "dev_cloud_balanced": "cloud_balanced",
}

#: The ONLY keys a twin may differ on. Everything here is storage-transport or identity.
ALLOWED_DIVERGENCE = {
    "profile",  # each profile names itself
    "audio_storage_backend",  # local (dev) vs remote (prod) — the point of the twin
    "audio_remote_rclone_remote",  # remote-only: absent in dev
    "audio_remote_base_path",  # remote-only: absent in dev
    "audio_evict_local_after_offload",  # remote-only: nothing to evict INTO on a local backend
    "audio_cache_in_corpus",  # dev keeps the archive inside the corpus
    "corpus_media_link_mode",  # hardlink is only footprint-halving under a LOCAL backend
}


def _load(name: str) -> Dict[str, Any]:
    path = _PROFILE_DIR / f"{name}.yaml"
    assert path.is_file(), f"profile not found: {path}"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


@pytest.mark.parametrize("dev,prod", sorted(TWINS.items()))
def test_twin_matches_prod_except_for_storage(dev: str, prod: str) -> None:
    """THE REGRESSION GUARD: edit prod and forget the twin, and this goes red."""
    d, p = _load(dev), _load(prod)

    drifted = sorted(
        k
        for k in (set(d) | set(p)) - ALLOWED_DIVERGENCE
        if d.get(k, "<missing>") != p.get(k, "<missing>")
    )
    assert not drifted, (
        f"{dev}.yaml has drifted from {prod}.yaml on: {drifted}. A dev twin exists so local runs "
        f"reproduce prod exactly; once they diverge it validates a pipeline nobody runs. Either "
        f"mirror the change into {dev}.yaml, or — if the difference is deliberate — add the key to "
        "ALLOWED_DIVERGENCE with a reason."
    )


@pytest.mark.parametrize("dev,prod", sorted(TWINS.items()))
def test_the_storage_difference_is_actually_present(dev: str, prod: str) -> None:
    """Pins the twin's whole reason for existing, in both directions.

    Without this, a twin that accidentally became byte-identical to prod would pass the drift test
    above while quietly requiring Hetzner credentials no dev box has.
    """
    d, p = _load(dev), _load(prod)
    assert p.get("audio_storage_backend") == "remote", f"{prod} should archive to cold"
    assert d.get("audio_storage_backend") == "local", f"{dev} should archive locally"


@pytest.mark.parametrize("dev", sorted(TWINS))
def test_twin_does_not_disable_the_archive(dev: str) -> None:
    """``audio_cache_enabled: false`` would disable the archive ENTIRELY, not just the local copy.

    ``resolve_backend`` returns None on that flag BEFORE the remote branch is reached, so a profile
    setting it archives nothing at all — no local file and no cold copy. On a dev profile that
    silently removes the reprocess source the twin exists to provide.
    """
    assert _load(dev).get("audio_cache_enabled") is not False, (
        f"{dev}.yaml must not set audio_cache_enabled: false — it gates resolve_backend() itself "
        "and would disable the archive entirely. Choose the transport with audio_storage_backend."
    )


@pytest.mark.parametrize("dev,prod", sorted(TWINS.items()))
def test_registry_preset_is_derived_not_copied(dev: str, prod: str) -> None:
    """The preset routing must be the prod values, not a hand-copy that can rot."""
    from podcast_scraper.providers.ml.model_registry import _PROFILE_PRESETS

    assert dev in _PROFILE_PRESETS, f"{dev} missing from the registry"
    a, b = _PROFILE_PRESETS[dev], _PROFILE_PRESETS[prod]
    for field in (
        "transcription",
        "summary",
        "kg",
        "ner",
        "clustering",
        "gi",
        "diarization",
        "grounding",
        "transcription_fallback",
        "diarization_fallback",
        "summary_fallback",
        "allow_cloud_fallback",
        "llm_pipeline_mode",
    ):
        assert getattr(a, field) == getattr(b, field), (
            f"{dev} preset diverges from {prod} on '{field}'. These are derived with "
            "dataclasses.replace(); do not hand-copy the routing."
        )
