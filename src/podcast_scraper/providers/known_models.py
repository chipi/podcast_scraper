"""The set of cloud LLM model ids we KNOW — the allowlist that stops a wrong/fictional model
from silently running to completion.

THE INCIDENT this exists for: an eval arm was configured with ``claude-sonnet-4-6`` — a model that
was never requested (the ask was ``claude-sonnet-4-5``) and is very likely not a real Anthropic id.
Nothing validated it. The string flowed config → API → 18 finished episodes → the scoreboard, and
the mismatch only surfaced days later when a human read the provider's usage dashboard by hand. Two
holes let that happen and this module closes both:

* **fail-closed allowlist** (``is_known_model``): a cloud model that is not on the list is REJECTED
  at config/profile load, before a cent is spent, with a "did you mean" hint. Failing closed on a
  real-but-unlisted model (operator adds a line) is cheap; failing open on a fictional one (money
  burned, results mislabeled) is not.
* **served-model verification** (``verify_served_model``): providers get the model the API *actually
  served* back in every response. Comparing it to what we requested catches a silent substitution —
  a provider quietly serving a different model than asked — which the allowlist alone cannot see.

THE LIST LIVES IN CONFIG, NOT CODE: ``config/known_models.yaml`` (with a wheel-bundled fallback at
``podcast_scraper/data/known_models.yaml``). Adding a real model is a one-line YAML edit — no code
change. This module only loads and matches.

Local providers (ollama and the DGX/vLLM/tailnet paths) are intentionally NOT governed: users pull
arbitrary local weights and no cloud money is at stake. Only providers under ``governed_providers``
in the YAML are validated.
"""

from __future__ import annotations

import difflib
import logging
import re
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

_DEFAULT_CONFIGURED_PATH = "config/known_models.yaml"
_CONTAINER_FALLBACK_PATHS: Tuple[Path, ...] = (Path("/app/config/known_models.yaml"),)

# A trailing dated/pinned suffix a provider appends to a family id, e.g.
# ``claude-sonnet-4-5-20250219`` or ``mistral-medium-2505`` or ``...-latest``. Stripping it lets a
# pinned id match its family entry without listing every date. The digit run is 4-8 (YYMM / YYYYMM /
# YYYYMMDD) — NOT 1-3, so a family minor like the ``-5`` in ``claude-sonnet-4-5`` is never stripped.
_PINNED_SUFFIX = re.compile(r"-(?:\d{4,8}|latest|v\d+|preview|exp)$", re.IGNORECASE)

# (governed_providers, {provider: frozenset(model_ids)}) — cached by resolved path + mtime.
_cache_key: Optional[str] = None
_cache_mtime: Optional[float] = None
_cache_value: Optional[Tuple[FrozenSet[str], Dict[str, FrozenSet[str]]]] = None


class UnknownModelError(ValueError):
    """A cloud model id that is not on the governed allowlist. Raised BEFORE any API spend."""


def clear_known_models_cache() -> None:
    """Reset the loader cache (for tests / after editing the YAML at runtime)."""
    global _cache_key, _cache_mtime, _cache_value
    _cache_key = None
    _cache_mtime = None
    _cache_value = None


def _bundled_path() -> Optional[Path]:
    """Packaged fallback shipped in the wheel (pipeline containers)."""
    try:
        from importlib import resources

        ref = resources.files("podcast_scraper").joinpath("data/known_models.yaml")
        with resources.as_file(ref) as extracted:
            p = Path(extracted)
            return p if p.is_file() else None
    except (ImportError, FileNotFoundError, TypeError, OSError):  # pragma: no cover
        return None


def _resolve_path() -> Optional[Path]:
    """Find the known-models YAML: repo config/, then container path, then wheel-bundled copy."""
    for base in [Path.cwd(), *Path.cwd().parents]:
        c = base / _DEFAULT_CONFIGURED_PATH
        if c.is_file():
            return c
    for p in _CONTAINER_FALLBACK_PATHS:
        if p.is_file():
            return p
    return _bundled_path()


def _parse_payload(payload: Dict[str, Any]) -> Tuple[FrozenSet[str], Dict[str, FrozenSet[str]]]:
    governed_raw = payload.get("governed_providers") or []
    governed = frozenset(str(p).strip().lower() for p in governed_raw if str(p).strip())
    provs_raw = payload.get("providers") or {}
    models: Dict[str, FrozenSet[str]] = {}
    if isinstance(provs_raw, dict):
        for prov, ids in provs_raw.items():
            if not isinstance(ids, (list, tuple)):
                continue
            models[str(prov).strip().lower()] = frozenset(
                str(m).strip().lower() for m in ids if str(m).strip()
            )
    return governed, models


def _load() -> Tuple[FrozenSet[str], Dict[str, FrozenSet[str]]]:
    """Load (and cache) the governed-providers set and per-provider model allowlist from YAML."""
    global _cache_key, _cache_mtime, _cache_value
    path = _resolve_path()
    if path is None or yaml is None:
        # No config and no bundled fallback: govern nothing rather than crash the pipeline.
        logger.warning("known_models.yaml not found; model allowlist validation is DISABLED")
        return frozenset(), {}
    try:
        mtime = path.stat().st_mtime
    except OSError:  # pragma: no cover
        return frozenset(), {}
    key = str(path.resolve())
    if _cache_key == key and _cache_mtime == mtime and _cache_value is not None:
        return _cache_value
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("known_models.yaml root must be a mapping")
    value = _parse_payload(data)
    _cache_key, _cache_mtime, _cache_value = key, mtime, value
    logger.debug("Loaded known-model allowlist from %s", path)
    return value


def is_cloud_governed(provider: Optional[str]) -> bool:
    """True when this provider's model ids are allowlisted (cloud, billed). Local → False."""
    if not provider:
        return False
    governed, _ = _load()
    return provider.strip().lower() in governed


def normalize_model_id(model: str) -> str:
    """Drop a trailing dated/pinned suffix so a pinned id matches its family entry.

    ``claude-sonnet-4-5-20250219`` → ``claude-sonnet-4-5``; ``mistral-medium-latest`` →
    ``mistral-medium``.
    """
    m = (model or "").strip()
    stripped = _PINNED_SUFFIX.sub("", m)
    return stripped or m


def is_known_model(provider: str, model: str) -> bool:
    """True if ``model`` is a governed-provider model we recognise (or the provider is local).

    Local/uncosted providers are always True (not governed). For a cloud provider, the model must be
    listed for it in the YAML after normalising a dated/pinned suffix.
    """
    if not is_cloud_governed(provider):
        return True
    _, models = _load()
    known = models.get(provider.strip().lower(), frozenset())
    if not model:
        return False
    ml = model.strip().lower()
    return ml in known or normalize_model_id(ml) in known


def suggest_model(provider: str, model: str) -> Optional[str]:
    """Closest known id for a rejected model, for a helpful error ('did you mean …')."""
    if not is_cloud_governed(provider):
        return None
    _, models = _load()
    known = models.get(provider.strip().lower(), frozenset())
    matches = difflib.get_close_matches(
        (model or "").strip().lower(), sorted(known), n=1, cutoff=0.6
    )
    return matches[0] if matches else None


def validate_model_or_raise(provider: str, model: str, *, context: str = "") -> None:
    """Fail closed on an unknown cloud model, with a suggestion. No-op for local providers.

    ``context`` labels where the model came from (an arm/config/profile id) so the error points at
    the thing to fix.
    """
    if is_known_model(provider, model):
        return
    where = f" ({context})" if context else ""
    hint = suggest_model(provider, model)
    did_you_mean = f" Did you mean {hint!r}?" if hint else ""
    raise UnknownModelError(
        f"Unknown {provider} model {model!r}{where} — it is not on the governed allowlist "
        f"(config/known_models.yaml).{did_you_mean} "
        f"If this model is real and new, add it to config/known_models.yaml; if it is a typo, fix "
        f"the config. Refusing to spend on an unrecognised model."
    )


def iter_profile_model_refs(profile: Dict[str, Any]) -> "list[Tuple[str, str, str]]":
    """Extract ``(provider, model, field)`` tuples a profile pins, for allowlist validation.

    Covers the two shapes profiles use:
    * ``<provider>_summary_model`` keys — the prefix names the provider directly
      (``anthropic_summary_model: claude-haiku-4-5`` → ``("anthropic", "claude-haiku-4-5", ...)``).
    * a bare ``summary_model`` paired with ``summary_provider``, and the value-gate pair
      ``gi_value_gate_model`` / ``gi_value_gate_provider``.

    Only cloud-governed providers matter downstream; local ones are returned too but validation is a
    no-op for them. Missing/blank fields are skipped.
    """
    refs: "list[Tuple[str, str, str]]" = []
    if not isinstance(profile, dict):
        return refs

    def _add(provider: Any, model: Any, field: str) -> None:
        p, m = str(provider or "").strip(), str(model or "").strip()
        if p and m:
            refs.append((p, m, field))

    for key, value in profile.items():
        if isinstance(key, str) and key.endswith("_summary_model") and value:
            _add(key[: -len("_summary_model")], value, key)
    if profile.get("summary_model") and profile.get("summary_provider"):
        _add(profile["summary_provider"], profile["summary_model"], "summary_model")
    if profile.get("gi_value_gate_model") and profile.get("gi_value_gate_provider"):
        _add(
            profile["gi_value_gate_provider"],
            profile["gi_value_gate_model"],
            "gi_value_gate_model",
        )
    return refs


def validate_profile_or_raise(profile: Dict[str, Any], *, context: str = "") -> None:
    """Fail closed if any cloud model a profile pins is not on the allowlist.

    Validates every ``(provider, model)`` the profile references (see ``iter_profile_model_refs``).
    ``context`` should name the profile (file/preset) so the error points at what to fix.
    """
    for provider, model, field in iter_profile_model_refs(profile):
        label = f"{context}:{field}" if context else field
        validate_model_or_raise(provider, model, context=label)


def verify_served_model(provider: str, requested: str, served: Optional[str]) -> Optional[str]:
    """Return a human-readable mismatch reason if the API served a different model than requested.

    Returns None when they agree (after normalising dated suffixes — a provider pinning
    ``claude-sonnet-4-5-20250219`` for a requested ``claude-sonnet-4-5`` is a match, not a
    substitution). Only meaningful for cloud providers with a non-empty ``served`` id.
    """
    if not is_cloud_governed(provider):
        return None
    if not served or not requested:
        return None
    req = normalize_model_id(requested.strip().lower())
    srv = normalize_model_id(served.strip().lower())
    if req == srv or srv.startswith(req) or req.startswith(srv):
        return None
    return (
        f"{provider} served model {served!r} but we requested {requested!r} — the provider may "
        f"have silently substituted a different model. Verify the request and the served-model id."
    )
