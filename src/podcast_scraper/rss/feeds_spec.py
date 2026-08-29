"""Structured corpus feed list: ``{ feeds: [...] }`` in YAML or JSON (#626).

Each entry is either a URL string or an object with required ``url`` plus optional
per-feed overrides (download resilience, timeouts, episode window). Validated with
Pydantic (same style as main :class:`~podcast_scraper.config.Config`).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypeVar
from urllib.parse import urlparse

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

_CfgT = TypeVar("_CfgT", bound=BaseModel)

logger = logging.getLogger(__name__)

# Canonical basename when the server resolves a path under the corpus root.
FEEDS_SPEC_DEFAULT_BASENAME = "feeds.spec.yaml"

# Keys allowed on a feed object beyond ``url`` (must exist on Config for model_copy).
RSS_FEED_ENTRY_OVERRIDE_KEYS: frozenset[str] = frozenset(
    {
        # A per-feed profile pin must survive the config-inline path too. Config's
        # ``_coerce_rss_urls_list`` pushes every ``rss_urls`` entry through this allowlist and
        # DROPS anything missing from it, silently — so a pin written inline resolved to the
        # corpus profile with no error at all, while the feeds.spec.yaml path honoured it.
        "profile",
        "user_agent",
        "timeout",
        "http_retry_total",
        "http_backoff_factor",
        "rss_retry_total",
        "rss_backoff_factor",
        "episode_retry_max",
        "episode_retry_delay_sec",
        "delay_ms",
        "host_request_interval_ms",
        "host_max_concurrent",
        "circuit_breaker_enabled",
        "circuit_breaker_failure_threshold",
        "circuit_breaker_window_seconds",
        "circuit_breaker_cooldown_seconds",
        "circuit_breaker_scope",
        "rss_conditional_get",
        "rss_cache_dir",
        "max_episodes",
        "episode_order",
        "episode_offset",
        "episode_since",
        "episode_until",
        "known_hosts",
        "show_centric",
        "diarization_min_segment_ms",
        "crosspromo_cue_patterns",
    }
)


def _validate_http_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"RSS URL must use http or https (got {parsed.scheme!r}): {url}")
    if not parsed.netloc:
        raise ValueError(f"RSS URL must include a valid hostname: {url}")
    return url


class RssFeedEntry(BaseModel):
    """One feed: ``url`` plus optional per-inner-run overrides (allowlist only)."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    url: str = Field(validation_alias=AliasChoices("url", "rss", "rss_url"))
    user_agent: Optional[str] = None
    timeout: Optional[int] = Field(default=None, ge=1)
    http_retry_total: Optional[int] = Field(default=None, ge=0, le=20)
    http_backoff_factor: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    rss_retry_total: Optional[int] = Field(default=None, ge=0, le=20)
    rss_backoff_factor: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    episode_retry_max: Optional[int] = Field(default=None, ge=0, le=10)
    episode_retry_delay_sec: Optional[float] = Field(default=None, ge=0.0, le=120.0)
    delay_ms: Optional[int] = Field(default=None, ge=0)
    host_request_interval_ms: Optional[int] = Field(default=None, ge=0, le=600_000)
    host_max_concurrent: Optional[int] = Field(default=None, ge=0, le=64)
    circuit_breaker_enabled: Optional[bool] = None
    circuit_breaker_failure_threshold: Optional[int] = Field(default=None, ge=1, le=100)
    circuit_breaker_window_seconds: Optional[int] = Field(default=None, ge=1, le=86400)
    circuit_breaker_cooldown_seconds: Optional[int] = Field(default=None, ge=1, le=86400)
    circuit_breaker_scope: Optional[Literal["feed", "host"]] = None
    rss_conditional_get: Optional[bool] = None
    rss_cache_dir: Optional[str] = None
    max_episodes: Optional[int] = None
    episode_order: Optional[Literal["newest", "oldest"]] = None
    episode_offset: Optional[int] = Field(default=None, ge=0)
    episode_since: Optional[str] = None
    episode_until: Optional[str] = None
    # Per-feed host names — override the show's roster hosts (Step B). For network feeds whose
    # author tag is the org and whose hosts never self-introduce, this is the cheapest way to
    # name a recurring host that would otherwise stay SPEAKER_NN. Overrides global known_hosts.
    # 2026-08-28: route ONE feed through a different deployment profile — the mechanism for
    # onboarding feeds onto the DGX (cloud_with_dgx_primary) while the proven feeds stay on
    # cloud_balanced, since a batch run otherwise applies one profile to every feed.
    profile: Optional[str] = None
    known_hosts: Optional[List[str]] = None
    # The show is the brand, not the host — an unnamed "Host" is expected here, not a failure.
    show_centric: Optional[bool] = None
    # Per-feed diarization squelch (ms). Override the global diarization_min_segment_ms — e.g. a
    # news-desk feed with no real brief cameos can squelch harder to kill phantom micro-speakers.
    diarization_min_segment_ms: Optional[int] = Field(default=None, ge=0, le=60000)
    # Per-feed opening cross-promo cue patterns (#1188). Extend the built-in cue set with a feed's
    # cross-promo phrasing at onboarding — the intended evolving surface for ads the defaults miss.
    crosspromo_cue_patterns: Optional[List[str]] = None

    @field_validator("url", mode="after")
    @classmethod
    def _check_url(cls, v: str) -> str:
        return _validate_http_url(v.strip())

    def override_update_dict(self) -> Dict[str, Any]:
        """Flat dict for ``Config.model_copy(update=...)`` — excludes ``url``."""
        data = self.model_dump(exclude_none=True)
        data.pop("url", None)
        return data


class FeedsSpecDocument(BaseModel):
    """Root document: ``{ feeds: [...] }`` plus optional ``_comment*`` keys (ignored)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    feeds: List[RssFeedEntry] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_feed_items(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        raw = out.get("feeds")
        if raw is None:
            out["feeds"] = []
            return out
        if not isinstance(raw, list):
            raise TypeError("feeds must be a list")
        normalized: List[Any] = []
        for item in raw:
            if item is None:
                continue
            if isinstance(item, str):
                s = item.strip()
                if s:
                    normalized.append({"url": s})
            elif isinstance(item, dict):
                normalized.append(dict(item))
            else:
                raise TypeError("each feeds[] entry must be a string or object")
        out["feeds"] = normalized
        return out


def load_feeds_spec_file(path: str | Path) -> FeedsSpecDocument:
    """Load and validate ``feeds.spec.{yaml,yml,json}`` (root object with ``feeds`` array only).

    Callers must pass a path already bound to a trusted corpus root (HTTP routes use
    ``normpath_if_under_root`` before calling); the CLI may pass explicit local paths.
    """
    p = Path(path).expanduser()
    # codeql[py/path-injection] -- caller supplies trusted path (Type 1; CODEQL_DISMISSALS.md).
    if not p.is_file():
        raise ValueError(f"Feeds spec file not found: {p}")
    suffix = p.suffix.lower()
    # codeql[py/path-injection] -- same as is_file guard above.
    text = p.read_text(encoding="utf-8")
    if suffix == ".json":
        raw = json.loads(text)
    elif suffix in (".yaml", ".yml"):
        raw = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported feeds spec extension (use .json, .yaml, .yml): {p}")
    if not isinstance(raw, dict):
        raise ValueError("Feeds spec must be a JSON/YAML object at the top level")
    unknown = [k for k in raw if k != "feeds" and not str(k).startswith("_")]
    if unknown:
        raise ValueError(
            "Unknown top-level keys in feeds spec (only `feeds` and `_comment*` allowed): "
            + ", ".join(sorted(unknown))
        )
    return FeedsSpecDocument.model_validate(raw)


def dump_feeds_spec_yaml(doc: FeedsSpecDocument) -> str:
    """Serialize document to YAML (default for viewer/API writes)."""
    items: List[Any] = []
    for e in doc.feeds:
        d = e.model_dump(mode="json", exclude_none=True)
        if set(d.keys()) == {"url"}:
            items.append(d["url"])
        else:
            items.append(d)
    payload: Dict[str, Any] = {"feeds": items}
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, default_flow_style=False)


def append_normalized_feed_items(bucket: List[dict], items: Optional[List[Any]]) -> None:
    """Append coerced feed dicts to *bucket*, deduping by ``url`` (first wins)."""
    if not items:
        return
    seen = {str(b.get("url", "")).strip() for b in bucket if b.get("url")}
    for u in items:
        if u is None:
            continue
        if isinstance(u, str):
            t = u.strip()
            if not t or t in seen:
                continue
            seen.add(t)
            bucket.append({"url": t})
        elif isinstance(u, dict):
            url = str(u.get("url") or u.get("rss") or u.get("rss_url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            d: Dict[str, Any] = {"url": url}
            for k, v in u.items():
                if k in ("url", "rss", "rss_url"):
                    continue
                if k in RSS_FEED_ENTRY_OVERRIDE_KEYS:
                    d[k] = v
            bucket.append(d)
        else:
            raise TypeError("feeds/rss_urls entries must be a string or a mapping with url")


def _is_routing_field(name: str) -> bool:
    """Does this field decide WHERE/WITH-WHAT a stage runs (as opposed to deployment policy)?

    Routing is owned by whichever profile the feed runs under; deployment policy (caps,
    storage, retry) is owned by the corpus and survives a pin. Matching on shape rather than
    an enumerated list on purpose: a hand-written list of provider-bound fields was wrong
    three times in one day, and a new ``*_provider`` added next month would silently miss it.
    """
    return name.endswith(("_provider", "_providers", "_model", "_models", "_api_base"))


def merge_feed_entry_into_config(cfg: _CfgT, entry: RssFeedEntry) -> _CfgT:
    """Return a new Config with ``rss_url`` set and per-feed overrides applied.

    A per-feed ``profile:`` is RESOLVED, not copied. ``model_copy(update=...)`` does not run
    validators, so assigning the profile name alone would relabel the config and route
    nothing — the feed would still transcribe wherever the batch-level profile pointed, with
    a log line claiming otherwise. Instead the named profile's registry+YAML layers are
    resolved through the same cascade ``Config._resolve_profile`` uses and applied UNDER the
    entry's own explicit overrides, so a feed can say "use the DGX profile, but keep my
    max_episodes".
    """
    updates: Dict[str, Any] = {"rss_url": entry.url, "rss_urls": None}
    updates.update(entry.override_update_dict())
    profile_name = updates.pop("profile", None)
    if profile_name and getattr(cfg, "profile_overrides_feed_pins", False):
        # A per-request override outranks a feed pin (#1872). The batch loop cannot infer
        # that from --profile alone — the corpus default arrives the same way — so the API
        # sets this flag explicitly when an operator chose the profile for THIS run.
        logger.info(
            "feeds.spec: feed %s pin %r ignored — this run's profile override wins",
            entry.url,
            profile_name,
        )
        profile_name = None
    if not profile_name:
        return cfg.model_copy(update=updates)

    from podcast_scraper.config import resolve_profile_layers

    profile_name = str(profile_name)
    _, resolved = resolve_profile_layers(
        profile_name, dgx_tailnet_host=getattr(cfg, "dgx_tailnet_host", None)
    )
    if not resolved:
        logger.warning(
            "feeds.spec: feed %s names profile %r, which matched no registry preset or "
            "config/profiles/ file — the feed runs on the batch profile instead",
            entry.url,
            profile_name,
        )
        return cfg.model_copy(update=updates)

    # Rebuild through Config so the pin goes through the SAME cascade a top-level ``profile:``
    # uses — registry < profile YAML < explicit fields — and so validators, the audio-preset
    # merge and the resilience derivations all run. The earlier model_copy version skipped
    # every one of those: it layered the profile OVER the operator's explicit values (a
    # precedence INVERSION — a pinned prod feed silently swapped litellm_api_base for the
    # profile YAML's default) and, because model_copy runs no validators, a mis-keyed pin was
    # caught nowhere and died mid-run after other feeds had already spent money.
    #
    # Which of the base config's fields count as "explicit"? ``model_fields_set`` is not the
    # answer on its own: the base profile's own merge marks ~80 fields as set, so replaying
    # them all would pin the OLD profile's routing on top of the new one — the opposite
    # inversion. So subtract the base profile's contribution: a field still holding exactly
    # what the base profile gave it is profile-derived and must yield to the new profile;
    # anything else was set by the operator YAML or the CLI and must win, exactly as it would
    # top-level. (A field the operator set to the same value the base profile uses is
    # indistinguishable from profile-derived and yields — the values agree, so only a
    # differing new profile changes anything, which is the pin doing its job.)
    # Compare against the base profile's EFFECTIVE config, not its raw layers.
    #
    # The raw-layer version of this was wrong in three ways at once, all found by an
    # adversarial sweep over every ordered profile pair (231 violations):
    #   * ~20 profiles express routing with the nested ``transcription:`` sugar, which the
    #     layer resolver does NOT flatten — flattening happens inside Config's validator — so
    #     the effective value never equalled the raw layer and every such field was replayed
    #     over the pin. cloud_qwen -> cloud_balanced silently kept whisper instead of deepgram.
    #   * ADR-122 derives resilience_run_context / resilience_failure_strategy from the profile
    #     NAME; they appear in no layer at all, so a reprocess pin ran with the wrong failure
    #     strategy.
    #   * a materialized flat field can disagree with its own nested block, so "raw" is not
    #     even a single answer.
    # Building the base profile once through the same validators removes the whole class: the
    # comparison is now effective-vs-effective, which is the only apples-to-apples there is.
    base_effective: Any = None
    base_profile = getattr(cfg, "profile", None)
    if base_profile:
        try:
            base_effective = type(cfg).model_validate(
                {"rss_url": entry.url, "profile": str(base_profile)}
            )
        except Exception:  # noqa: BLE001 — fall back to "everything is explicit" (safe)
            base_effective = None

    # What does the PIN declare? The raw layers cannot answer that — the same raw-vs-effective
    # trap already fixed on the base side, left in place on this one. resolve_profile_layers
    # does not flatten the nested ``transcription:`` sugar, does not include registry-derived
    # or name-derived fields, and so under-reports what a profile actually contributes: pins
    # to defaults-based profiles kept the BASE's diarization, KG and ASR-fallback routing.
    #
    # Effective-vs-bare answers it for every contribution mechanism at once: build the pin
    # alone, build a config with no profile, and every field that differs is something the pin
    # genuinely supplies. Silence stays distinguishable from a default, which is what the raw
    # layers were being used for.
    pin_effective: Any = None
    pin_declares: set[str] = set()
    try:
        pin_effective = type(cfg).model_validate({"rss_url": entry.url, "profile": profile_name})
        bare = type(cfg).model_validate({"rss_url": entry.url})
        for name in type(cfg).model_fields:
            if getattr(pin_effective, name, None) != getattr(bare, name, None):
                pin_declares.add(name)
    except Exception:  # noqa: BLE001 — fall back to the layer view rather than mis-yielding
        pin_effective = None

    # UNION with the raw layers. Neither signal is complete on its own and they fail in
    # opposite directions: effective-vs-bare misses a profile that declares a value which
    # happens to EQUAL the model default (cloud_with_dgx_primary's ``diarize: true``), while
    # the raw layers miss everything contributed by nested sugar, the registry, or the profile
    # NAME. Using either alone leaves a class of pins silently un-applied.
    layers_only, _ = resolve_profile_layers(
        profile_name, dgx_tailnet_host=getattr(cfg, "dgx_tailnet_host", None)
    )
    pin_declares |= {k for k in layers_only if k in type(cfg).model_fields}
    nested_tx = layers_only.get("transcription")
    if isinstance(nested_tx, dict):
        if "primary" in nested_tx:
            pin_declares.add("transcription_provider")
        if "fallback" in nested_tx:
            pin_declares.add("transcription_fallback_provider")
    _MISSING = object()
    operator_explicit: Dict[str, Any] = {}
    for field in getattr(cfg, "model_fields_set", set()):
        if field == "profile":
            continue
        value = getattr(cfg, field, None)
        # Yield ONLY when the value is indistinguishable from what the base profile itself
        # produces AND the pinned profile supplies its own — i.e. the pin genuinely re-routes
        # this field. Without the second condition an operator setting that merely coincided
        # with the base profile's value fell all the way to the model default (silent config
        # loss); without the first, the operator's explicit values are overwritten by profile
        # defaults (the precedence inversion this rebuild exists to fix).
        base_value = getattr(base_effective, field, _MISSING) if base_effective else _MISSING
        profile_derived = base_value is not _MISSING and base_value == value
        if profile_derived and (field in pin_declares or _is_routing_field(field)):
            # ROUTING comes wholly from the pin; DEPLOYMENT POLICY overlays from the corpus.
            #
            # Yielding only on ``field in pin_declares`` left the base profile's routing in
            # place wherever the pin was silent — so an airgapped pin kept the corpus's cloud
            # kg_extraction_provider, and a cloud pin kept the corpus's DGX ASR ladder. A
            # profile that does not mention a stage is not endorsing whatever the previous
            # profile chose for it; it is running that stage on ITS OWN defaults, which is
            # exactly what the profile resolves to standalone.
            #
            # Non-routing settings (cost caps, storage backend, retry policy, incident log)
            # are deployment decisions the corpus owns, and a pin must not reset them — that
            # distinction is the whole reason this is a rule and not "drop everything".
            continue
        operator_explicit[field] = value

    # Stage coupling: a model name belongs to the provider that serves it. When the pin moves
    # a stage's PROVIDER, any base-derived model field bound to that stage must go with it —
    # otherwise the pin produces an incoherent pairing. Demonstrated: pinning
    # bakeoff_gemini_flash onto a cloud_balanced corpus gave summary_provider=gemini with
    # summary_model='podcast-flash-0731', a LiteLLM alias meaningless to Gemini, because the
    # pin declares the provider but not the model and the base's model looked "operator
    # explicit". Deployment-scoped settings (cost caps, storage, retry) are NOT coupled and
    # still survive — a pin overlays routing, it does not reset the deployment.
    _STAGE_COUPLING: Dict[str, tuple[str, ...]] = {
        "summary_provider": (
            "summary_model",
            "litellm_summary_model",
            "gemini_summary_model",
            "deepseek_summary_model",
            "qwen_summary_model",
            "openai_summary_model",
            "anthropic_summary_model",
            "ollama_summary_model",
            "vllm_summary_model",
            "litellm_api_base",
            "vllm_api_base",
            "ollama_api_base",
        ),
        "transcription_provider": (
            "transcription_model",
            "whisper_model",
            "deepgram_model",
            "dgx_whisper_model",
            "openai_transcription_model",
            "groq_transcription_model",
            "transcription_fallback_provider",
            "transcription_fallback_providers",
            "transcription_coverage_failover_provider",
            "transcription_coverage_failover_model",
        ),
        "diarization_provider": (
            "diarization_model",
            "dgx_diarize_model",
            "deepgram_diarization_model",
            "diarization_fallback_providers",
        ),
        "speaker_detector_provider": ("speaker_llm_model", "ner_model"),
        "kg_extraction_provider": ("kg_extraction_model",),
        "gi_value_gate_provider": ("gi_value_gate_model",),
        "quote_extraction_provider": ("quote_extraction_model",),
        "entailment_provider": ("entailment_model",),
        "vector_embedding_provider": ("vector_embedding_model", "embed_model"),
    }
    for provider_field, model_fields in _STAGE_COUPLING.items():
        # Compare EFFECTIVE providers. Reading the pin's provider from raw layers missed every
        # profile that routes via nested sugar or registry defaults — which is most of them —
        # so the coupling silently never fired for those pins.
        pinned_provider = getattr(pin_effective, provider_field, None) if pin_effective else None
        base_provider = getattr(base_effective, provider_field, None) if base_effective else None
        if pinned_provider is None or base_provider is None:
            continue
        if str(pinned_provider) == str(base_provider):
            continue  # the stage did not move; its model may legitimately carry over
        for model_field in (provider_field, *model_fields):
            base_model = (
                getattr(base_effective, model_field, _MISSING) if base_effective else _MISSING
            )
            if (
                model_field in operator_explicit
                and base_model is not _MISSING
                and operator_explicit[model_field] == base_model
            ):
                del operator_explicit[model_field]

    payload: Dict[str, Any] = {**operator_explicit, **updates, "profile": profile_name}
    # Build through type(cfg), not a hard-coded Config: this helper is generic over the
    # config type it is handed, and a subclass must come back as itself.
    sub = type(cfg).model_validate(payload)
    logger.info(
        "feeds.spec: feed %s routed through profile %r (transcription=%s summary=%s)",
        entry.url,
        profile_name,
        getattr(sub, "transcription_provider", "?"),
        getattr(sub, "summary_provider", "?"),
    )
    return sub
