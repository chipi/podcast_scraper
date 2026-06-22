"""#1046 sniff-pass orchestrator for tailnet_dgx_whisper.

When ``cfg.dgx_whisper_sniff_model`` is set, this module:

1. Runs the sniff (cheap) model on the episode first via ``model_override``.
2. Counts PERSON + ORG entities on the sniff transcript via spaCy NER.
3. If the count meets ``cfg.dgx_whisper_sniff_gate_min_entities`` it runs the
   deep (large) model on the same audio and returns its transcript.
4. Otherwise it keeps the sniff transcript and skips the expensive deep pass.

Both models live in the same speaches container on DGX (#1046 measurement
pass 1: 3.26 GiB total resident for ``small.en`` + ``large-v3`` — well
within the 121 GiB unified-memory budget); the only added cost when the
gate fires is the sniff pass itself (~1/3rd of large at the corpus-mean
ratio of ~2.7×). When the gate does NOT fire we pay sniff-only.

Activation requires:

- ``cfg.transcription_provider == "tailnet_dgx_whisper"`` — only this
  provider supports per-call ``model_override``.
- ``cfg.dgx_whisper_sniff_model`` non-empty.
- ``spacy`` + ``en_core_web_sm`` importable — else the gate fails OPEN
  (runs deep), preserving quality.

When activation conditions aren't met the caller's regular deep-only path
runs unchanged; this module is a no-op.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Optional

from .. import config as _config_mod

logger = logging.getLogger(__name__)

# Gate decisions tagged onto the returned dict so downstream stages
# (artifact writers, the eval harness) can attribute behaviour without
# re-running NER. Strings live here so callers can match safely.
GATE_DECISION_RAN_DEEP = "ran_deep"
GATE_DECISION_KEPT_SNIFF = "kept_sniff"
GATE_DECISION_DISABLED = "disabled"
GATE_DECISION_NER_UNAVAILABLE = "ner_unavailable"

# spaCy labels that count toward the gate. PERSON + ORG are the
# discriminating signal for podcast content (host/guest names, brands).
# GPE/LOC are noisy — common-noun hallucinations on small.en transcripts
# inflate them in ways that hurt gate accuracy.
GATE_NER_LABELS = ("PERSON", "ORG")


def is_enabled(cfg: _config_mod.Config) -> bool:
    """True iff the operator has wired the sniff-pass gate on."""
    return cfg.transcription_provider == "tailnet_dgx_whisper" and bool(
        (cfg.dgx_whisper_sniff_model or "").strip()
    )


@lru_cache(maxsize=1)
def _load_nlp():
    """Lazy-load spaCy + en_core_web_sm once per process.

    Returns None if either is missing — callers must handle by falling open.
    """
    try:
        import spacy

        return spacy.load("en_core_web_sm")
    except (ImportError, OSError) as exc:
        logger.warning(
            "[#1046] spaCy/en_core_web_sm unavailable (%s); sniff-pass gate "
            "falls open to deep-only.",
            exc,
        )
        return None


def count_gate_entities(text: str) -> Optional[int]:
    """Count PERSON+ORG entities in ``text``; ``None`` if NER unavailable."""
    nlp = _load_nlp()
    if nlp is None:
        return None
    doc = nlp(text)
    return sum(1 for ent in doc.ents if ent.label_ in GATE_NER_LABELS)


def transcribe_with_sniff_gate(
    *,
    media_path: str,
    cfg: _config_mod.Config,
    provider: Any,
    pipeline_metrics: Any | None = None,
    episode_duration_seconds: Optional[float] = None,
    call_metrics: Any | None = None,
) -> tuple[dict[str, Any], float]:
    """Run sniff → gate → conditional-deep orchestration.

    Returns the same shape as ``provider.transcribe_with_segments`` (a
    ``(result_dict, elapsed_seconds)`` tuple) with an extra ``sniff_gate``
    sub-dict on ``result_dict`` describing the decision taken.

    Caller is responsible for checking ``is_enabled(cfg)`` first. When
    invoked with the gate disabled this function still runs deep-only and
    tags ``decision == "disabled"`` for symmetry — downstream artifact
    writers can therefore always expect the ``sniff_gate`` sub-dict.
    """
    sniff_model = (cfg.dgx_whisper_sniff_model or "").strip()
    deep_model = cfg.dgx_whisper_model
    threshold = cfg.dgx_whisper_sniff_gate_min_entities

    def _deep_only(decision_tag: str, extra: dict[str, Any]) -> tuple[dict[str, Any], float]:
        result, elapsed = provider.transcribe_with_segments(
            media_path,
            language=cfg.language,
            pipeline_metrics=pipeline_metrics,
            episode_duration_seconds=episode_duration_seconds,
            call_metrics=call_metrics,
        )
        result["sniff_gate"] = {"decision": decision_tag, "deep_model": deep_model, **extra}
        return result, elapsed

    if not sniff_model:
        # Caller didn't gate on is_enabled; preserve deep-only behaviour.
        return _deep_only(GATE_DECISION_DISABLED, {})

    sniff_result, sniff_elapsed = provider.transcribe_with_segments(
        media_path,
        language=cfg.language,
        pipeline_metrics=pipeline_metrics,
        episode_duration_seconds=episode_duration_seconds,
        call_metrics=call_metrics,
        model_override=sniff_model,
    )
    entity_count = count_gate_entities(str(sniff_result.get("text", "")))

    if entity_count is None:
        # NER broken — fall open to deep to preserve quality. Total wall
        # clock is sniff + deep; the alternative (kept_sniff) would be
        # silently degraded transcription, which is worse than slow.
        deep_result, deep_elapsed = provider.transcribe_with_segments(
            media_path,
            language=cfg.language,
            pipeline_metrics=pipeline_metrics,
            episode_duration_seconds=episode_duration_seconds,
            call_metrics=call_metrics,
        )
        deep_result["sniff_gate"] = {
            "decision": GATE_DECISION_NER_UNAVAILABLE,
            "sniff_model": sniff_model,
            "sniff_elapsed_s": round(sniff_elapsed, 3),
            "deep_model": deep_model,
            "threshold": threshold,
        }
        return deep_result, sniff_elapsed + deep_elapsed

    if entity_count < threshold:
        sniff_result["sniff_gate"] = {
            "decision": GATE_DECISION_KEPT_SNIFF,
            "entity_count": entity_count,
            "threshold": threshold,
            "sniff_model": sniff_model,
            "deep_model_skipped": deep_model,
        }
        logger.info(
            "[#1046] sniff gate did NOT fire (entities=%d < threshold=%d); "
            "keeping sniff transcript (model=%s).",
            entity_count,
            threshold,
            sniff_model,
        )
        return sniff_result, sniff_elapsed

    deep_result, deep_elapsed = provider.transcribe_with_segments(
        media_path,
        language=cfg.language,
        pipeline_metrics=pipeline_metrics,
        episode_duration_seconds=episode_duration_seconds,
        call_metrics=call_metrics,
    )
    deep_result["sniff_gate"] = {
        "decision": GATE_DECISION_RAN_DEEP,
        "entity_count": entity_count,
        "threshold": threshold,
        "sniff_model": sniff_model,
        "deep_model": deep_model,
        "sniff_elapsed_s": round(sniff_elapsed, 3),
        "deep_elapsed_s": round(deep_elapsed, 3),
    }
    logger.info(
        "[#1046] sniff gate fired (entities=%d >= threshold=%d); ran deep " "(sniff=%s, deep=%s).",
        entity_count,
        threshold,
        sniff_model,
        deep_model,
    )
    return deep_result, sniff_elapsed + deep_elapsed
