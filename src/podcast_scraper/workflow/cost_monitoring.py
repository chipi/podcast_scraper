"""Real-time LLM cost emission and soft caps (#804)."""

from __future__ import annotations

import json
import logging
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

CostSoftCapAction = Literal["abort", "warn", "observe"]


def feed_url_for_cost_incident(feed: Any, cfg: Any) -> Optional[str]:
    """RSS URL for corpus incident rows (``RssFeed`` uses ``base_url``, not ``link``)."""
    if feed is not None:
        url = getattr(feed, "base_url", None) or getattr(feed, "link", None)
        if url:
            return str(url)
    rss = getattr(cfg, "rss_url", None)
    return str(rss) if rss else None


class CostCapExceeded(RuntimeError):
    """Pipeline run exceeded configured per-run soft cap."""

    def __init__(self, spent_usd: float, cap_usd: float) -> None:
        self.spent_usd = spent_usd
        self.cap_usd = cap_usd
        super().__init__(f"cost soft cap exceeded: ${spent_usd:.4f} > ${cap_usd:.4f}")


def run_cost_usd_from_pipeline_metrics(pipeline_metrics: Any) -> float:
    """Sum authoritative per-stage USD fields on :class:`workflow.metrics.Metrics`."""
    if pipeline_metrics is None:
        return 0.0
    return round(
        float(getattr(pipeline_metrics, "llm_transcription_cost_usd", 0.0) or 0.0)
        + float(getattr(pipeline_metrics, "llm_summarization_cost_usd", 0.0) or 0.0)
        + float(getattr(pipeline_metrics, "llm_speaker_detection_cost_usd", 0.0) or 0.0)
        + float(getattr(pipeline_metrics, "llm_cleaning_cost_usd", 0.0) or 0.0)
        + float(getattr(pipeline_metrics, "llm_gi_cost_usd", 0.0) or 0.0)
        + float(getattr(pipeline_metrics, "llm_kg_cost_usd", 0.0) or 0.0)
        + float(getattr(pipeline_metrics, "llm_bundled_clean_summary_cost_usd", 0.0) or 0.0),
        6,
    )


def emit_llm_cost_event(
    cfg: Any,
    *,
    provider: str,
    stage: str,
    model: str,
    estimated_cost_usd: float = 0.0,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    corpus_path: Optional[str] = None,
    feed_id: Optional[str] = None,
    run_id: Optional[str] = None,
    triggered_guardrail: bool = False,
    served_model: Optional[str] = None,
    response: Any = None,
    cached_input_tokens: Optional[int] = None,
    cache_write_tokens: Optional[int] = None,
    episode_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> None:
    """Emit one JSON usage/cost record per LLM call (Loki ``| json`` / Grafana #804).

    TOKENS ARE THE GROUND TRUTH; COST IS A PROJECTION. This record is emitted whenever the call
    produced any tokens — NOT gated on whether a price is known — because cost can be re-derived
    from tokens after the fact but tokens never recorded are gone forever. That gate is why OpenAI
    (no pricing row) and every gi/evidence/cleaning call used to log nothing; see token_accounting.

    Pass the raw ``response`` and this function extracts the normalised token usage itself (input,
    output, cached-read, cache-write) via ``extract_token_usage`` — so cache tokens are captured
    uniformly across providers without per-call-site plumbing. Explicit ``prompt_tokens`` /
    ``completion_tokens`` / ``cached_input_tokens`` still win when provided (back-compat).

    ``triggered_guardrail`` (ADR-100, #1003): True on calls whose response tripped a response-shape
    guardrail — lets cost-rollup surface paid-but-rejected spend.

    ``served_model``: the id the provider reported serving. A mismatch vs the requested ``model`` on
    a cloud provider is a SILENT SUBSTITUTION → a loud ``llm_model_substitution`` warning (runs even
    for zero-cost calls). ``episode_id`` / ``run_id`` / ``stage`` are the slicing dimensions.
    """
    # Extract token usage from the response when the caller passed it (fills cached/cache-write and
    # backfills prompt/completion if not given explicitly).
    if response is not None:
        try:
            from .token_accounting import extract_token_usage

            usage = extract_token_usage(provider, response)
            if prompt_tokens is None:
                prompt_tokens = usage.input_tokens
            if completion_tokens is None:
                completion_tokens = usage.output_tokens
            if cached_input_tokens is None:
                cached_input_tokens = usage.cached_input_tokens
            if cache_write_tokens is None:
                cache_write_tokens = usage.cache_write_tokens
            if not served_model:
                served_model = getattr(response, "model", None)
            if request_id is None:
                # The provider's own request id (openai/anthropic responses carry ``.id``) ties this
                # record to their dashboard AND lets the rollup de-dup a doubly-logged line — one
                # API call must count once.
                request_id = getattr(response, "id", None)
        except Exception:  # noqa: BLE001 - telemetry must never break a provider call
            pass

    # Served-model verification (runs regardless of cost/tokens so it also covers 0-cost calls).
    if served_model:
        from ..providers.known_models import verify_served_model

        mismatch = verify_served_model(provider, model, served_model)
        if mismatch:
            logger.warning(
                "%s",
                json.dumps(
                    {
                        "event_type": "llm_model_substitution",
                        "provider": provider,
                        "stage": stage,
                        "requested_model": model,
                        "served_model": served_model,
                        "run_id": run_id,
                        "detail": mismatch,
                    },
                    ensure_ascii=False,
                ),
            )

    # Emit whenever there is SOMETHING to record — tokens or positive cost. Only a truly empty call
    # (no tokens, no cost) is dropped. This is the fix: token telemetry is no longer cost-gated.
    has_tokens = bool(prompt_tokens or completion_tokens)
    cost = round(float(estimated_cost_usd or 0.0), 6)
    if not has_tokens and cost <= 0:
        return

    # Stamp the correlation dimensions (#1053) when the caller did not pass them, so a direct
    # provider emit (gi/cleaning) carries the same run/episode join keys as the record_provider_call
    # path. correlation is the established source; the per-episode fuse is a fallback.
    if run_id is None or episode_id is None:
        try:
            from ..utils import correlation

            if run_id is None:
                run_id = correlation.get_run_id()
            if episode_id is None:
                episode_id = correlation.get_episode_id()
        except Exception:  # noqa: BLE001
            pass
    if episode_id is None:
        try:
            from ..utils.llm_call_fuse import current_episode_id

            episode_id = current_episode_id()
        except Exception:  # noqa: BLE001
            episode_id = None

    fields: dict[str, Any] = {
        "provider": provider,
        "stage": stage,
        "operation": stage,  # alias: the function/operation this call served (gi/evidence/…)
        "model": model,
        "served_model": served_model or model,
        "request_id": request_id,
        "episode_id": episode_id,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_input_tokens": cached_input_tokens,
        "cache_write_tokens": cache_write_tokens,
        "estimated_cost_usd": cost,
        "corpus_path": corpus_path or getattr(cfg, "output_dir", None),
        "feed_id": feed_id or getattr(cfg, "rss_url", None),
        "run_id": run_id,
        "triggered_guardrail": bool(triggered_guardrail),
    }
    # Canonical event emission (ADR-119): the shared {ts, schema, event_type} envelope
    # plus this call's fields, one JSON line so Loki/Grafana ``| json`` parses directly.
    # Emitted through THIS module's logger (not the events logger) so existing consumers
    # and log filters keep the ``cost_monitoring`` logger name.
    from ..obs.events import emit_event

    line = emit_event("llm_cost", logger=logger, **fields)
    if line and getattr(cfg, "jsonl_metrics_echo_stdout", False):
        print(line, flush=True)

    # Optional Langfuse span — THE single emission point for LLM tracing, so gi/evidence/cleaning
    # (which never went through record_provider_call_cost) now get spans too, exactly once per call.
    # No-op unless the [langfuse] extra is installed AND both keys are set.
    try:
        import os as _os

        from ..utils.langfuse_tracing import emit_langfuse_span

        emit_langfuse_span(
            provider=provider,
            capability=stage,
            model=model,
            served_model=served_model or model,
            cost=cost if cost > 0 else None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_write_tokens=cache_write_tokens,
            request_id=request_id,
            run_seed=run_id or getattr(cfg, "output_dir", None),
            episode_id=episode_id,
            feed_id=feed_id or getattr(cfg, "rss_url", None),
            triggered_guardrail=bool(triggered_guardrail),
            env=_os.environ.get("PODCAST_ENV"),
        )
    except Exception as exc:  # noqa: BLE001 - tracing must never break a run
        logger.debug("langfuse span emission skipped: %s", exc)


def enforce_cost_soft_cap(cfg: Any, pipeline_metrics: Any) -> None:
    """Abort, warn, or observe when per-run spend crosses ``cost_soft_cap_usd_per_run``."""
    cap = getattr(cfg, "cost_soft_cap_usd_per_run", None)
    if cap is None or float(cap) <= 0:
        return
    spent = run_cost_usd_from_pipeline_metrics(pipeline_metrics)
    if spent <= float(cap):
        return
    action: CostSoftCapAction = getattr(cfg, "cost_soft_cap_action", "observe") or "observe"
    msg = f"cost soft cap: spent ${spent:.4f} exceeds ${float(cap):.4f} (action={action})"
    if action == "warn":
        logger.warning(msg)
        return
    if action == "observe":
        logger.info(msg)
        return
    raise CostCapExceeded(spent, float(cap))


def check_cost_soft_cap_at_stage(
    cfg: Any,
    pipeline_metrics: Any,
    *,
    stage: str,
    incident_log_path: Optional[str] = None,
    feed_url: Optional[str] = None,
) -> None:
    """Run soft-cap check after a pipeline stage that records LLM costs (#804)."""
    try:
        enforce_cost_soft_cap(cfg, pipeline_metrics)
    except CostCapExceeded as exc:
        if incident_log_path:
            from podcast_scraper.utils.corpus_incidents import append_corpus_incident

            append_corpus_incident(
                incident_log_path,
                scope="batch",
                category="policy",
                message=str(exc),
                exception_type="CostCapExceeded",
                stage=stage,
                feed_url=feed_url,
            )
        raise


def maybe_emit_run_cost_sentry_alert(cfg: Any, pipeline_metrics: Any) -> None:
    """Sentry warning when a single run exceeds ``cost_daily_alert_usd`` (#804).

    Note: per-run threshold until daily aggregation exists in metrics store.
    """
    threshold = float(getattr(cfg, "cost_daily_alert_usd", 10.0) or 10.0)
    spent = run_cost_usd_from_pipeline_metrics(pipeline_metrics)
    if spent < threshold:
        return
    try:
        import sentry_sdk
    except ImportError:
        return
    sentry_sdk.capture_message(
        f"Pipeline run estimated cost ${spent:.2f} exceeds run alert threshold "
        f"${threshold:.2f}",
        level="warning",
    )
    sentry_sdk.set_tag("cost_anomaly", "run_threshold")


# Back-compat alias (call sites from initial #804 land).
maybe_emit_daily_cost_sentry_alert = maybe_emit_run_cost_sentry_alert
