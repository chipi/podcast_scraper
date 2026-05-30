"""Real-time LLM cost emission and soft caps (#804)."""

from __future__ import annotations

import json
import logging
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

CostSoftCapAction = Literal["abort", "warn", "observe"]


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
    estimated_cost_usd: float,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    corpus_path: Optional[str] = None,
    feed_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> None:
    """Structured log line for Loki / Grafana (GitHub #804 D1)."""
    if estimated_cost_usd <= 0:
        return
    event = {
        "event_type": "llm_cost",
        "provider": provider,
        "stage": stage,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "estimated_cost_usd": round(float(estimated_cost_usd), 6),
        "corpus_path": corpus_path or getattr(cfg, "output_dir", None),
        "feed_id": feed_id or getattr(cfg, "rss_url", None),
        "run_id": run_id,
    }
    line = json.dumps(event, ensure_ascii=False)
    logger.info("llm_cost_event %s", line)
    if getattr(cfg, "jsonl_metrics_echo_stdout", False):
        print(line, flush=True)


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


def maybe_emit_daily_cost_sentry_alert(cfg: Any, pipeline_metrics: Any) -> None:
    """Sentry warning when run spend exceeds ``cost_daily_alert_usd`` (#804 D4)."""
    threshold = float(getattr(cfg, "cost_daily_alert_usd", 10.0) or 10.0)
    spent = run_cost_usd_from_pipeline_metrics(pipeline_metrics)
    if spent < threshold:
        return
    try:
        import sentry_sdk
    except ImportError:
        return
    sentry_sdk.capture_message(
        f"Pipeline run estimated cost ${spent:.2f} exceeds daily alert threshold ${threshold:.2f}",
        level="warning",
    )
    sentry_sdk.set_tag("cost_anomaly", "run_threshold")
