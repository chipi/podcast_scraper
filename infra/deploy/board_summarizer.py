#!/usr/bin/env python3
"""Board summarizer sidecar (RFC-127): the board summaries, written in the background.

The first LLM call outside the pipeline. It lives here, in a sidecar on the player stack, and not
in the player API, for one reason: the API serves requests, and nothing that serves a request may
be able to call a model. The API only reads what this loop wrote
(``board_summaries.summary_for_brief``).

Each cycle:

1. Reads the kill switch (``board_summaries_settings.json`` in the appdata root). Off → skip.
2. Builds the provider from the profile, exactly as the pipeline does
   (``Config(profile=...)`` → ``create_summarization_provider``). The profile is
   ``BOARD_SUMMARY_PROFILE``, default ``prod_dgx_full``: the model on our DGX, costing nothing.
   ``initialize()`` checks the DGX serves that model; if it cannot be reached the cycle ends and
   the next one tries again (the ``:8003`` slot is up only in the GPU's research mode).
3. Runs one ``board_summaries.sweep``: only boards that changed, have been quiet, were not
   summarised in the last day; per-user and per-sweep caps; every call priced first with the
   pipeline's pricing table against a budget of zero by default, so a paid profile stops the job.

The loop and its heartbeat follow ``digest_scheduler.py``: interval-aligned wakes, one cycle at
start, a bad cycle never kills the loop, a heartbeat file for the container healthcheck.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, NoReturn, Optional

HEARTBEAT = Path(os.environ.get("BOARD_SUMMARY_HEARTBEAT_FILE", "/heartbeat/tick"))
CORPUS_ROOT = Path(os.environ.get("BOARD_SUMMARY_CORPUS_ROOT", "/app/output"))
DATA_DIR = Path(os.environ.get("APP_DATA_DIR", "/app/appdata"))
PROFILE = os.environ.get("BOARD_SUMMARY_PROFILE", "prod_dgx_full").strip() or "prod_dgx_full"
INTERVAL_S = int(os.environ.get("BOARD_SUMMARY_INTERVAL_SECONDS", "3600"))
# Off the digest's offset, so the two sidecars never wake in the same second.
OFFSET_S = int(os.environ.get("BOARD_SUMMARY_INTERVAL_OFFSET_SECONDS", "900"))


def _log(msg: str) -> None:
    print(f"[board-summarizer] {msg}", flush=True)


def _log_json(**fields: object) -> None:
    print("[board-summarizer-json] " + json.dumps(fields, sort_keys=True, default=str), flush=True)


def _beat() -> None:
    try:
        HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
        HEARTBEAT.write_text(str(int(time.time())), encoding="utf-8")
    except OSError as exc:
        _log(f"heartbeat write failed: {exc}")


def _provider() -> tuple[Any, Any]:
    """The profile's provider, built and initialised as the pipeline builds it."""
    from podcast_scraper import config
    from podcast_scraper.summarization.factory import create_summarization_provider

    cfg = config.Config(profile=PROFILE)
    provider = create_summarization_provider(cfg)
    provider.initialize()
    return cfg, provider


def _pricer(cfg: Any) -> Callable[[str, int], Optional[float]]:
    """``(prompt, max_tokens) → USD`` from the pipeline's pricing table; None when unpriced."""
    from podcast_scraper.workflow.helpers import calculate_provider_cost

    def price(prompt: str, max_tokens: int) -> Optional[float]:
        return calculate_provider_cost(
            cfg=cfg,
            provider_type=str(cfg.summary_provider),
            capability="summarization",
            model=str(cfg.summary_model),
            prompt_tokens=max(1, len(prompt) // 3),
            completion_tokens=max_tokens,
        )

    return price


def _run_once() -> None:
    from podcast_scraper.server import board_summaries

    ts = int(time.time())
    if not board_summaries.summaries_enabled(DATA_DIR):
        _log("switched off (board_summaries_settings.json); nothing summarised")
        _log_json(event="board_summary.tick", stopped="switched_off", ts=ts)
        return
    try:
        cfg, provider = _provider()
    except Exception as exc:  # noqa: BLE001 — the model is away; the next cycle tries again
        _log(f"model not reachable on profile {PROFILE}: {exc!r}")
        _log_json(event="board_summary.tick", stopped="model_unreachable", profile=PROFILE, ts=ts)
        return
    complete = getattr(provider, "complete_prose", None)
    if complete is None:
        _log(f"profile {PROFILE}: provider {cfg.summary_provider} has no complete_prose")
        _log_json(event="board_summary.tick", stopped="unsupported_provider", ts=ts)
        return
    limits = board_summaries.Limits.from_env()
    result = board_summaries.sweep(
        CORPUS_ROOT,
        DATA_DIR,
        complete=lambda prompt: complete(
            prompt, stage=board_summaries.STAGE, max_tokens=limits.max_tokens
        ),
        price=_pricer(cfg),
        model=str(cfg.summary_model),
        limits=limits,
    )
    for user_id, board_id in result.summarised:
        _log_json(event="board_summary.written", user=user_id, board=board_id, ts=ts)
    _log_json(
        event="board_summary.tick",
        profile=PROFILE,
        model=str(cfg.summary_model),
        counts=dict(result.counts),
        stopped=result.stopped,
        ts=ts,
    )
    _log(
        f"tick: summarised {len(result.summarised)} "
        f"[{', '.join(f'{k}={v}' for k, v in sorted(result.counts.items()))}]"
        + (f" stopped={result.stopped}" if result.stopped else "")
    )


def _sleep_to_next_interval(sleep: Callable[[float], object] = time.sleep) -> None:
    now = time.time()
    sleep(INTERVAL_S - (now % INTERVAL_S) + OFFSET_S)


def _cycle() -> None:
    try:
        _run_once()
    except Exception as exc:  # noqa: BLE001 — a bad cycle must never kill the loop
        _log(f"cycle error: {exc!r}")
    _beat()


def main() -> NoReturn:
    _log(
        f"start: corpus={CORPUS_ROOT} data_dir={DATA_DIR} profile={PROFILE} "
        f"interval={INTERVAL_S}s offset={OFFSET_S}s heartbeat={HEARTBEAT}"
    )
    while True:
        _cycle()
        _sleep_to_next_interval()


if __name__ == "__main__":
    main()
