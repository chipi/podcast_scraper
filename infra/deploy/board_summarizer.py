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
   It calls the DGX exactly as the pipeline does. ``initialize()``, which asks the DGX which model
   it serves, runs only before the first call a sweep makes; if the DGX cannot be reached the
   sweep ends and the next one tries again (the ``:8003`` slot is up only in research mode).
3. Runs one ``board_summaries.sweep``: only boards that changed, have been quiet, were not
   summarised in the last day; per-user and per-sweep caps; every call priced first with the
   pipeline's pricing table against a budget of zero by default, so a paid profile stops the job.
   The call itself is the provider's own ``summarize`` with the board use case
   (``providers/common/summary_use_case.py``): the episode summary's flow, its prompts excepted.

Before the summaries, each cycle also runs the reference pass (#2345,
``server/reference_integrity.py``): listeners' stored theme, storyline, topic and person ids are
re-keyed when the corpus renamed them and flagged when they are gone. No model is involved, so it
runs whether or not summaries are switched on.

The loop and its heartbeat follow ``digest_scheduler.py``: interval-aligned wakes, one cycle at
start, a bad cycle never kills the loop, a heartbeat file for the container healthcheck.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, NoReturn, Optional

HEARTBEAT = Path(os.environ.get("BOARD_SUMMARY_HEARTBEAT_FILE", "/heartbeat/tick"))
CORPUS_ROOT = Path(os.environ.get("BOARD_SUMMARY_CORPUS_ROOT", "/app/output"))
DATA_DIR = Path(os.environ.get("APP_DATA_DIR", "/app/appdata"))
PROFILE = os.environ.get("BOARD_SUMMARY_PROFILE", "prod_dgx_full").strip() or "prod_dgx_full"
INTERVAL_S = int(os.environ.get("BOARD_SUMMARY_INTERVAL_SECONDS", "300"))
# Under the interval, and off the digest's (120 s past the hour): with a 300 s interval this wakes
# at :01, :06, :11 …, never with the digest. A sweep with nothing changed reads files only.
OFFSET_S = int(os.environ.get("BOARD_SUMMARY_INTERVAL_OFFSET_SECONDS", "60"))
_LOGGER = logging.getLogger("podcast_scraper.board_summarizer")


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
    """The profile's provider, built as the pipeline builds it — with no failover, and NOT yet
    initialised: building it touches no network, initialising it does (the vLLM provider asks the
    DGX which model it serves). A sweep with nothing to summarise must not talk to the model
    server at all, so ``_BoardUseCase`` initialises it on the first call it actually makes.

    Every call is priced against this provider before it is made, against a budget of zero;
    failing over would make it on a provider nobody priced (a free DGX call becoming a paid cloud
    one). So the ladder is emptied here, and ``_run_once`` refuses a provider that still comes
    back wrapped for failover (the older ``degradation_policy`` route): fail closed."""
    from podcast_scraper import config
    from podcast_scraper.summarization.factory import create_summarization_provider

    cfg = config.Config(profile=PROFILE, summary_fallback_providers=[])
    return cfg, create_summarization_provider(cfg)


class _BoardUseCase:
    """The board summary as a use case of the provider's own ``summarize`` (RFC-127): the same
    retries, metrics, cost recording, guardrails and prompt metadata as the episode summary. The
    prompt names come from the profile (``board_summary_*_prompt``), resolved under the provider's
    prompt folder with the ``prompts/shared/`` fallback."""

    def __init__(self, cfg: Any, provider: Any, kind: str) -> None:
        from podcast_scraper.providers.common.summary_use_case import use_case_prompt_metadata
        from podcast_scraper.server.board_summaries import STAGE

        self.cfg = cfg
        self.provider = provider
        self.ready = False
        self.stage = STAGE
        # "chat": system + user prompts; "document": a seq2seq model's prose input (BART/LED).
        self.kind = kind
        # The model the provider calls, as its own metadata records it — not ``cfg.summary_model``,
        # the local-ML field, which names the served model only on the vLLM profiles.
        self.model = str(getattr(provider, "summary_model", None) or cfg.summary_model)
        # The OpenAI-compatible family keeps its prompts under openai/ whatever the sibling.
        self.namespace = getattr(provider, "_PROMPT_NS", None) or str(cfg.summary_provider)
        # The identity of the templates this provider renders: an edited one re-summarises once.
        meta = (
            use_case_prompt_metadata(
                self.namespace, self.use_case({}), "document" if kind == "document" else "chat"
            )
            or {}
        )
        self.prompt = "+".join(f"{m['name']}@{str(m['sha256'])[:12]}" for m in meta.values() if m)

    def use_case(self, material: dict[str, Any]) -> Any:
        from podcast_scraper.providers.common.summary_use_case import SummaryUseCase

        return SummaryUseCase(
            system_prompt=str(self.cfg.board_summary_system_prompt),
            user_prompt=str(self.cfg.board_summary_user_prompt),
            stage=self.stage,
            max_length=int(self.cfg.board_summary_max_tokens),
            params=material,
            document_prompt=str(self.cfg.board_summary_document_prompt),
        )

    def _initialize(self) -> None:
        """Once, before the first call of the sweep, never before. A provider that cannot get
        ready (the DGX in another GPU mode, a wrong model on the slot) ends the sweep as
        unreachable; the next sweep tries again."""
        if self.ready:
            return
        try:
            self.provider.initialize()
        except Exception as exc:  # noqa: BLE001 — any reason it is not ready ends this sweep
            raise ConnectionError(f"provider not ready: {exc!r}") from exc
        self.ready = True

    def summarize(self, material: dict[str, Any]) -> str:
        self._initialize()
        out = self.provider.summarize(text="", params={"use_case": self.use_case(material)})
        return str(out.get("summary") or "")

    def price(self, material: dict[str, Any]) -> Optional[float]:
        """The call's estimated cost from the pipeline's pricing table, on the prompt as rendered;
        None when unpriced, which the sweep treats as not zero."""
        from podcast_scraper.providers.common.summary_use_case import (
            build_use_case_document,
            build_use_case_prompts,
        )
        from podcast_scraper.workflow.helpers import calculate_provider_cost

        use_case = self.use_case(material)
        if self.kind == "document":
            system, user = "", build_use_case_document(self.namespace, use_case)[0]
        else:
            system, user, _sn, _un = build_use_case_prompts(self.namespace, use_case)
        cost: Optional[float] = calculate_provider_cost(
            cfg=self.cfg,
            provider_type=str(self.cfg.summary_provider),
            capability="summarization",
            model=self.model,
            prompt_tokens=max(1, (len(system) + len(user)) // 3),
            completion_tokens=int(self.cfg.board_summary_max_tokens),
        )
        return cost


def _run_once() -> None:
    from podcast_scraper.providers.common.summary_use_case import use_case_input_kind
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
    if type(provider).__name__ == "FallbackAwareSummarizationProvider":
        _log(f"profile {PROFILE} fails over to another provider; refusing (nothing priced it)")
        _log_json(event="board_summary.tick", stopped="failover_configured", profile=PROFILE, ts=ts)
        return
    kind = use_case_input_kind(provider)
    if kind is None:
        # A summarizer that carries no use case would summarise the empty transcript and ignore
        # the board. Every one the factory builds carries it today; one added without it stops here.
        _log(f"profile {PROFILE} summarises with {cfg.summary_provider}, which runs no use case")
        _log_json(
            event="board_summary.tick", stopped="unsupported_provider", profile=PROFILE, ts=ts
        )
        return
    board = _BoardUseCase(cfg, provider, kind)
    result = board_summaries.sweep(
        CORPUS_ROOT,
        DATA_DIR,
        summarize=board.summarize,
        price=board.price,
        model=board.model,
        prompt=board.prompt,
        limits=board_summaries.Limits.from_env(),
    )
    for user_id, board_id in result.summarised:
        _log_json(event="board_summary.written", user=user_id, board=board_id, ts=ts)
    _log_json(
        event="board_summary.tick",
        profile=PROFILE,
        model=board.model,
        prompt=board.prompt,
        counts=dict(result.counts),
        stopped=result.stopped,
        ts=ts,
    )
    _log(
        f"tick: summarised {len(result.summarised)} "
        f"[{', '.join(f'{k}={v}' for k, v in sorted(result.counts.items()))}]"
        + (f" stopped={result.stopped}" if result.stopped else "")
    )


def _references_once() -> None:
    """The reference pass (#2345): every listener's stored ids re-keyed or flagged."""
    from podcast_scraper.server import reference_integrity

    ts = int(time.time())
    results = reference_integrity.repair_all(DATA_DIR, CORPUS_ROOT)
    rekeyed = sum(len(r.receipts) for r in results)
    flagged = sum(len(r.flagged) for r in results)
    _log_json(event="references.tick", users=len(results), rekeyed=rekeyed, flagged=flagged, ts=ts)
    _log(f"references: {len(results)} users, {rekeyed} re-keyed, {flagged} flagged")


def _sleep_to_next_interval(sleep: Callable[[float], object] = time.sleep) -> None:
    now = time.time()
    sleep(INTERVAL_S - (now % INTERVAL_S) + OFFSET_S)


def _cycle() -> None:
    try:
        _references_once()
    except Exception:  # noqa: BLE001 — the summaries still run
        _LOGGER.exception("reference pass failed")
    try:
        _run_once()
    except Exception:  # noqa: BLE001 — a bad cycle must never kill the loop
        # ERROR through logging, so Sentry's logging integration reports it, as for the pipeline.
        _LOGGER.exception("board summary cycle failed")
    _beat()


def _startup() -> None:
    """What the pipeline CLI does before any work (``cli.main``): Sentry as the pipeline, OTEL,
    and logging to the console at INFO, so the ``llm_cost`` events reach stdout and Alloy."""
    from podcast_scraper.utils.otel_init import init_otel
    from podcast_scraper.utils.sentry_init import init_sentry
    from podcast_scraper.workflow.orchestration import apply_log_level

    init_sentry("pipeline")
    init_otel()
    apply_log_level("INFO")


def main() -> NoReturn:
    _startup()
    _log(
        f"start: corpus={CORPUS_ROOT} data_dir={DATA_DIR} profile={PROFILE} "
        f"interval={INTERVAL_S}s offset={OFFSET_S}s heartbeat={HEARTBEAT}"
    )
    while True:
        _cycle()
        _sleep_to_next_interval()


if __name__ == "__main__":
    main()
