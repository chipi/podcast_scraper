"""HTTP timeout configuration utilities.

This module provides helpers for configuring HTTP client timeouts with separate
connect and read timeouts for better control over network behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from podcast_scraper import config

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore


def get_http_timeout(
    cfg: config.Config,
    connect_timeout: float | None = None,
    read_timeout: float | None = None,
    write_timeout: float | None = None,
    pool_timeout: float | None = None,
) -> httpx.Timeout | float | None:
    """Get HTTP timeout configuration for httpx clients.

    This function creates an httpx.Timeout object with separate timeouts for
    connect, read, write, and pool operations. This provides better control
    than a single timeout value.

    Args:
        cfg: Configuration object
        connect_timeout: Connect timeout in seconds (default: 10.0)
        read_timeout: Read timeout in seconds (default: from cfg.timeout or 60.0)
        write_timeout: Write timeout in seconds (default: 10.0)
        pool_timeout: Pool timeout in seconds (default: 10.0)

    Returns:
        httpx.Timeout object if httpx is available, otherwise float timeout value
        or None if httpx is not available

    Note:
        - Connect timeout should be short (10s) to fail fast on connection issues
        - Read timeout should match operation needs (60s default, longer for transcription)
        - Write timeout should be short (10s) for request sending
        - Pool timeout should be short (10s) for connection pool operations
    """
    if httpx is None:
        # Fallback to simple timeout if httpx not available
        return read_timeout or getattr(cfg, "timeout", 60.0)

    # Default values
    connect = connect_timeout if connect_timeout is not None else 10.0
    read = read_timeout if read_timeout is not None else getattr(cfg, "timeout", 60.0)
    write = write_timeout if write_timeout is not None else 10.0
    pool = pool_timeout if pool_timeout is not None else 10.0

    # Ensure connect timeout is always strictly less than read timeout
    # This prevents connection issues from blocking for too long
    if connect >= read:
        # If read timeout is very small, reduce connect proportionally
        # But ensure it's always strictly less than read
        connect = min(max(0.1, read * 0.5), read - 0.1)  # At least 0.1s, max read-0.1s

    return httpx.Timeout(
        connect=connect,
        read=read,
        write=write,
        pool=pool,
    )


def get_transcription_timeout(cfg: config.Config) -> httpx.Timeout | float | None:
    """Get timeout configuration for transcription operations.

    Transcription operations can be long-running (up to 30 minutes for long episodes),
    so we use a longer read timeout while keeping connect timeout short.

    Args:
        cfg: Configuration object

    Returns:
        httpx.Timeout object with transcription-appropriate timeouts
    """
    transcription_timeout = getattr(cfg, "transcription_timeout", 1800)  # 30 min
    return get_http_timeout(
        cfg,
        connect_timeout=10.0,  # Fast fail on connection issues
        read_timeout=float(transcription_timeout),  # Long for transcription
        write_timeout=10.0,
        pool_timeout=10.0,
    )


def get_summarization_timeout(cfg: config.Config) -> httpx.Timeout | float | None:
    """Get timeout configuration for summarization operations.

    Summarization operations are typically faster than transcription but can still
    take several minutes for long transcripts.

    Args:
        cfg: Configuration object

    Returns:
        httpx.Timeout object with summarization-appropriate timeouts
    """
    summarization_timeout = getattr(cfg, "summarization_timeout", 1200)  # 20 min
    return get_http_timeout(
        cfg,
        connect_timeout=10.0,  # Fast fail on connection issues
        read_timeout=float(summarization_timeout),  # Long for summarization
        write_timeout=10.0,
        pool_timeout=10.0,
    )


#: Fraction of the metadata-generation deadline any SINGLE chat call may consume (#1894).
#:
#: The per-call transport timeout used to equal the deadline itself, which made a hung call
#: indistinguishable from a legitimately long stage: one stuck request could burn the entire
#: budget before anything fired, and the only symptom was "DEADLINE EXCEEDED ... and is STILL
#: RUNNING" — after the fact, with no way to tell which of ~40 calls was at fault.
#:
#: The deadline wraps summary + GI + KG, which is DOZENS of calls. Measured over 82 episodes of
#: the 2026-08-31 batch, that stage totals p50=1130s, p90=2006s, max=3776s — so no single call
#: should ever need anything close to the whole budget. A third leaves generous headroom for the
#: slowest legitimate call while catching a genuine hang in minutes rather than at the deadline.
#:
#: VALIDATED 2026-09-02 against 40 episodes of the Batch A ingestion (prod_dgx_full):
#:   - 1/40 episodes would have fired the old flat 1200s deadline; **0/40 fire the scaled one**
#:   - headroom (deadline / actual): min 2.01x, median 2.70x — never tight, never absurd
#:   - Pearson(words, metadata_sec) = 0.645 over all 40, 0.676 above the crossover. The n=15
#:     figure quoted when this shipped was 0.868; the relationship is MODERATE, not strong, and
#:     the original number was optimistic. The fix does not depend on strong linearity — it needs
#:     the budget to clear healthy work and still bound a hang, and both hold.
#:
#: Deliberately a FRACTION, not a new absolute: the deadline is already profile-configurable,
#: and a second independent number would drift out of step with it — which is how the timeout
#: came to equal the deadline in the first place.
SINGLE_CALL_TIMEOUT_FRACTION = 1.0 / 3.0

#: Never go below this regardless of the fraction: a very short configured deadline must not
#: produce a per-call timeout that kills healthy calls on a slow model.
MIN_SINGLE_CALL_TIMEOUT_SEC = 120.0


#: Seconds of metadata generation (summary+GI+KG) to budget per 1000 transcript words (#1920).
#:
#: The deadline at ``workflow/stages/processing.py`` was a flat ``summarization_timeout`` (1200s)
#: wrapping work that scales with transcript length. Measured over the 15 episodes of the
#: 2026-09-01 Batch A pass on ``prod_dgx_full``:
#:
#:     Pearson(word_count, metadata_sec) = 0.868      observed max = 74.5 s per 1k words
#:
#: The single overrun in that pass was the single longest episode (16,345 words, 1218s against
#: the 1200s flat budget) — it completed fine and still raised an ERROR-level DEADLINE EXCEEDED.
#:
#: This matters by policy, not by luck: §5h of the onboarding plan sets the episode ceiling at
#: TWO HOURS (~20k words), which at the observed rate needs ~1500s. A flat 1200s budget is
#: guaranteed to fire on the longest episodes the corpus explicitly permits.
#:
#: 150 is ~2x the observed worst case — enough headroom that a contended GPU does not trip it,
#: while still bounded. Transcript words, not audio minutes: it predicts better, it is the direct
#: driver of token count, and unlike the audio file it is guaranteed present at the call site.
#:
#: VALIDATED 2026-09-02 against 40 episodes of the Batch A ingestion (prod_dgx_full):
#:
#:   - 1/40 episodes would have fired the old flat 1200s deadline; **0/40 fire the scaled one**.
#:   - headroom (deadline / actual): min 2.01x, median 2.70x — never tight, never absurd.
#:   - Pearson(words, metadata_sec) = 0.645 over all 40 (0.676 above the crossover). The n=15
#:     figure quoted when this shipped was 0.868 — the relationship is MODERATE, not strong, and
#:     that number was optimistic. The fix does not depend on strong linearity: it needs the
#:     budget to clear healthy work and still bound a hang, and both hold.
#:
#: Read the REGIME before comparing rates to this constant. The flat floor governs below ~8000
#: words (1200/150), so 150 only applies above that. In the regime it governs, the worst observed
#: rate is 74.5 s/1k words and 150 is exactly 2.0x it. BELOW the crossover the apparent rate
#: reaches 251 s/1k (a 1230-word episode costing 309s) because fixed per-episode overhead
#: dominates a tiny transcript — alarming against this constant, and irrelevant to it, because
#: the flat floor covers those episodes entirely.
#:
#: Cost of the headroom, stated plainly: a genuine wedge on a 20k-word episode is now detected at
#: ~3000s instead of 1200s — 2.5x slower on exactly the episodes most likely to wedge. Accepted
#: because the alternative (a flat budget that fires on healthy work) trains readers to ignore
#: the line entirely.
#:
#: **This value is the DGX rate, and it is a DEFAULT, not a law.** It was measured on n=15
#: episodes of ONE profile (``prod_dgx_full`` / vLLM Qwen3-30B on the local GPU). A cloud
#: provider's seconds-per-word is a different number entirely — different hardware, network
#: latency, rate limits, retry behaviour — and nobody has measured it. Applying a locally-derived
#: constant to a cloud profile is mixing environments that share nothing but the code path, so
#: profiles override it via ``metadata_sec_per_1k_words`` rather than inheriting a number that
#: describes someone else's hardware.
METADATA_SEC_PER_1K_TRANSCRIPT_WORDS = 150.0


def get_metadata_generation_timeout(
    cfg: config.Config, transcript_word_count: int
) -> Optional[float]:
    """Deadline for metadata generation (summary+GI+KG), scaled by transcript length (#1920).

    Never returns less than the configured ``summarization_timeout``, so short episodes keep
    exactly today's budget and nothing regresses. A missing or nonsensical word count also
    falls back to the flat value rather than producing a tiny deadline.

    **Returns None when the deadline is disabled.** ``summarization_timeout`` is
    ``Optional[int]`` and its config docstring says "Set to None to disable timeout";
    ``timeout_context`` documents "None or <= 0 disables observation entirely". A first cut of
    this function did ``float(getattr(cfg, "summarization_timeout", 1200))``, which raises
    ``TypeError: float() argument must be ... not 'NoneType'`` on that documented setting — and
    the caller's broad ``except Exception`` would have turned it into *every episode failed*.
    Scaling a disabled deadline back into an enabled one is equally wrong, so the disable
    semantics pass straight through.

    Note the per-call transport timeout (#1894) does NOT scale with this value: it is a fraction
    of the FLAT ``cfg.summarization_timeout``, read once at provider init
    (``get_single_chat_call_timeout`` / ``openai_provider``). So hang detection is unchanged by
    this scaling, and for a very long episode that flat per-call bound is the tighter of the two.
    """
    per_1k = float(
        getattr(cfg, "metadata_sec_per_1k_words", None) or METADATA_SEC_PER_1K_TRANSCRIPT_WORDS
    )
    raw = getattr(cfg, "summarization_timeout", 1200)
    if raw is None:
        return None
    flat = float(raw)
    if flat <= 0:
        return flat  # 0 / negative = disabled; preserve, do not scale into an enabled deadline
    if not transcript_word_count or transcript_word_count <= 0:
        return flat
    scaled = (float(transcript_word_count) / 1000.0) * per_1k
    return max(flat, scaled)


def get_single_chat_call_timeout(cfg: config.Config) -> float:
    """Transport timeout for ONE chat request, strictly below the stage deadline (#1894).

    The acceptance criterion on that issue is exactly this: "the underlying call has a
    configurable timeout set to a value LESS than the deadline". Equal is not less — with the
    two identical, the deadline can only ever report a hang that has already cost the full
    budget, and cannot distinguish it from the aggregate simply being expensive (41% of healthy
    episodes exceeded the 1200s deadline in production).

    Making the single call fail EARLY is what lets the RFC-106 fallback ladder do its job: a
    timeout that fires at a third of the budget leaves room to retry or fail over and still
    finish the episode, where one that fires at the deadline leaves none.
    """
    deadline = float(getattr(cfg, "summarization_timeout", 1200) or 1200)
    return max(MIN_SINGLE_CALL_TIMEOUT_SEC, deadline * SINGLE_CALL_TIMEOUT_FRACTION)


def get_openai_client_timeout(cfg: config.Config) -> httpx.Timeout | float | None:
    """HTTP timeout for the unified OpenAI SDK client (Whisper + chat completions).

    One ``OpenAI`` client handles audio transcription and chat (summarization,
    hybrid transcript cleaning, speaker detection). ``cfg.timeout`` is often set
    low for RSS/media downloads; using it alone as the read timeout causes
    spurious timeouts on long chat requests (e.g. cleaning a full episode).

    Read timeout is the maximum of ``timeout``, ``summarization_timeout``, and
    ``transcription_timeout`` (using config defaults when a value is unset).

    Args:
        cfg: Configuration object

    Returns:
        Same shape as :func:`get_http_timeout` (``httpx.Timeout`` when httpx is available).
    """
    from .. import config_constants

    summ = getattr(cfg, "summarization_timeout", None)
    if summ is None:
        summ = config_constants.DEFAULT_SUMMARIZATION_TIMEOUT_SECONDS
    trans = getattr(cfg, "transcription_timeout", None)
    if trans is None:
        trans = config_constants.DEFAULT_TRANSCRIPTION_TIMEOUT_SECONDS

    base = get_http_timeout(cfg)
    if httpx is None:
        # ``get_http_timeout`` returns ``float`` here; annotation is wider for the httpx path.
        if isinstance(base, (int, float)):
            base_read = float(base)
        elif base is None:
            base_read = float(getattr(cfg, "timeout", 60.0))
        else:
            raw_read = base.read
            base_read = float(raw_read if raw_read is not None else getattr(cfg, "timeout", 60.0))
        return max(base_read, float(summ), float(trans))

    if not isinstance(base, httpx.Timeout):
        base_read = float(base) if base is not None else float(getattr(cfg, "timeout", 60.0))
    else:
        raw_read = base.read
        base_read = float(raw_read if raw_read is not None else getattr(cfg, "timeout", 60.0))
    read_timeout = max(base_read, float(summ), float(trans))
    return get_http_timeout(cfg, read_timeout=read_timeout)
