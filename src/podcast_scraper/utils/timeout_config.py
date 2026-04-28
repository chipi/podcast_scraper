"""HTTP timeout configuration utilities.

This module provides helpers for configuring HTTP client timeouts with separate
connect and read timeouts for better control over network behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
