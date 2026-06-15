"""Shared resilience primitives for the DGX provider calls (#946 / #954).

The DGX Whisper (``:8002``) and pyannote diarize (``:8001``) clients both upload
an audio file to a single-GPU service on the shared GB10 box and must degrade
gracefully when that GPU is contended or wedged. They share the same defences,
collected here so the two providers stay in lockstep:

- :func:`probe_audio_duration_sec` / :func:`effective_timeout_sec` — size a
  request timeout from the audio length so long episodes don't false-fail under
  brief contention.
- :data:`TimeoutLike` — the exception classes that mean "the server is slow /
  still working" (fall over, don't re-queue a duplicate) vs a connection blip.
- :func:`run_with_watchdog` — a hard process-side wall-clock deadline. httpx's
  own read/write timeout has been observed to never fire when a co-tenant
  workload (e.g. a vLLM crash-loop, #954) intermittently stalls the GPU: the
  multipart upload trickles, each accepted chunk resets httpx's per-write
  timeout, and the request hangs indefinitely. The watchdog guarantees the
  caller regains control and fails over.
- :class:`CircuitBreaker` — a trimmed-down take on
  ``rss/http_policy.CircuitBreaker``: once DGX has failed, trip a cooldown during
  which callers skip DGX entirely (no per-request timeout tax on a wedged batch);
  a half-open probe after the cooldown re-tests so it self-heals on recovery.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Callable, cast, Deque, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default wall-clock slack added on top of a request's duration-scaled timeout
# before the watchdog gives up (covers upload + response serialisation).
WATCHDOG_GRACE_SEC = 30.0

# Timeout-class errors: the server is slow/contended and (likely) still working
# the request, so retrying would double the load. Resolved lazily so a missing
# httpx doesn't break import.
try:  # pragma: no cover - import guard
    import httpx as _httpx

    TimeoutLike: tuple[type[Exception], ...] = (_httpx.TimeoutException, TimeoutError)
except ImportError:  # pragma: no cover
    TimeoutLike = (TimeoutError,)


def probe_audio_duration_sec(audio_path: str) -> Optional[float]:
    """Best-effort audio duration (seconds) for timeout scaling; None on failure.

    Used only to size the request timeout, so a miss is harmless — it just falls
    back to the flat base budget. Dependency-light (soundfile is already a
    transitive dep) and never raises.
    """
    try:
        import soundfile as sf

        info = sf.info(audio_path)
        if info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:  # noqa: BLE001 - duration is advisory only
        return None
    return None


def effective_timeout_sec(
    base_sec: float, per_audio_min_sec: float, duration_sec: Optional[float]
) -> float:
    """Duration-scaled request budget: ``base + (audio_minutes * per_audio_min)``.

    A flat timeout false-fails long episodes whenever the shared GPU is briefly
    contended. Scaling by audio length lets a call wait the contention out instead
    of bailing prematurely.
    """
    base = float(base_sec)
    if duration_sec and duration_sec > 0 and per_audio_min_sec:
        base += (float(duration_sec) / 60.0) * float(per_audio_min_sec)
    return base


def run_with_watchdog(fn: Callable[[], T], deadline_sec: float, *, label: str) -> T:
    """Run ``fn()`` under a hard wall-clock deadline; raise ``TimeoutError`` if it
    overruns.

    Runs ``fn`` in a daemon worker thread and waits at most ``deadline_sec``. If the
    call hasn't returned by then we stop waiting and raise — guaranteeing the caller
    regains control even when ``fn`` (e.g. a trickling httpx upload) ignores its own
    timeout. The orphaned worker is a daemon: it holds at most one connection, never
    blocks process exit, and unwinds whenever the underlying call finally errors.
    Exceptions raised inside ``fn`` propagate to the caller unchanged.
    """
    box: dict[str, Any] = {}

    def _run() -> None:
        try:
            box["res"] = fn()
        except BaseException as exc:  # noqa: BLE001 - propagate to caller thread
            box["err"] = exc

    worker = threading.Thread(target=_run, name=label, daemon=True)
    worker.start()
    worker.join(deadline_sec)
    if worker.is_alive():
        raise TimeoutError(
            f"{label} exceeded hard wall-clock deadline {deadline_sec:.0f}s; "
            "abandoning request and failing over"
        )
    if "err" in box:
        raise box["err"]
    return cast(T, box["res"])


# TCP keepalive schedule: probe an idle socket after ~30s, then every 15s up to 4
# times, so a connection whose underlying path died mid-request (e.g. a Tailscale
# path switch) is declared dead in ~90s instead of the OS default (2h on macOS).
_KEEPALIVE_IDLE_SEC = 30
_KEEPALIVE_INTERVAL_SEC = 15
_KEEPALIVE_PROBES = 4


def keepalive_socket_options(
    *,
    idle_sec: int = _KEEPALIVE_IDLE_SEC,
    interval_sec: int = _KEEPALIVE_INTERVAL_SEC,
    probes: int = _KEEPALIVE_PROBES,
) -> "list[tuple[int, int, int]]":
    """TCP keepalive ``setsockopt`` tuples, built defensively for the host platform.

    A long-blocking DGX POST holds an idle socket open for the whole multi-minute GPU
    run; if the path dies underneath it the socket stays ESTABLISHED for the OS default
    and the request hangs. Enabling ``SO_KEEPALIVE`` with a short probe schedule lets the
    kernel detect the dead path in ~90s and surface a connection error to fail over on.
    Each option is guarded with ``hasattr`` so a platform-absent constant is skipped, not
    raised (Linux exposes ``TCP_KEEPIDLE``; macOS exposes ``TCP_KEEPALIVE``).
    """
    import socket

    opts: "list[tuple[int, int, int]]" = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    for name in ("TCP_KEEPIDLE", "TCP_KEEPALIVE"):  # Linux, then macOS
        if hasattr(socket, name):
            opts.append((socket.IPPROTO_TCP, getattr(socket, name), idle_sec))
            break
    if hasattr(socket, "TCP_KEEPINTVL"):
        opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, interval_sec))
    if hasattr(socket, "TCP_KEEPCNT"):
        opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPCNT, probes))
    return opts


def dgx_http_client(timeout: Any, *, headers: "Optional[dict[str, str]]" = None) -> Any:
    """An ``httpx.Client`` hardened for long-blocking calls to a contended DGX service.

    On top of the per-request fresh client + :func:`run_with_watchdog` deadline, this adds
    the two transport-level defences that actually fit a long-blocking upload (#956):

    - **TCP keepalive** (:func:`keepalive_socket_options`) so a socket whose path died
      mid-request is reaped in ~90s instead of hanging on the OS default.
    - **``Connection: close``** so the server tears the socket down after the response and
      no half-dead connection lingers.

    A per-read timeout is deliberately NOT set: these POSTs stream zero bytes during the
    multi-minute GPU run, so any read deadline shorter than processing would false-abort a
    healthy call — the duration-scaled watchdog is the correct backstop.
    """
    try:
        import httpx
    except ImportError as exc:  # friendly, actionable error for the DGX providers
        raise RuntimeError("httpx required for tailnet_dgx provider HTTP calls") from exc

    merged: dict[str, str] = {"Connection": "close"}
    if headers:
        merged.update(headers)
    transport = httpx.HTTPTransport(socket_options=keepalive_socket_options())
    return httpx.Client(timeout=timeout, transport=transport, headers=merged)


class CircuitBreaker:
    """closed → rolling-window failures → open(cooldown) → half-open probe → closed.

    A trimmed-down take on ``rss/http_policy.CircuitBreaker`` for a single DGX
    endpoint. While open, :meth:`allow` returns False so callers skip DGX and go
    straight to their fallback — a wedged batch isn't paced by per-request timeouts.
    After the cooldown one half-open probe is allowed; its outcome closes or
    re-opens the breaker. A ``hard`` failure (a definitive timeout — strong evidence
    the endpoint is unusable right now) trips immediately so the very first wedge
    spares the rest of the batch. Thread-safe and process-wide.
    """

    def __init__(
        self,
        failure_threshold: int,
        window_sec: float,
        cooldown_sec: float,
        *,
        name: str = "dgx",
    ) -> None:
        self._name = name
        self._threshold = max(1, failure_threshold)
        self._window = max(1.0, window_sec)
        self._cooldown = max(1.0, cooldown_sec)
        self._failures: Deque[float] = deque(maxlen=64)
        self._state = "closed"
        self._open_until = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        return self._state

    def allow(self) -> bool:
        """False while the breaker is open (cooldown not elapsed); else True."""
        with self._lock:
            if self._state == "open":
                if time.monotonic() < self._open_until:
                    return False
                self._state = "half_open"  # allow a single probe
                logger.info("%s circuit breaker half-open: probing DGX once", self._name)
            return True

    def record_success(self) -> None:
        """Close the breaker and clear the failure window — DGX answered cleanly."""
        with self._lock:
            if self._state != "closed":
                logger.info("%s circuit breaker closed: DGX recovered", self._name)
            self._state = "closed"
            self._failures.clear()

    def record_failure(self, *, hard: bool = False) -> None:
        """Record a DGX failure; open the breaker on a ``hard`` failure (a definitive
        timeout) or a failed half-open probe immediately, else once the rolling window
        reaches the failure threshold."""
        now = time.monotonic()
        with self._lock:
            if hard or self._state == "half_open":
                if self._state != "open":
                    logger.warning(
                        "%s circuit breaker OPEN for %.0fs: %s",
                        self._name,
                        self._cooldown,
                        "hard timeout" if hard else "half-open probe failed",
                    )
                self._state = "open"
                self._open_until = now + self._cooldown
                self._failures.clear()
                return
            cutoff = now - self._window
            while self._failures and self._failures[0] < cutoff:
                self._failures.popleft()
            self._failures.append(now)
            if len(self._failures) >= self._threshold:
                logger.warning(
                    "%s circuit breaker OPEN for %.0fs: %d failures within %.0fs",
                    self._name,
                    self._cooldown,
                    len(self._failures),
                    self._window,
                )
                self._state = "open"
                self._open_until = now + self._cooldown

    def reset(self) -> None:
        """Force back to closed (test/ops hook)."""
        with self._lock:
            self._state = "closed"
            self._failures.clear()
            self._open_until = 0.0


# ---------------------------------------------------------------------------
# Response-shape guardrails (#999 / ADR-099)
# ---------------------------------------------------------------------------
#
# The connection-level primitives above (timeouts, watchdog, breaker) catch
# transport failures from the self-deployed DGX services. They cannot catch the
# "200 OK + semantically corrupted content" failure mode surfaced by the #996
# catastrophic-tail sweep. ``GuardrailViolation`` is the failure-class added to
# close that gap; the existing consumer fallback paths (which already catch
# ``TimeoutLike``) treat it as a sibling exception — DGX fails, breaker counts,
# cloud fallback fires. See ``docs/adr/ADR-099-response-shape-guardrails-*``.
#
# Initial thresholds are first-principles defaults locked at ship time;
# fine-tuning against firing-rate data is tracked in #1002.


# Service identifiers — also used as the Prometheus label and the dispatch key.
SERVICE_WHISPER = "whisper"
SERVICE_OLLAMA = "ollama"
SERVICE_VLLM = "vllm"
SERVICE_PYANNOTE = "pyannote"

# Reason identifiers — per-service enum. Keep this list small; Prometheus
# cardinality budget assumes ~10 distinct (service, reason) combos total.
REASON_WHISPER_LENGTH_FLOOR = "length_floor_violated"
REASON_WHISPER_EMPTY = "empty_response"
REASON_OLLAMA_EMPTY = "empty_content"
REASON_OLLAMA_THINKING_PROSE = "thinking_prose_detected"
REASON_VLLM_BAD_JSON = "json_parse_failed"
REASON_VLLM_FINISH_LENGTH = "finish_reason_length"
REASON_PYANNOTE_EMPTY_SEGMENTS = "empty_segments"

# Whisper length floor: 50% of expected word count, where the speaking rate
# estimate is 2.5 words/sec (median podcast pace). Catches the WER=1.0 mode
# without false-firing on natural silence stretches.
_WHISPER_WORDS_PER_SEC_ESTIMATE = 2.5
_WHISPER_LENGTH_FLOOR_FRACTION = 0.5

# Pyannote: only enforce the empty-segments check for non-trivially-short audio
# — < 5s clips legitimately diarize as empty in some configs.
_PYANNOTE_MIN_DURATION_FOR_CHECK_SEC = 5.0

# Ollama: reasoning-prose markers seen in qwen3.5 thinking-budget failures.
_OLLAMA_THINKING_MARKERS: tuple[str, ...] = (
    "<think>",
    "<think ",
    "Okay, so I need to",
    "Let me think",
)


class GuardrailViolation(Exception):
    """A self-deployed inference service returned a successful HTTP response
    whose content fails a structural sanity check (ADR-099).

    Behaviour: callers that already handle ``TimeoutLike`` should add a
    sibling ``except GuardrailViolation`` block. The fallback path is
    identical — DGX call counted as a failure, breaker records it, cloud
    fallback fires. No retry against DGX: a guardrail-violating response
    under contention is unlikely to repair on immediate retry, and the
    fallback path is already paid for.

    Attributes carry enough context for log + telemetry without needing
    high-cardinality label values on Prometheus side (those live in the
    log body and Sentry event scope).
    """

    def __init__(self, service: str, reason: str, response_summary: str = "") -> None:
        self.service = service
        self.reason = reason
        # Truncated to ~200 chars to keep log lines bounded; full response
        # body goes to the consumer's debug log if it needs investigation.
        self.response_summary = (response_summary or "")[:200]
        super().__init__(
            f"guardrail violation: service={service} reason={reason} "
            f"summary={self.response_summary!r}"
        )


# Prometheus counter — lazy-imported on first use, cached at module scope so the
# Counter is registered exactly once even when this module is reloaded under
# pytest. Pattern matches ``server/pipeline_run_prometheus.py:_PROM_STATE``.
_GUARDRAIL_COUNTER: Any = None


def _record_guardrail_violation(service: str, reason: str) -> None:
    """Increment the Prometheus counter for a guardrail violation.

    Strict label discipline: only `service` + `reason` are labels. Both are
    drawn from the fixed enum constants above. **Never** add high-cardinality
    fields (audio filename, request ID, response content) as labels — those
    belong in the structured log body / Sentry context. If the active series
    count for this counter climbs above ~50, that is a bug.
    """
    global _GUARDRAIL_COUNTER
    if _GUARDRAIL_COUNTER is None:
        try:
            from prometheus_client import Counter

            _GUARDRAIL_COUNTER = Counter(
                "dgx_guardrail_violations_total",
                "Count of response-shape guardrail violations from self-deployed "
                "inference services (ADR-099). Labels: service, reason.",
                labelnames=("service", "reason"),
            )
        except Exception:  # pragma: no cover - prometheus_client optional
            _GUARDRAIL_COUNTER = False  # sentinel — don't retry the import
            return
    if _GUARDRAIL_COUNTER:
        try:
            _GUARDRAIL_COUNTER.labels(service=service, reason=reason).inc()
        except Exception:  # pragma: no cover - telemetry never breaks the caller
            pass


def _raise_violation(service: str, reason: str, summary: str = "") -> None:
    """Emit telemetry then raise ``GuardrailViolation``. Centralised so the
    log + counter pair stays in lockstep across every per-service callback.
    """
    logger.warning(
        "DGX guardrail violation: service=%s reason=%s summary=%s",
        service,
        reason,
        summary[:200] if summary else "",
    )
    _record_guardrail_violation(service, reason)
    raise GuardrailViolation(service, reason, summary)


def check_whisper_response(text: str, *, audio_duration_sec: Optional[float]) -> None:
    """Whisper length-floor guardrail.

    Raises if the returned transcript has < 50% of the word count expected
    from audio duration. Catches the WER=1.0 mode observed in #996. When
    ``audio_duration_sec`` is None (probe failed), the check is skipped — a
    miss on length scaling is harmless; the check is advisory in that case.
    """
    if text is None or text == "":
        _raise_violation(SERVICE_WHISPER, REASON_WHISPER_EMPTY, "")
    if audio_duration_sec is None or audio_duration_sec <= 0:
        return
    word_count = len(text.split())
    floor = int(
        audio_duration_sec * _WHISPER_WORDS_PER_SEC_ESTIMATE * _WHISPER_LENGTH_FLOOR_FRACTION
    )
    if word_count < floor:
        summary = f"word_count={word_count} floor={floor} " f"duration_sec={audio_duration_sec:.1f}"
        _raise_violation(SERVICE_WHISPER, REASON_WHISPER_LENGTH_FLOOR, summary)


def check_ollama_response(content: Optional[str]) -> None:
    """Ollama empty / thinking-prose guardrail.

    Catches the qwen3.5 thinking-budget trap (#959/#985 evidence). Empty
    content is a hard fail; reasoning-prose prefix markers (``<think>``,
    "Okay, so I need to", "Let me think") are a hard fail because the
    response indicates the model burned its budget on hidden reasoning
    before producing user-facing text.
    """
    if content is None or content == "":
        _raise_violation(SERVICE_OLLAMA, REASON_OLLAMA_EMPTY, "")
    head = (content or "")[:200]
    for marker in _OLLAMA_THINKING_MARKERS:
        if marker in head:
            _raise_violation(SERVICE_OLLAMA, REASON_OLLAMA_THINKING_PROSE, head)


def check_vllm_response(
    content: Optional[str],
    *,
    finish_reason: Optional[str] = None,
    expect_json: bool = False,
) -> None:
    """vLLM JSON / finish-reason guardrail.

    Preventive for the autoresearch path — failure modes observed in #985
    cross-provider sweeps. ``finish_reason='length'`` means the model
    truncated mid-output and the response is structurally incomplete;
    ``expect_json=True`` enables a JSON-parse check on the content.
    """
    if finish_reason == "length":
        summary = f"finish_reason=length content_head={(content or '')[:80]!r}"
        _raise_violation(SERVICE_VLLM, REASON_VLLM_FINISH_LENGTH, summary)
    if expect_json:
        if content is None or content == "":
            _raise_violation(SERVICE_VLLM, REASON_VLLM_BAD_JSON, "empty content")
        try:
            import json as _json

            # ``content`` is non-empty by the early-return above; type-narrow
            # for mypy (Optional[str] -> str).
            assert content is not None
            _json.loads(content)
        except (ValueError, TypeError) as exc:
            summary = f"parse_error={exc!s} head={(content or '')[:80]!r}"
            _raise_violation(SERVICE_VLLM, REASON_VLLM_BAD_JSON, summary)


def check_pyannote_response(segments: list[Any], *, audio_duration_sec: Optional[float]) -> None:
    """Pyannote empty-segments guardrail.

    Preventive — no observed cases yet. Empty diarization output for audio
    longer than 5 s is structurally invalid (every non-trivial audio file
    has at least one speech segment). Skipped for shorter clips because
    pyannote legitimately can return ``[]`` on near-empty audio.
    """
    if (
        not segments
        and audio_duration_sec
        and audio_duration_sec > _PYANNOTE_MIN_DURATION_FOR_CHECK_SEC
    ):
        summary = f"duration_sec={audio_duration_sec:.1f} segments=[]"
        _raise_violation(SERVICE_PYANNOTE, REASON_PYANNOTE_EMPTY_SEGMENTS, summary)
