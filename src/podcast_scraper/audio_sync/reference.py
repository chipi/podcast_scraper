"""An episode's reference for listener-side sync: fingerprints of the original at checkpoints
(RFC-129).

The original is the audio we transcribed, from the archive. A listener's app compares short byte
ranges of its own stream against these windows, so only the places where the offset can change are
kept, not the whole episode:

- ``talk_start``: where talk begins, after any opening ads;
- ``after_ad``: where talk resumes after each ad region our transcript marks. Comparing versions
  cannot find these all: a slot whose ad happens to be the same in both looks like talk (measured
  2026-10-16), so the original's own ad regions are the source;
- ``interval``: every ``INTERVAL_S`` inside long stretches of talk, to catch a change we did not
  know about.

Each window is ``WINDOW_S`` of 16-band spectra (``fingerprint.spectra``), quantized to int8 and
base64-encoded: 8 KB per 10 s window.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Sequence, Tuple, TYPE_CHECKING

from podcast_scraper.audio_sync import fingerprint as fp

if TYPE_CHECKING:
    import numpy as np

VERSION = 2
WINDOW_S = 10.0
# After a break whose ads changed, the app does not know where the talk resumes. Version 1 made it
# download a wide range of its stream (±60 s, ~3 MB at 192 kbit/s) and search it: 28 MB for an
# hour with three changed breaks, measured 2026-10-16. Version 2 keeps a LONG window at every
# checkpoint here instead: the app downloads PROBE_S of its stream (~0.3 MB) and finds it inside
# the window, which reaches REACH_S either side of its guess. Interval checkpoints need one too: a
# host can insert a break where the original had none (live, 2026-10-16: +130 s found at an
# interval, 6.6 MB of wide search).
REACH_S = 90.0
PROBE_S = 12.0
LONG_WINDOW_S = 2 * REACH_S + PROBE_S
INTERVAL_S = 300.0
MIN_GAP_S = 30.0
AD_JOIN_S = 3.0
# Where our transcript says an ad ends and where talk resumes can be seconds apart (49.4 s against
# 53.0 s on a real episode, 2026-10-16). A window straddling that edge never matches, and each miss
# costs the app a full search, so the window starts this far in.
AFTER_AD_LEAD_S = 5.0
SCALE = 20.0  # int8 steps per log10 unit: spectra span about ±6


def ad_regions(
    spans_ms: Sequence[Tuple[int, int]], join_s: float = AD_JOIN_S
) -> List[Tuple[float, float]]:
    """Per-segment ad spans (milliseconds) joined into regions (seconds): one read comes back as
    many pieces, one per transcript segment."""
    out: List[List[float]] = []
    for a, b in sorted(spans_ms):
        s, e = a / 1000.0, b / 1000.0
        if out and s - out[-1][1] <= join_s:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def checkpoint_times(
    duration_s: float, regions: Sequence[Tuple[float, float]]
) -> List[Tuple[float, str]]:
    """(second, kind) for each checkpoint, in order."""
    last = duration_s - WINDOW_S
    if last <= 0:
        return []
    start = 0.0
    for a, b in regions:
        if a <= start + 1.0:
            start = max(start, b)
    lead = AFTER_AD_LEAD_S if start > 0 else 0.0
    marks: List[Tuple[float, str]] = [(start + lead, "talk_start")]
    marks += [(b + AFTER_AD_LEAD_S, "after_ad") for a, b in regions if b > start + 1.0]
    talk_ranges = []
    edge = start
    for a, b in regions:
        if b <= start:
            continue
        if a > edge:
            talk_ranges.append((edge, a))
        edge = max(edge, b)
    talk_ranges.append((edge, duration_s))
    for a, b in talk_ranges:
        t = a + INTERVAL_S
        while t < b - WINDOW_S:
            marks.append((t, "interval"))
            t += INTERVAL_S
    out: List[Tuple[float, str]] = []
    for t, kind in sorted(marks):
        if t > last:
            continue
        # A window that runs into another ad never matches (back-to-back ads, measured 2026-10-16),
        # and each miss costs the app a full search.
        if any(a < t + WINDOW_S and t < b for a, b in regions):
            continue
        if out and t - out[-1][0] < MIN_GAP_S and kind == "interval":
            continue
        out.append((round(t, 2), kind))
    return out


def _encode(frames: "np.ndarray") -> str:
    import numpy as np

    q = np.clip(np.round(frames * SCALE), -127, 127).astype(np.int8)
    return base64.b64encode(q.tobytes()).decode("ascii")


def decode_window(b64: str) -> "np.ndarray":
    """A checkpoint's frames back as float spectra, shape (n, BANDS)."""
    import numpy as np

    q = np.frombuffer(base64.b64decode(b64), dtype=np.int8)
    return q.reshape(-1, fp.BANDS).astype(np.float64) / SCALE


def build(samples: "np.ndarray", spans_ms: Sequence[Tuple[int, int]]) -> Dict[str, Any]:
    """The reference document for one episode's original audio."""
    duration = len(samples) / fp.SR
    spec = fp.spectra(samples)
    n = int(WINDOW_S * fp.FPS)
    n_long = int(LONG_WINDOW_S * fp.FPS)
    checkpoints: List[Dict[str, Any]] = []
    regions = ad_regions(spans_ms)
    for t, kind in checkpoint_times(duration, regions):
        i = int(round(t * fp.FPS))
        window = spec[i : i + n]
        if len(window) < n:
            continue
        cp: Dict[str, Any] = {"t": t, "kind": kind, "frames": _encode(window)}
        # The long window stops where the original's next ad starts: everything inside it then has
        # this checkpoint's offset, so a probe that matches past the next break cannot be read as
        # this one (found 2026-10-16 in a test, a probe 90 s on landing after the next break). A
        # late or short stretch gets a shorter window, never none.
        nxt = min((a for a, _ in regions if a > t + 1.0), default=duration)
        end = min(i + n_long, int(nxt * fp.FPS))
        if end - i > n:
            cp["long"] = _encode(spec[i:end])
        checkpoints.append(cp)
    return {
        "version": VERSION,
        "sample_rate": fp.SR,
        "hop": fp.HOP,
        "nfft": fp.NFFT,
        "bands": fp.BANDS,
        "scale": SCALE,
        "window_s": WINDOW_S,
        "reach_s": REACH_S,
        "probe_s": PROBE_S,
        "duration_s": round(duration, 2),
        "checkpoints": checkpoints,
    }


def locate(
    reference_frames: "np.ndarray", chunk: "np.ndarray", chunk_start_s: float
) -> Tuple[float, float]:
    """Where a checkpoint window sits in a decoded chunk of the listener's stream: (stream second,
    match). ``match`` is the mean per-frame correlation of the spectra, 1.0 for the same audio.
    This is what the app does on the device; kept here to test the reference against real streams.
    """
    import numpy as np

    spec = fp.spectra(chunk)
    n = len(reference_frames)
    if len(spec) < n:
        return chunk_start_s, -1.0
    a = reference_frames - reference_frames.mean(1, keepdims=True)
    an = np.sqrt((a * a).sum(1)) + 1e-9
    best, at = -2.0, 0
    for j in range(len(spec) - n + 1):
        b = spec[j : j + n] - spec[j : j + n].mean(1, keepdims=True)
        c = float(np.mean((a * b).sum(1) / (an * (np.sqrt((b * b).sum(1)) + 1e-9))))
        if c > best:
            best, at = c, j
    return chunk_start_s + at / fp.FPS, best


def prepared(spec: "np.ndarray") -> "np.ndarray":
    """Spectra with each frame centred and scaled to unit length, for ``locate_everywhere``."""
    import numpy as np

    c = spec - spec.mean(1, keepdims=True)
    return c / (np.sqrt((c * c).sum(1, keepdims=True)) + 1e-9)


def locate_everywhere(reference_frames: "np.ndarray", stream: "np.ndarray") -> Tuple[float, float]:
    """Where a checkpoint window sits anywhere in a whole stream (``prepared`` spectra): (stream
    second, match), the same mean per-frame correlation as ``locate``, for every position at once
    (one FFT per band)."""
    import numpy as np

    a = prepared(reference_frames)
    n, m = len(a), len(stream)
    if m < n or n == 0:
        return 0.0, -1.0
    size = 1 << int(np.ceil(np.log2(m + n)))
    total = np.zeros(size)
    for b in range(stream.shape[1]):
        total += np.fft.irfft(
            np.fft.rfft(stream[:, b], size) * np.conj(np.fft.rfft(a[:, b], size)), size
        )
    scores = total[: m - n + 1] / n
    j = int(np.argmax(scores))
    return j / fp.FPS, float(scores[j])
