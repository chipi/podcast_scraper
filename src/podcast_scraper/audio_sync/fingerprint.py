"""Audio fingerprints and alignment of a served stream against the original (RFC-129).

Hosts that insert ads per request move the talk around. This finds, for a stream as served, the
stretches of the original it carries and how far each one moved:

1. **Coarse:** every 5 s of the original, a 10 s window of loudness (log energy per 20 ms frame) is
   found in the stream by normalized cross-correlation, then pinned to one frame by the smallest
   loudness difference within ±0.5 s.
2. **Stretches:** consecutive windows with the same offset (within 0.04 s) form a stretch.
3. **Edges:** each stretch's edges are walked out frame by frame while the 16-band spectra still
   match at its offset. Loudness alone could not place edges: tried and dropped 2026-10-16, each
   threshold fixed one case and broke another.
4. **Talk:** a recording can match in two places (an ad used in two breaks). The talk is the
   heaviest chain of stretches in order in both files; the rest are left out.

Needs ``numpy`` (the ``ml`` extra) and ``ffmpeg`` on PATH.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

SR = 8000
HOP = 160  # 20 ms
FPS = SR / HOP
NFFT = 512
BANDS = 16
WIN_S, STEP_S = 10.0, 5.0
SAME_OFFSET_S = 0.04
MIN_STRETCH_S = 3.0
SILENT_STD = 0.05
OVERLAP_S = 2.0


@dataclass(frozen=True)
class Stretch:
    """A part of the original found in the stream: original seconds [start, end), played
    ``offset`` seconds later in the stream (negative: earlier)."""

    start: float
    end: float
    offset: float

    @property
    def seconds(self) -> float:
        return self.end - self.start


# What a phone browser sends: hosts choose the ads per request, so a stream fetched for a map is
# fetched as a listener's would be.
PHONE_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1"
)


def decode(path: str, timeout_s: Optional[float] = None) -> "np.ndarray":
    """Mono float32 samples at 8 kHz, from a file or an http(s) URL."""
    import numpy as np

    cmd = ["ffmpeg", "-v", "error"]
    if path.startswith(("http://", "https://")):
        cmd += ["-user_agent", PHONE_UA]
    cmd += ["-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True, timeout=timeout_s).stdout
    return np.frombuffer(raw, dtype=np.float32)


def loudness(x: "np.ndarray") -> "np.ndarray":
    """log10 energy per 20 ms frame."""
    import numpy as np

    n = len(x) // HOP
    return np.log10(np.mean(x[: n * HOP].reshape(n, HOP).astype(np.float64) ** 2, axis=1) + 1e-9)


def spectra(x: "np.ndarray") -> "np.ndarray":
    """Per 20 ms frame, log energy in 16 bands (log-spaced), with the frame's mean removed: the
    spectrum's shape, not its level."""
    import numpy as np

    n = max(0, (len(x) - NFFT) // HOP + 1)
    out = np.empty((n, BANDS))
    window = np.hanning(NFFT)
    edges = np.geomspace(4, NFFT // 2, BANDS + 1).astype(int)
    for a in range(0, n, 4096):  # in blocks: a whole hour at once is ~1 GB of spectra
        z = min(n, a + 4096)
        idx = np.arange(NFFT)[None, :] + HOP * np.arange(a, z)[:, None]
        power = np.abs(np.fft.rfft(x[idx] * window, axis=1)) ** 2
        bands = np.stack(
            [power[:, lo : max(hi, lo + 1)].sum(1) for lo, hi in zip(edges[:-1], edges[1:])], 1
        )
        logb = np.log10(bands + 1e-9)
        out[a:z] = logb - logb.mean(1, keepdims=True)
    return out


def _ncc(needle: "np.ndarray", hay: "np.ndarray") -> "np.ndarray":
    """Normalized cross-correlation of ``needle`` at every offset of ``hay``."""
    import numpy as np

    m = len(needle)
    a = (needle - needle.mean()) / (needle.std() + 1e-9)
    nfft = 1 << int(np.ceil(np.log2(len(hay) + m)))
    corr = np.fft.irfft(np.fft.rfft(hay, nfft) * np.conj(np.fft.rfft(a, nfft)), nfft)
    corr = corr[: len(hay) - m + 1]
    cs = np.concatenate([[0.0], np.cumsum(hay)])
    cs2 = np.concatenate([[0.0], np.cumsum(hay * hay)])
    s, s2 = cs[m:] - cs[:-m], cs2[m:] - cs2[:-m]
    sd = np.sqrt(np.maximum(s2 / m - (s / m) ** 2, 1e-12))
    return corr / (m * sd)


def coarse(
    orig: "np.ndarray", stream: "np.ndarray", min_ncc: float = 0.9
) -> List[Tuple[float, float]]:
    """(original second, offset) for each 10 s window of the original found in the stream."""
    import numpy as np

    w, step = int(WIN_S * FPS), int(STEP_S * FPS)
    out: List[Tuple[float, float]] = []
    if len(stream) < w:
        return out
    for i in range(0, len(orig) - w, step):
        seg = orig[i : i + w]
        if seg.std() < SILENT_STD:
            continue  # silence matches anywhere
        c = _ncc(seg, stream)
        j = int(np.argmax(c))
        if c[j] < min_ncc:
            continue
        lo, hi = max(0, j - 25), min(len(stream) - w, j + 25)
        diffs = [float(np.mean(np.abs(seg - stream[k : k + w]))) for k in range(lo, hi + 1)]
        best = lo + int(np.argmin(diffs))
        out.append((i / FPS, (best - i) / FPS))
    return out


def _group(points: Sequence[Tuple[float, float]]) -> List[Stretch]:
    runs: List[List[float]] = []  # [start, end, offset, windows]
    for t, off in points:
        if runs and abs(runs[-1][2] - off) <= SAME_OFFSET_S:
            runs[-1][1] = t + WIN_S
            runs[-1][3] += 1
        else:
            runs.append([t, t + WIN_S, off, 1])
    return [Stretch(r[0], r[1], round(r[2], 2)) for r in runs if r[3] >= 2]


def _edges(
    o_spec: "np.ndarray", s_spec: "np.ndarray", run: Stretch, min_corr: float = 0.5
) -> Stretch:
    """Walk the stretch's edges out to the last frame where the spectra still match (a 0.5 s
    average of per-frame correlation >= ``min_corr``), allowing 1 s of mismatch inside."""
    import numpy as np

    off = int(round(run.offset * FPS))
    lo, hi = max(0, -off), min(len(o_spec), len(s_spec) - off)
    corr = np.full(len(o_spec), -1.0)
    if hi > lo:
        a = o_spec[lo:hi] - o_spec[lo:hi].mean(1, keepdims=True)
        b = s_spec[lo + off : hi + off] - s_spec[lo + off : hi + off].mean(1, keepdims=True)
        corr[lo:hi] = (a * b).sum(1) / (np.sqrt((a * a).sum(1) * (b * b).sum(1)) + 1e-9)
    k = int(0.5 * FPS)
    smooth = np.convolve(corr, np.ones(k) / k, mode="same")
    good = smooth >= min_corr
    hold = int(FPS)
    mid = min(len(good) - 1, int((run.start + run.end) / 2 * FPS))

    def walk(step: int) -> int:
        i, last, bad = mid, mid, 0
        while 0 <= i < len(good):
            if good[i]:
                last, bad = i, 0
            else:
                bad += 1
                if bad >= hold:
                    break
            i += step
        return last

    return Stretch(round(walk(-1) / FPS, 2), round((walk(1) + 1) / FPS, 2), run.offset)


def _merge(stretches: Sequence[Stretch]) -> List[Stretch]:
    """One stretch per offset and place: windows on either side of an inserted recording grow into
    the same stretch once their edges are walked out."""
    out: List[Stretch] = []
    for s in sorted(stretches, key=lambda s: (s.offset, s.start)):
        last = out[-1] if out else None
        if last and abs(last.offset - s.offset) <= SAME_OFFSET_S and s.start <= last.end:
            out[-1] = Stretch(last.start, max(last.end, s.end), last.offset)
        else:
            out.append(s)
    return out


def talk(stretches: Sequence[Stretch]) -> List[Stretch]:
    """The heaviest chain of stretches in order in both the original and the stream: a recording
    that also matches elsewhere (an ad reused in another break) falls out of the chain. Edges of
    neighbours can overlap by up to ``OVERLAP_S`` (a pause at the break matches either side); the
    shared edge goes to the middle of the overlap."""
    items = sorted(_merge(stretches), key=lambda s: s.start)
    best: List[float] = []
    prev: List[int] = []
    for i, s in enumerate(items):
        best.append(s.seconds)
        prev.append(-1)
        for j in range(i):
            p = items[j]
            in_order = p.start < s.start and p.end <= s.start + OVERLAP_S
            if in_order and p.end + p.offset <= s.start + s.offset + OVERLAP_S:
                if best[j] + s.seconds > best[i]:
                    best[i], prev[i] = best[j] + s.seconds, j
    if not items:
        return []
    i = max(range(len(items)), key=lambda k: best[k])
    chain: List[Stretch] = []
    while i >= 0:
        chain.append(items[i])
        i = prev[i]
    chain.reverse()
    for k in range(1, len(chain)):
        a, b = chain[k - 1], chain[k]
        if a.end > b.start:
            mid = round((a.end + b.start) / 2, 2)
            chain[k - 1], chain[k] = Stretch(a.start, mid, a.offset), Stretch(mid, b.end, b.offset)
    return chain


def align_samples(orig: "np.ndarray", stream: "np.ndarray") -> List[Stretch]:
    """The talk of ``orig`` found in ``stream``, with frame-exact edges, in order."""
    found = _group(coarse(loudness(orig), loudness(stream)))
    o_spec, s_spec = spectra(orig), spectra(stream)
    edged = [_edges(o_spec, s_spec, r) for r in found]
    return talk([s for s in edged if s.seconds >= MIN_STRETCH_S])


def align_files(original: str, stream: str) -> List[Stretch]:
    """``align_samples`` for two audio files (or a file and a URL ffmpeg can read)."""
    return align_samples(decode(original), decode(stream))


def time_map(stretches: Sequence[Stretch]) -> List[List[float]]:
    """As ``audio_time_map`` (pairs [transcript second, offset]): each stretch's offset from its
    start on, the first from 0."""
    out = [[round(s.start, 2), s.offset] for s in stretches]
    if out:
        out[0][0] = 0.0
    return out
