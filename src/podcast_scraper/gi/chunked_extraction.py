"""Extract insights in passes over the transcript, not one pass over the whole thing.

Local models saturate per CALL, not per episode. qwen3.5:35b emits a roughly constant number of
insights however long the episode is (+2.3 insights going from a <40k to a >=40k transcript) while
gemini-2.5-flash-lite scales with the material (+6.6). Asked for 12 / 16 / 20 / 25 insights, qwen
returns 11.7 / 15.3 / 18.0 / 18.0 — it stops at about eighteen and will not go further, whatever
the prompt says.

Context is not the constraint: a 90k-char transcript fits inside qwen's window with room to spare.
The ceiling is per-call. So give it more calls.

Measured on 65-77k-char episodes (the 45-90 minute format):

    mode      insights   CORE   USEFUL+   grounding
    1 call        24.7   10.7      20.0    80-97%
    3 chunks      56.0   17.3      40.3    96-98%

The extra insights are distinct (semantic dedup barely removes any), they are genuinely
substantive (a blind judge scores CORE +62%), and they ground BETTER than single-pass ones — a
chunk-local insight sits closer to the passage that supports it, so its quote is easier to find
verbatim.

Filler rises too (4.7 -> 15.7 per episode), which is what the value gate is for. Trimming filler is
cheap; nothing recovers knowledge that was never extracted.

Chunking scales with length: a 30-minute episode is one pass and pays nothing.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# Below this, chunking cannot help: the model is nowhere near its per-call ceiling.
MIN_CHARS_TO_CHUNK = 40_000
# Never split into more passes than this — cost grows linearly and the tail chunks get thin.
MAX_CHUNKS = 6

_encoder_cache: dict = {}
_encoder_lock = threading.Lock()


def _encoder(model_id: str = "sentence-transformers/all-MiniLM-L6-v2") -> Any:
    enc = _encoder_cache.get(model_id)
    if enc is not None:
        return enc
    with _encoder_lock:
        # Re-check under the lock: concurrent episodes must not each construct one (torch's lazy
        # init races when they do, and it costs the episode its insights).
        enc = _encoder_cache.get(model_id)
        if enc is None:
            from sentence_transformers import SentenceTransformer

            enc = SentenceTransformer(model_id)
            _encoder_cache[model_id] = enc
    return enc


def plan_chunks(text: str, chunk_chars: int) -> int:
    """How many passes this transcript warrants. 1 means do not chunk."""
    if chunk_chars <= 0 or not text or len(text) < MIN_CHARS_TO_CHUNK:
        return 1
    return max(1, min(MAX_CHUNKS, math.ceil(len(text) / chunk_chars)))


def split(text: str, n: int) -> List[str]:
    """Split into ``n`` pieces without cutting mid-sentence.

    Prefers line boundaries (a diarized transcript has one line per speaker turn). Falls back to
    sentence boundaries when the transcript has too few lines — some transcripts are written as a
    single unbroken string, and silently declining to chunk those would be exactly the kind of
    quiet no-op this codebase keeps producing.
    """
    if n <= 1:
        return [text]

    lines = text.splitlines()
    if len(lines) >= n:
        per = math.ceil(len(lines) / n)
        by_line = ["\n".join(lines[i : i + per]) for i in range(0, len(lines), per)]
        return [c for c in by_line if c.strip()]

    # Too few lines to split on: cut on sentence ends near each target offset instead.
    target = math.ceil(len(text) / n)
    out: List[str] = []
    start = 0
    for _ in range(n - 1):
        want = start + target
        if want >= len(text):
            break
        cut = -1
        for mark in (". ", "? ", "! "):
            found = text.rfind(mark, start, want)
            cut = max(cut, found + len(mark) if found != -1 else -1)
        if cut <= start:
            cut = want  # no sentence end in range: fall back to the hard offset
        out.append(text[start:cut])
        start = cut
    out.append(text[start:])
    return [c for c in out if c.strip()]


def dedupe(texts: List[str], threshold: float) -> List[str]:
    """Drop insights that restate one already kept.

    Chunks overlap in subject matter even when they do not overlap in text, so the merged list has
    to be deduplicated on meaning rather than on string equality. In practice this removes very
    little (57 -> 57, 52 -> 52 on real episodes), which is itself the evidence that chunking finds
    new knowledge rather than restating the same claims three times.
    """
    if len(texts) < 2 or threshold >= 1.0:
        return texts
    try:
        import numpy as np

        emb = _encoder().encode(texts, normalize_embeddings=True)
    except Exception as exc:  # noqa: BLE001 — dedup is an optimisation, never a hard failure
        logger.warning(
            "insight dedup unavailable (%s); keeping all %d", type(exc).__name__, len(texts)
        )
        return texts

    kept: List[str] = []
    kept_emb: List[Any] = []
    for i, t in enumerate(texts):
        e = np.asarray(emb[i])
        if kept_emb and max(float(np.dot(e, k)) for k in kept_emb) >= threshold:
            continue
        kept.append(t)
        kept_emb.append(e)
    return kept


def generate_chunked(
    generate: Any,
    text: str,
    *,
    episode_title: Optional[str],
    max_insights: int,
    chunk_chars: int,
    dedupe_threshold: float,
    pipeline_metrics: Optional[Any] = None,
) -> List[Any]:
    """Run ``generate`` over successive slices of the transcript and merge the results.

    Falls back to a single whole-transcript call whenever chunking is off, the episode is short, or
    the chunked path yields nothing — never returns less than the unchunked path would.
    """
    n = plan_chunks(text, chunk_chars)
    if n == 1:
        return list(
            generate(
                text=text,
                episode_title=episode_title,
                max_insights=max_insights,
                params=None,
                pipeline_metrics=pipeline_metrics,
            )
            or []
        )

    merged: List[Any] = []
    for idx, piece in enumerate(split(text, n)):
        try:
            got = generate(
                text=piece,
                episode_title=episode_title,
                max_insights=max_insights,
                params=None,
                pipeline_metrics=pipeline_metrics,
            )
        except Exception as exc:  # noqa: BLE001 — one bad chunk must not cost the episode
            logger.warning(
                "insight chunk %d/%d failed (%s); continuing", idx + 1, n, type(exc).__name__
            )
            continue
        merged.extend(got or [])

    if not merged:
        logger.warning("chunked extraction produced nothing; falling back to a single pass")
        return list(
            generate(
                text=text,
                episode_title=episode_title,
                max_insights=max_insights,
                params=None,
                pipeline_metrics=pipeline_metrics,
            )
            or []
        )

    texts = [m if isinstance(m, str) else str((m or {}).get("text", "")) for m in merged]
    keep = dedupe([t for t in texts if t.strip()], dedupe_threshold)
    dropped = len(texts) - len(keep)
    logger.info(
        "chunked extraction: %d chars -> %d passes -> %d insights (%d duplicates removed)",
        len(text),
        n,
        len(keep),
        dropped,
    )
    _bump(pipeline_metrics, "gi_insight_chunks", n)
    _bump(pipeline_metrics, "gi_insights_deduped", dropped)
    return keep


def _bump(metrics: Optional[Any], name: str, amount: int) -> None:
    if metrics is None or not amount:
        return
    try:
        setattr(metrics, name, getattr(metrics, name, 0) + amount)
    except Exception:  # noqa: BLE001
        pass
