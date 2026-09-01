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
import re
import threading
from collections import Counter
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

            from ..config_constants import get_pinned_revision_for_model
            from ..providers.ml.embedding_device import resolve_embedding_device

            # Explicit device (never mps) — auto-detect picks MPS on macOS, which SIGSEGVs.
            enc = SentenceTransformer(
                model_id,
                device=resolve_embedding_device(),
                revision=get_pinned_revision_for_model(model_id),
            )
            _encoder_cache[model_id] = enc
    return enc


def plan_chunks(text: str, chunk_chars: int) -> int:
    """How many passes this transcript warrants. 1 means do not chunk."""
    if chunk_chars <= 0 or not text or len(text) < MIN_CHARS_TO_CHUNK:
        return 1
    return max(1, min(MAX_CHUNKS, math.ceil(len(text) / chunk_chars)))


def per_chunk_budget(max_insights: int, chunks: int) -> int:
    """Split an EPISODE insight budget across chunks, instead of handing each chunk the whole one.

    ``max_insights`` is an episode-level ceiling — it comes from ``duration_scaled_max_insights``,
    which has *already* scaled it by episode length. Passing that same number to every chunk
    multiplied it by the chunk count, and the chunk count is itself derived from episode length.
    Duration was therefore counted twice and ``gi_max_insights: 50`` meant 50 nowhere:

        transcript   chunks   cap/chunk   effective ceiling
            52k         2         50            100
           120k         4        125            500
           200k         6        200           1200

    Measured consequence on the 2026-08-31 batch: a median of 79.5 insights per episode against
    a configured 50, max 157. That over-count is not just wasted generation — insight count
    drives the per-insight downstream fan-out, and quote extraction is 72% of all input tokens
    (r=0.58 insights vs quote calls, r=0.76 vs entailment calls, measured over 71 episodes).

    Rounded UP so the pieces still cover the ceiling: with 50 over 4 chunks a floor would give
    12*4 = 48 and quietly under-run the configured budget. The ceiling is a limit, not a target,
    so a small overshoot is the right side to err on. Always at least 1 — a chunk allowed zero
    insights is a pass that cannot contribute, which is worse than a slightly loose bound.
    """
    return max(1, math.ceil(max(1, int(max_insights)) / max(1, int(chunks))))


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


#: Tokens carrying no distinguishing meaning. Dropped before comparison so two sentences are
#: judged on what they SAY, not on how much English scaffolding they share.
_STOPWORDS = frozenset(
    """a an and are as at be been being but by can could did do does for from had has have
    he her his in into is it its of on or she should so than that the their them these they
    this those to was were will with would""".split()
)

_WORD_RE = re.compile(r"[a-z0-9']+")

#: Lexical-similarity bar for calling two insights the same claim.
#:
#: NOT the same scale as the embedding threshold, and deliberately not sharing its config value:
#: embedding cosine puts a genuine paraphrase around 0.85-0.95, while term-frequency cosine puts
#: the SAME pair far lower because it only sees shared words.
#:
#: Set at 0.90, MEASURED over all 14 episodes of the 2026-08-16 acceptance corpus (14,539
#: surviving insight pairs). The earlier 0.99 came from short episodes only, where no pair
#: scored above 0.60; long episodes have maximal chunk overlap and tell a different story.
#:
#: The whole distribution, so the choice can be re-argued rather than trusted:
#:
#:     [0.95, 0.99)   4 pairs   ALL duplicates   verb swaps: "says"/"argues"/"claims"
#:     [0.90, 0.95)   1 pair    duplicate        "was right" vs "was justified"
#:     [0.85, 0.90)   0 pairs
#:     [0.70, 0.85)   1 pair    duplicate        full paraphrase, cos 0.7108
#:     [0.00, 0.70)   14,533 pairs
#:
#: Every pair in [0.85, 0.99) was a genuine duplicate — 5 of 5, no false positives. Below them
#: is a wide EMPTY gap: the next pair down is 0.7108. A bar anywhere in 0.72-0.93 separates the
#: two populations on this corpus; 0.90 sits inside that gap with ~0.037 of margin above and
#: ~0.23 below, so it is not balanced on a single observation.
#:
#: The antonym case that justified 0.99 survives the change, which is why 0.90 and not lower:
#:
#:     "Kalanick believes regulation will SLOW autonomous vehicle deployment."
#:     "Kalanick believes regulation will ACCELERATE autonomous vehicle deployment."
#:
#: Six shared tokens, one different, cosine 0.857 — below this bar, so it is still NOT merged.
#: A bag of words cannot see polarity, and that example is synthetic; the five duplicates are
#: real. Keeping the bar above 0.857 honours the polarity risk without paying for it with
#: measured recall.
#:
#: WHAT THIS STILL MISSES, quantified rather than waved at: duplicates continue below the gap —
#: the 0.7108 pair above, and a 0.6390 pair ("the value of rich data grows combinatorially, not
#: linearly or exponentially" vs "the value of rich telemetry data goes up combinatorially, not
#: linearly or even exponentially") are the same claim in different words. Catching those means
#: reaching into a band holding 14,533 mostly-distinct pairs, which no bag-of-words threshold
#: can do safely. That is the embedding tier's job and the honest ceiling of this method: a
#: recall gap, not a correctness one. Since dropping a distinct insight destroys knowledge while
#: keeping a duplicate merely repeats it, the bar stays where the false positives stop.
DEFAULT_LEXICAL_DEDUPE_THRESHOLD = 0.90


def _content_tokens(text: str) -> List[str]:
    return [t for t in _WORD_RE.findall(text.lower()) if t not in _STOPWORDS]


def _tf_cosine(a: Counter, b: Counter) -> float:
    """Cosine over term-frequency vectors. Pure stdlib — no model, no wheel, no network."""
    shared = set(a) & set(b)
    if not shared:
        return 0.0
    dot = sum(a[t] * b[t] for t in shared)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _lexical_duplicates(texts: List[str], threshold: float) -> List[str]:
    """Keep the first of any group of insights that say the same thing in the same words.

    Runs ALWAYS, because it needs nothing but the standard library. The embedding deduper it
    backs up lives in ``sentence-transformers``, an optional dependency, and every environment
    that lacks it used to get NO dedup at all: the call logged "insight dedup unavailable
    (ModuleNotFoundError); keeping all N" and shipped whatever the model emitted.

    WHERE THAT ACTUALLY BITES — corrected 2026-08-16, having first written the wrong thing
    here. The original note claimed ``sentence-transformers`` is absent from ``[llm]``, "the
    extra the production pipeline image is built with", and concluded dedup never ran in
    production. That is false. ``docker/pipeline/Dockerfile`` installs
    ``.[llm,search,sentry,langfuse]`` for the llm variant, ``[search]`` pins
    ``sentence-transformers>=5.6.0``, and the runtime stage copies the whole site-packages tree
    across — so the production image HAS the embedding tier.

    The observation behind the wrong claim was real but local: the acceptance run executed FROM
    SOURCE on a macOS x86_64 box where torch/lancedb publish no wheels, so the ML extras cannot
    install there at all. A true statement about one dev machine was generalised into a false
    one about production.

    The lexical tier is still worth having on its own terms — it needs no wheel, no model and no
    network, so it is the only tier that runs everywhere, including that dev box and any
    air-gapped or minimal deployment. It is a floor, not a replacement.

    Two passes, cheapest first:

    1. exact match on the normalised token sequence — zero false positives by construction, and
       the shape actually observed in the wild ("emitted verbatim twice");
    2. term-frequency cosine above ``threshold`` — catches the same claim reworded.

    Conservative on purpose: dropping a distinct insight destroys knowledge, while keeping a
    duplicate merely repeats it. When the two errors are not symmetric, the threshold belongs
    nowhere near the middle.
    """
    kept: List[str] = []
    kept_norm: set = set()
    kept_counters: List[Counter] = []

    for text in texts:
        tokens = _content_tokens(text)
        signature = " ".join(tokens)
        if signature and signature in kept_norm:
            continue
        counter = Counter(tokens)
        if counter and any(_tf_cosine(counter, k) >= threshold for k in kept_counters):
            continue
        kept.append(text)
        if signature:
            kept_norm.add(signature)
        kept_counters.append(counter)
    return kept


def dedupe(
    texts: List[str],
    threshold: float,
    *,
    lexical_threshold: float = DEFAULT_LEXICAL_DEDUPE_THRESHOLD,
) -> List[str]:
    """Drop insights that restate one already kept.

    Chunks overlap in subject matter even when they do not overlap in text, so the merged list
    has to be deduplicated on meaning rather than on string equality. In practice this removes
    very little (57 -> 57, 52 -> 52 on real episodes), which is itself the evidence that chunking
    finds new knowledge rather than restating the same claims three times.

    Two tiers, and the lexical one is not optional:

    * LEXICAL (always) — stdlib only, so it works on the ``[llm]`` production image where the
      embedding model does not exist. This is the tier that actually runs in production.
    * EMBEDDING (when available) — better semantic recall, catches a restatement that shares no
      vocabulary. Used on ``[ml]``/``[search]`` images. Its absence is now a downgrade in recall
      rather than a total loss of the feature, so it is logged at DEBUG, not WARNING.

    ``threshold >= 1.0`` disables the semantic tier, matching the documented "off" switch; exact
    restatements are still collapsed, because emitting one claim twice is never intended.
    """
    if len(texts) < 2:
        return texts

    kept = _lexical_duplicates(texts, lexical_threshold if threshold < 1.0 else 1.01)
    if threshold >= 1.0:
        return kept

    try:
        import numpy as np

        emb = _encoder().encode(kept, normalize_embeddings=True)
    except Exception as exc:  # noqa: BLE001 — dedup is an optimisation, never a hard failure
        logger.debug(
            "insight dedup: embedding tier unavailable (%s); lexical tier kept %d of %d",
            type(exc).__name__,
            len(kept),
            len(texts),
        )
        return kept

    final: List[str] = []
    kept_emb: List[Any] = []
    for i, t in enumerate(kept):
        e = np.asarray(emb[i])
        if kept_emb and max(float(np.dot(e, k)) for k in kept_emb) >= threshold:
            continue
        final.append(t)
        kept_emb.append(e)
    return final


def _as_insight_list(got: Any) -> List[Any]:
    """Accept a provider's return only if it is actually a sequence of insights.

    This was ``list(got or [])``, and ``list()`` on a mapping yields its KEYS. A provider that
    answered ``{"insights": [...]}`` — the shape half of them use — was therefore silently
    turned into the single insight ``"insights"``, which then flowed through classification and
    grounding as if a model had said it. Wrong in the worst direction: not a visible failure,
    but a plausible-looking artifact built from a dict's key names.

    Anything that is not a list/tuple returns empty, which the caller reports as an empty
    extraction with a reason rather than inventing content from the container.
    """
    if isinstance(got, (list, tuple)):
        return list(got)
    if got:
        logger.warning(
            "insight provider returned %s, not a list of insights; treating as no insights",
            type(got).__name__,
        )
    return []


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
        return _as_insight_list(
            generate(
                text=text,
                episode_title=episode_title,
                max_insights=max_insights,
                params=None,
                pipeline_metrics=pipeline_metrics,
            )
        )

    # The episode budget is DIVIDED across the passes, not handed whole to each — see
    # ``per_chunk_budget``. Handing each chunk the episode ceiling multiplied it by the chunk
    # count, which is itself derived from episode length, so duration was counted twice.
    chunk_cap = per_chunk_budget(max_insights, n)
    logger.info(
        "chunked extraction: episode ceiling %d over %d passes -> %d insights per pass",
        max_insights,
        n,
        chunk_cap,
    )

    merged: List[Any] = []
    for idx, piece in enumerate(split(text, n)):
        try:
            got = generate(
                text=piece,
                episode_title=episode_title,
                max_insights=chunk_cap,
                params=None,
                pipeline_metrics=pipeline_metrics,
            )
        except Exception as exc:  # noqa: BLE001 — one bad chunk must not cost the episode
            logger.warning(
                "insight chunk %d/%d failed (%s); continuing", idx + 1, n, type(exc).__name__
            )
            continue
        # Same guard as the unchunked path: a mapping here would extend `merged` with its keys.
        merged.extend(_as_insight_list(got))

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
