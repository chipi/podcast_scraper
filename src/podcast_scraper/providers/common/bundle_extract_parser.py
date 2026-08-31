"""Robust JSON parser for the bundled ``extract_quotes`` response (#698 Layer A).

The bundled call asks the LLM to return ``{insight_id_str: [quote_text, ...]}``
covering all N insights in one response. This parser tolerates:

- Code fences (`````json ... `````) wrapping the JSON.
- Top-level ``{"insights": {...}}`` envelope (some models add an outer key).
- Missing keys for some insight indices (returned as empty lists).
- Non-string ids (cast to str).
- Non-list values (treated as empty for that index).
- Per-quote dict shape ``{"text": "..."}`` instead of bare strings.

The same fallback policy as ``mega_bundled`` applies upstream: if this parser
returns nothing useful for an insight, the dispatcher falls back to the per-
insight staged path for that insight (or the whole batch on hard parse failure).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BundleExtractParseError(Exception):
    """Raised when the bundled extract response is unparsable as JSON.

    Carries structured diagnostics because "the model emitted malformed JSON" and "the response
    was cut off mid-write" are different faults with different fixes, and the bare
    ``json.JSONDecodeError`` message reads identically for both.

    ``truncation_suspected`` means only THE DOCUMENT ENDED EARLY — it does NOT name the cause.
    The parser cannot know why: it never sees ``finish_reason``. Only the call site can pair the
    two, which is why the provider logs them together.

    MEASURED 2026-08-17, 15 episodes, live provider: ``finish_reason`` was ``stop`` on every
    single call, with responses up to 33,626 chars against an 8,192-token budget, and zero parse
    failures. The production failures were at ~10.6k chars — a third of a response that succeeds
    here. So an early-ending document on this stack is the MODEL emitting bad JSON, not the
    budget running out, and raising ``max_tokens`` would not have fixed it. An earlier version of
    this docstring assumed the opposite.
    """

    def __init__(
        self,
        message: str,
        *,
        content_length: int = 0,
        error_position: Optional[int] = None,
        truncation_suspected: bool = False,
    ) -> None:
        super().__init__(message)
        self.content_length = content_length
        self.error_position = error_position
        self.truncation_suspected = truncation_suspected


class BundleOutputBudgetExceeded(BundleExtractParseError):
    """The response was cut off because ``max_tokens`` ran out — the batch was too big.

    Distinct from its parent for one reason: the fix is the CALLER'S, not another vendor's.
    ``_maybe_prefetch_bundled_candidates`` already bisects a failing batch and retries the
    halves, which resolves this exactly; failing over to a weaker model instead produces a
    worse answer to a question the strong model could have answered in two calls.

    Only raise this when ``finish_reason == "length"``. Do NOT infer it from
    ``truncation_suspected``: per the parent's docstring, a document that ends early is
    usually the model emitting bad JSON (15/15 measured calls on 2026-08-17 had
    ``finish_reason == "stop"``), and bisecting that would retry a prompt problem forever.

    ``caller_can_retry_smaller`` is the marker :mod:`summarization.fallback` reads to skip
    the failover chain. It lives on the exception rather than in a method denylist because
    the same method must still fail over when the endpoint is genuinely down.
    """

    #: Read by ``FallbackAwareSummarizationProvider._wrap_call`` — see that method.
    caller_can_retry_smaller = True


#: Decode failures that mean "the document ran out" BY THEMSELVES, with no position test.
#:
#: An unterminated string cannot happen in the middle of an otherwise complete document: once a
#: string never closes, the decoder consumes everything after it to the end. So the message
#: alone proves the document ended early.
#:
#: WATCH THE OFFSET. Python reports this error at the position where the string STARTED, not
#: where the content stopped — "Unterminated string starting at: line 1 column 8 (char 7)". The
#: four production failures in the 2026-08-16 acceptance run were logged as "at char 1364 /
#: 4986 / 10630 / 10992", which reads as a cutoff point and is not one. Those are opening
#: quotes. Treating them as cutoffs is what made the failures look like they happened at wildly
#: varying budgets and therefore "not a clean max_tokens cutoff" — a position test built on
#: them classifies backwards.
_DEFINITE_TRUNCATION_MARKERS = (
    "unterminated string",
    "unexpected end of data",
)

#: Failures that are truncation ONLY when they land at the end of the document. Each of these
#: can equally be a structural mistake the model made mid-document, so position decides.
_POSITIONAL_TRUNCATION_MARKERS = (
    "expecting ',' delimiter",
    "expecting ':' delimiter",
    "expecting value",
    "expecting property name",
)


def _looks_truncated(exc: json.JSONDecodeError, text: str) -> bool:
    """True when the decode failure is shaped like a cut-off response rather than bad JSON.

    The distinction is the whole point: a budget cutoff is fixed by raising max_tokens, bad
    JSON is not, and the raw decoder message reads the same for both.
    """
    msg = str(exc.msg).lower()
    if any(marker in msg for marker in _DEFINITE_TRUNCATION_MARKERS):
        return True
    if not any(marker in msg for marker in _POSITIONAL_TRUNCATION_MARKERS):
        return False
    # Within the last 5 % counts as "at the end", with a 16-char floor so a very short response
    # is not disqualified by arithmetic alone. The floor must stay SMALL: at 64 it exceeded the
    # length of any short document, so every short malformed response was mislabelled a budget
    # cutoff — sending an operator to raise max_tokens for a fault that cannot fix.
    tail_window = max(16, len(text) // 20)
    return exc.pos >= max(0, len(text) - tail_window)


def _strip_code_fences(content: str) -> str:
    text = (content or "").strip()
    if text.startswith("```"):
        # `````json\n...\n````` or `````\n...\n`````
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _coerce_quote_strings(raw: Any) -> List[str]:
    """Normalise per-insight quote payload to a list of non-empty strings."""
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        return [s] if s else []
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for item in raw:
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
            continue
        if isinstance(item, dict):
            text_val = item.get("text") or item.get("quote") or item.get("quote_text")
            if isinstance(text_val, str):
                s = text_val.strip()
                if s:
                    out.append(s)
    return out


def parse_bundled_extract_response(
    content: str,
    expected_count: int,
) -> Dict[int, List[str]]:
    """Parse the bundled extract response into ``{insight_idx: [quote_text, ...]}``.

    Args:
        content: Raw model response text (may include code fences).
        expected_count: Number of insights the bundled call covered. Used to
            seed the result with empty lists for missing indices so the caller
            can iterate uniformly.

    Returns:
        Dict mapping each insight index in ``range(expected_count)`` to its
        list of quote strings (possibly empty).

    Raises:
        BundleExtractParseError: When the content is not valid JSON or the
            top-level shape isn't a mapping. Callers should fall back to the
            staged extract path for the whole batch.
    """
    if expected_count <= 0:
        return {}

    text = _strip_code_fences(content)
    if not text:
        raise BundleExtractParseError("empty content", content_length=len(content or ""))

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        truncated = _looks_truncated(exc, text)
        # The classification goes in the MESSAGE, not only in the attributes: every provider's
        # ``extract_quotes_bundled`` already logs the exception, and the failover wrapper
        # surfaces it as "Primary error: ...". Putting it here upgrades all of those call sites
        # at once, without six identical edits that could drift apart.
        raise BundleExtractParseError(
            f"invalid JSON: {exc} " f"[chars={len(text)} fail_at={exc.pos} "
            # Names the SHAPE, not the cause: pair with finish_reason at the call site.
            # "length" -> budget cutoff, raise max_tokens. "stop" -> the model emitted bad JSON,
            # and on this stack that is what 15/15 measured calls showed.
            f"diagnosis={'DOCUMENT_ENDED_EARLY' if truncated else 'MALFORMED_MID_DOCUMENT'}]",
            content_length=len(text),
            error_position=exc.pos,
            truncation_suspected=truncated,
        ) from exc

    if not isinstance(obj, dict):
        raise BundleExtractParseError(
            f"top-level must be an object, got {type(obj).__name__}",
            content_length=len(text),
        )

    # Tolerate envelope: ``{"insights": {...}}`` or ``{"quotes": {...}}``.
    inner = obj
    for envelope_key in ("insights", "quotes", "by_insight", "results"):
        v = obj.get(envelope_key)
        if isinstance(v, dict):
            inner = v
            break

    out: Dict[int, List[str]] = {idx: [] for idx in range(expected_count)}
    for raw_key, raw_val in inner.items():
        try:
            idx = int(str(raw_key).strip())
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= expected_count:
            continue
        out[idx] = _coerce_quote_strings(raw_val)

    return out
