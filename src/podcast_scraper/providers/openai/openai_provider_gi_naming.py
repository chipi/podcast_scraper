"""OpenAICompatibleProvider's GI and naming methods (E04 split line, S1).

Moved verbatim out of ``OpenAICompatibleProvider`` (``openai_provider``), which inherits this
class: insight extraction, grounding and its quotes, entailment scoring, insight
classification, speaker and host detection, and the bundled summary calls that carry
naming fields. They leave public with GI and naming in the core release; the provider
keeps its call surface.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from ... import config_constants, models

if TYPE_CHECKING:
    from ...models import Episode
else:
    Episode = models.Episode  # type: ignore[assignment]
from ...utils.log_redaction import format_exception_for_log
from ...utils.provider_metadata import warn_if_truncated
from ...utils.timeout_config import get_small_chat_call_timeout
from ...workflow import metrics
from .. import guardrails as _guardrails, insight_salvage as _insight_salvage
from ..ml.speaker_detection import DEFAULT_SPEAKER_NAMES
from ._openai_provider_calls import (
    _openai_chat_usage_tokens,
    _openai_response_cost_usd,
    _repetition_signal,
)

# The provider module's logger (one logger object): records keep their logger name.
logger = logging.getLogger("podcast_scraper.providers.openai.openai_provider")


#: Cap on the raw model reply stored in a bundled-quote failure capture. A runaway reply is the
#: thing worth keeping and the biggest seen so far is ~24 KB; 64 KB leaves headroom without
#: letting one pathological reply fill the corpus.
_BUNDLE_CAPTURE_MAX_CHARS = 64_000


#: Where captures land, under the corpus. `.podcast_scraper/` is the established home for
#: internal files (corpus-art, audio-cache, audio-fingerprints) and is excluded from artifact
#: scans, so this cannot be mistaken for corpus content.
_BUNDLE_CAPTURE_DIR = ".podcast_scraper/quote-bundle-failures"


def _tail_degeneracy(text: str, window: int = 2_000) -> Dict[str, Any]:
    """Signals for the degenerate modes :func:`_repetition_signal` is BLIND to (#2053).

    That function splits on whitespace, so two classic runaway shapes score exactly 0 and would be
    misreported as "the model genuinely had more to say" — sending the fix in the wrong direction:

    * a **whitespace/newline tail** — the model stops emitting content but never emits EOS and pads
      to ``max_tokens``. This is the documented failure mode of schema-less
      ``response_format={"type": "json_object"}`` guided decoding, where the grammar admits only
      whitespace once the object closes. `320f2db0` ENABLED that guided decoding on this very call,
      so it is the mode most likely to appear next.
    * a **single repeated token with no spaces** — one enormous "word", so there are no whitespace
      n-grams to count at all.

    ``tail_nonspace_fraction`` near 0 is the first; ``tail_distinct_chars`` of 1-3 is the second.
    """
    tail = (text or "")[-window:]
    if not tail:
        return {"tail_nonspace_fraction": None, "tail_distinct_chars": None}
    nonspace = sum(1 for c in tail if not c.isspace())
    return {
        "tail_nonspace_fraction": round(nonspace / len(tail), 3),
        "tail_distinct_chars": len(set(tail)),
    }


#: A 12-gram repeated this often is a decoding loop. Healthy replies repeat one once or twice; the
#: 13 loops captured on 2026-10-02 ("like, you know, like, you know, ...") repeated 185-2,516 times.
_DECODING_LOOP_MIN_REPEATS = 50


_JSON_STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')


_BUNDLE_KEY_RE = re.compile(r'"(\d+)"\s*:\s*\[')


def _is_decoding_loop(text: str) -> bool:
    """True when *text* repeats one 12-word run often enough to be a loop, not real content."""
    return _repetition_signal(text)[0] >= _DECODING_LOOP_MIN_REPEATS


def _salvage_closed_quotes(text: str, insight_count: int) -> Dict[int, List[str]]:
    """The quote strings that CLOSED in a cut-off ``{"0": [...], "1": [...]}`` reply.

    The string the loop was writing never closes, so it is never matched; everything before it
    is a complete JSON string. Each kept quote still has to resolve verbatim against the
    transcript downstream, so a closed string that is itself filler is dropped there.
    """
    keys = [(m.start(), m.end(), int(m.group(1))) for m in _BUNDLE_KEY_RE.finditer(text)]
    out: Dict[int, List[str]] = {}
    for i, (_start, body_start, idx) in enumerate(keys):
        if not 0 <= idx < insight_count:
            continue
        body_end = keys[i + 1][0] if i + 1 < len(keys) else len(text)
        quotes: List[str] = []
        for m in _JSON_STRING_RE.finditer(text, body_start, body_end):
            try:
                quotes.append(json.loads(f'"{m.group(1)}"'))
            except ValueError:
                continue
        if quotes:
            out[idx] = quotes
    return out


#: Ceiling on the episode description handed to speaker detection (#2011).
#:
#: That call sends title + description + known hosts and asks for 300 output tokens — no
#: transcript. So the ONLY thing that can overflow a context window is the description, and some
#: publishers put an entire show-notes dump in it. Measured 2026-09-13 across the live corpus:
#:
#:     Latent Space        max 137,398 chars   median 41,590   (~39,000 tokens)
#:     Peter Attia           max 4,069
#:     Dwarkesh              max 4,024
#:     In Moscow's Shadows   max 1,919
#:
#: Against Qwen3-30B's 32,768-token window that is a guaranteed 400, and the episode then gets NO
#: speaker attribution at all — the stage fails outright rather than degrading.
#:
#: 8,000 chars (~2,300 tokens) is ~2x the largest description on any feed that has never
#: overflowed, so nothing that works today changes. It is generous for the purpose: speaker
#: detection is looking for NAMES, and show notes lead with the guest — the tail is timestamps,
#: sponsor copy and links.
#:
#: Unlike every other clip in this file, this one is NOT derived from the served window and must
#: not become so: the call sends no transcript, so 8,000 chars (~2,300 tokens) fits every window
#: in the fleet. Deriving it would make the bound grow with the window for no reason (#2050).
_SPEAKER_DESCRIPTION_MAX_CHARS = 8_000


def _clip_speaker_description(description: Optional[str], episode_title: str = "") -> str:
    """Bound the description so speaker detection cannot overflow the context window.

    Truncating loses nothing that matters here — see ``_SPEAKER_DESCRIPTION_MAX_CHARS``. The
    alternative is not "a longer prompt", it is a 400 and zero speakers for that episode.
    """
    text = description or ""
    if len(text) <= _SPEAKER_DESCRIPTION_MAX_CHARS:
        return text
    logger.info(
        "speaker detection: episode description clipped %d -> %d chars for %r (#2011); "
        "the full text would overflow the model context and fail the stage outright",
        len(text),
        _SPEAKER_DESCRIPTION_MAX_CHARS,
        (episode_title or "")[:60],
    )
    return text[:_SPEAKER_DESCRIPTION_MAX_CHARS]


def _capture_bundle_failure(
    *,
    cfg: Any,
    content: str,
    error: str,
    finish_reason: Optional[str],
    out_tok: Optional[int],
    max_out: Optional[int],
    insight_texts: List[str],
    model: Optional[str],
    prompt_chars: Optional[int] = None,
    in_tok: Optional[int] = None,
) -> None:
    """Persist one bundled-quote parse failure for offline diagnosis. Never raises.

    WHY: the WARNING above records that the reply was unparsable and whether it was budget-shaped,
    but NOT the reply itself — and the reply is the only thing that can tell a repetition loop
    apart from genuine over-generation. Those need opposite fixes (a decoding penalty vs a budget
    computed against the served context), and four attempts to reconstruct a failing prompt
    offline all produced healthy replies, because the batch that fails cannot be rebuilt from the
    artifact: `gi_require_grounding` drops the insights that lost their quotes, so the survivors
    are exactly the wrong sample.

    Capturing at the moment of failure sidesteps that entirely. Costs nothing when nothing fails.
    """
    try:
        out_dir = getattr(cfg, "output_dir", None)
        if not out_dir:
            return
        import hashlib

        dest = os.path.join(str(out_dir), _BUNDLE_CAPTURE_DIR)
        os.makedirs(dest, exist_ok=True)
        rep_count, rep_gram = _repetition_signal(content)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        digest = hashlib.sha256((content or "")[:4096].encode("utf-8", "replace")).hexdigest()[:10]
        payload = {
            "captured_at": stamp,
            "model": model,
            "finish_reason": finish_reason,
            "output_tokens": out_tok,
            "max_output_tokens": max_out,
            "insight_count": len(insight_texts or []),
            "prompt_chars": prompt_chars,
            "reply_chars": len(content or ""),
            "error": error,
            # The loop tell, precomputed so it survives truncation of `reply` below.
            "repeated_12gram_count": rep_count,
            "repeated_12gram": rep_gram,
            # The two shapes the 12-gram signal cannot see — see _tail_degeneracy.
            **_tail_degeneracy(content),
            # The server's OWN prompt-token count, so every capture yields a real chars/token
            # ratio for prod content. The budget derives at 3.5 (CHARS_PER_TOKEN_BUDGET_RATIO)
            # and nobody has measured it on screenplay+diarized transcripts — this settles it
            # with data instead of another argument.
            "prompt_tokens_reported": in_tok,
            "measured_chars_per_token": (
                round(prompt_chars / in_tok, 2) if (prompt_chars and in_tok) else None
            ),
            "insight_texts": [t[:400] for t in (insight_texts or [])],
            "reply_truncated_for_capture": len(content or "") > _BUNDLE_CAPTURE_MAX_CHARS,
            "reply": (content or "")[:_BUNDLE_CAPTURE_MAX_CHARS],
        }
        path = os.path.join(dest, f"{stamp}-{digest}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        logger.info(
            "bundled-quote failure captured -> %s (reply %d chars, top 12-gram x%d)",
            os.path.join(_BUNDLE_CAPTURE_DIR, os.path.basename(path)),
            len(content or ""),
            rep_count,
        )
    except Exception as exc:  # noqa: BLE001 — diagnostics must never mask the parse failure
        logger.debug("bundled-quote failure capture skipped: %s", exc)


def _record_openai_summarization_call(
    response: Any,
    pipeline_metrics: Optional[Any],
    *,
    cfg: Any,
    model: str,
    provider_type: str = "openai",
) -> None:
    """Record one summarization LLM call into pipeline_metrics.

    Used by bundle-mode methods (summarize_bundled, summarize_mega_bundled,
    summarize_extraction_bundled) — these make one summary-pricing LLM call
    and historically skipped ``record_llm_summarization_call`` entirely, so
    cost/tokens vanished at pipeline level even though call_metrics saw them.

    ``provider_type`` names the cost/pricing namespace (ADR-147): a sibling
    (litellm/vllm/deepseek) passes its own ``_TELEMETRY_PROVIDER`` so cost is
    never mis-attributed to ``openai`` pricing.
    """
    if pipeline_metrics is None or not hasattr(pipeline_metrics, "record_llm_summarization_call"):
        return
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    in_raw = getattr(usage, "prompt_tokens", None)
    out_raw = getattr(usage, "completion_tokens", None)
    if not isinstance(in_raw, (int, float)) or not isinstance(out_raw, (int, float)):
        return
    in_tok = int(in_raw)
    out_tok = int(out_raw)
    from ...workflow.helpers import calculate_provider_cost

    cost = _openai_response_cost_usd(response) or calculate_provider_cost(
        cfg=cfg,
        provider_type=provider_type,
        capability="summarization",
        model=model,
        prompt_tokens=in_tok,
        completion_tokens=out_tok,
    )
    pipeline_metrics.record_llm_summarization_call(in_tok, out_tok, cost_usd=cost)


def _as_known_host(name: str, known_hosts: Set[str]) -> str:
    """``name`` in the feed's spelling when it is one of the feed's known hosts written another way
    (accents, case, or a given name only one known host has); otherwise ``name`` unchanged."""
    folded = _fold_name(name)
    for host in known_hosts:
        if _fold_name(host) == folded:
            return host
    if len(folded.split()) == 1:
        owners = [h for h in known_hosts if _fold_name(h).split(" ")[0] == folded]
        if len(owners) == 1:
            return owners[0]
    return name


def _fold_name(name: str) -> str:
    """A name without accents, case or extra spaces: "Nína Pániková" folds to "nina panikova"."""
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", name or "")
    bare = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(bare.casefold().split())


class OpenAICompatibleProviderGiNamingMethods:
    """OpenAICompatibleProvider's GI and naming methods; ``self`` is the provider."""

    def _build_speaker_detection_prompt(
        self,
        episode_title: str,
        episode_description: str | None,
        known_hosts: Set[str],
    ) -> str:
        """Build prompt for speaker detection using prompt_store.

        Args:
            episode_title: Episode title
            episode_description: Optional episode description
            known_hosts: Set of known host names

        Returns:
            Rendered prompt string
        """
        from ...prompts.store import render_prompt

        # Use prompt_store to load versioned prompt template
        prompt_name: str = getattr(self.cfg, f"{self._CONFIG_NS}_speaker_user_prompt")

        # Merge config params with template params
        template_params = {
            "episode_title": episode_title,
            "episode_description": _clip_speaker_description(episode_description, episode_title),
            "known_hosts": ", ".join(sorted(known_hosts)) if known_hosts else "",
        }
        template_params.update(self.cfg.ner_prompt_params)

        return render_prompt(prompt_name, **template_params)

    def _bundled_quote_response_format(self, num_insights: int, max_out: int) -> Dict[str, Any]:
        """``response_format`` for the bundled-quote call. JSON mode here; a server known to
        enforce schema bounds overrides it with a bounded schema (see ``VLLMProvider``)."""
        return {"type": "json_object"}

    def _parse_speakers_from_response(
        self, response_text: str, known_hosts: Set[str]
    ) -> Tuple[list[str], Set[str], bool]:
        """Parse speaker names from OpenAI API JSON response.

        Args:
            response_text: JSON response from OpenAI API
            known_hosts: Set of known host names (for filtering)

        Returns:
            Tuple of (speaker_names_list, detected_hosts_set, detection_succeeded)
        """
        # The model's raw answer, for the episode's naming decision trace (#2276).
        self.last_speaker_detection_raw = response_text
        self.last_speaker_detection_stated_only: List[str] = []
        try:
            data = json.loads(response_text)

            # Extract speakers, hosts, and guests from JSON
            all_speakers = data.get("speakers", [])
            detected_hosts_list = data.get("hosts", [])
            guests_list = data.get("guests", [])

            # Normalize names (strip whitespace, filter empty)
            all_speakers = [name.strip() for name in all_speakers if name.strip()]
            detected_hosts_list = [name.strip() for name in detected_hosts_list if name.strip()]
            guests_list = [name.strip() for name in guests_list if name.strip()]

            # A name the model writes is the feed's KNOWN host when it is that host in another
            # spelling — accents or case ("Nina Pániková" for the stated "Nina Panikova"). It takes
            # the feed's spelling; kept apart it became a second spare guest name and cost the real
            # guest his name (gold dev c085, #2276).
            # So is a bare given name only one known host has ("Elad" for "Elad Gil").
            all_speakers = [_as_known_host(n, known_hosts) for n in all_speakers]
            detected_hosts_list = [_as_known_host(n, known_hosts) for n in detected_hosts_list]
            guests_list = [_as_known_host(n, known_hosts) for n in guests_list]

            # A host the model names is TRUSTED as a host only if the feed already states it, and a
            # host it names that the feed does not state is dropped: kept as a stated person, it
            # became a guest candidate and cost a real host or guest their name (gold val: Kenny
            # Malone unnamed, Kris Maher named after a reporter the model called "host"; #2276).
            detected_hosts = {host for host in detected_hosts_list if host in known_hosts}
            speaker_names = list(detected_hosts) + guests_list

            # A person the model lists among the SPEAKERS but neither as host nor guest is a person
            # the episode states — the reporter who files the story, the guest it did not label —
            # and dropping them lost real names. Kept apart from the guest list, though: they are
            # stated (an LLM-resolver candidate, counted when unplaced) but never corroborated, so
            # no count-based placement can paint them on a voice. Gold val, same answers: 4 better,
            # 0 worse; as corroborated guests: 5 better, 2 worse (#2276).
            labelled = {*detected_hosts_list, *guests_list, *known_hosts}
            self.last_speaker_detection_stated_only = list(
                dict.fromkeys(n for n in all_speakers if n not in labelled)
            )

            # Ensure we have at least MIN_SPEAKERS_REQUIRED speakers
            min_speakers = getattr(self.cfg, "screenplay_num_speakers", 2)
            if len(speaker_names) < min_speakers:
                # Add default speakers if needed
                defaults_needed = min_speakers - len(speaker_names)
                speaker_names.extend(DEFAULT_SPEAKER_NAMES[:defaults_needed])

            # Detection succeeded if we have real names (not just defaults)
            detection_succeeded = bool(detected_hosts or guests_list or (len(all_speakers) > 0))

            # NOT truncated to `min_speakers` — see `_build_speaker_names_list` (#2095/#2078).
            # That value is the SCREENPLAY's seat count, and applying it here deleted people the
            # episode states: on a two-host show the guest fell off the end and reached neither
            # `detected_guests` nor `metadata_named`, so the record builder never saw them at all.
            # This provider is the one production runs (`prod_dgx_full` -> `vllm`, which inherits
            # this method), so capping here is exactly where the bug bit hardest. The seat cap now
            # lives at the screenplay formatter, which is the only consumer with seats.
            return speaker_names, detected_hosts, detection_succeeded

        except (json.JSONDecodeError, KeyError, AttributeError) as exc:
            logger.warning(
                "Failed to parse OpenAI response as JSON: %s", format_exception_for_log(exc)
            )
            logger.debug("Response text: %s", response_text[:200])
            # Fallback: try to extract names from text response
            return self._parse_speakers_from_text(response_text, known_hosts)

    def _parse_speakers_from_text(
        self, response_text: str, known_hosts: Set[str]
    ) -> Tuple[list[str], Set[str], bool]:
        """Fallback: Parse speaker names from text response (not JSON).

        Args:
            response_text: Text response from OpenAI API
            known_hosts: Set of known host names

        Returns:
            Tuple of (speaker_names_list, detected_hosts_set, detection_succeeded)
        """
        # Try to extract names using simple patterns
        # Look for common patterns like "Speakers: Name1, Name2" or "Hosts: Name1"
        names = set()

        # Pattern: "speakers": ["Name1", "Name2"]
        json_pattern = r'"speakers"\s*:\s*\[(.*?)\]'
        match = re.search(json_pattern, response_text, re.IGNORECASE)
        if match:
            names_str = match.group(1)
            # Extract quoted strings
            quoted_names = re.findall(r'"([^"]+)"', names_str)
            names.update(quoted_names)

        # Pattern: "Hosts: Name1, Name2"
        hosts_pattern = r"hosts?\s*:\s*([^\n]+)"
        match = re.search(hosts_pattern, response_text, re.IGNORECASE)
        if match:
            hosts_str = match.group(1)
            host_names = [n.strip() for n in re.split(r"[,;]", hosts_str)]
            names.update(host_names)

        # Pattern: "Guests: Name1, Name2"
        guests_pattern = r"guests?\s*:\s*([^\n]+)"
        match = re.search(guests_pattern, response_text, re.IGNORECASE)
        if match:
            guests_str = match.group(1)
            guest_names = [n.strip() for n in re.split(r"[,;]", guests_str)]
            names.update(guest_names)

        # Filter out generic names
        names = {n for n in names if n.lower() not in ("host", "guest", "speaker")}

        # Separate hosts and guests
        detected_hosts = {name for name in names if name in known_hosts}
        guests = [name for name in names if name not in known_hosts]

        # Build speaker names list
        speaker_names = list(detected_hosts) + guests
        min_speakers = getattr(self.cfg, "screenplay_num_speakers", 2)
        if len(speaker_names) < min_speakers:
            speaker_names.extend(DEFAULT_SPEAKER_NAMES[: min_speakers - len(speaker_names)])

        detection_succeeded = bool(detected_hosts or guests)

        # Not truncated — same reason as the JSON path above (#2095/#2078): `min_speakers` is the
        # screenplay's seat count and must not delete people the episode states.
        return speaker_names, detected_hosts, detection_succeeded

    def _score_entailment_bundled_chunk(
        self,
        chunk_pairs: List[Tuple[str, str]],
        pipeline_metrics: Optional[Any],
    ) -> Dict[int, float]:
        """One bundled NLI OpenAI call for up to ``chunk_size`` pairs."""
        from ...providers.common.bundle_nli_parser import (
            BundleNliParseError,
            parse_bundled_nli_response,
        )
        from ...providers.common.bundled_prompts import (
            score_entailment_bundled_max_tokens,
            SCORE_ENTAILMENT_BUNDLED_SYSTEM,
            score_entailment_bundled_user,
        )
        from ...utils.provider_metrics import (
            _safe_openai_retryable,
            apply_gil_evidence_llm_call_metrics,
            merge_gil_evidence_call_metrics_on_failure,
            ProviderCallMetrics,
            retry_with_metrics,
        )

        system = SCORE_ENTAILMENT_BUNDLED_SYSTEM
        user = score_entailment_bundled_user(chunk_pairs)

        call_metrics = ProviderCallMetrics()
        call_metrics.set_provider_name(self._TELEMETRY_PROVIDER)
        call_metrics.set_breaker_config_from_cfg(self.cfg)
        max_out = score_entailment_bundled_max_tokens(len(chunk_pairs))

        def _make_api_call() -> Any:
            return self._chat_create(
                model=self.summary_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                **self._token_kwarg(max_out),
                _lf_stage="score_entailment",
            )

        try:
            response = retry_with_metrics(
                _make_api_call,
                max_retries=3,
                initial_delay=1.0,
                max_delay=30.0,
                retryable_exceptions=_safe_openai_retryable(),
                metrics=call_metrics,
            )
        except Exception:
            merge_gil_evidence_call_metrics_on_failure(call_metrics, pipeline_metrics)
            raise
        in_tok, out_tok = _openai_chat_usage_tokens(response)
        apply_gil_evidence_llm_call_metrics(
            call_metrics,
            pipeline_metrics,
            in_tok,
            out_tok,
            cfg=self.cfg,
            provider_type=self._TELEMETRY_PROVIDER,
            model=self.summary_model,
            stage="score_entailment",
        )

        content = (response.choices[0].message.content or "").strip()
        try:
            return parse_bundled_nli_response(content, expected_count=len(chunk_pairs))
        except BundleNliParseError as exc:
            logger.debug("OpenAI score_entailment_bundled parse failed: %s", exc)
            raise

    def classify_insights(self, insights: List[str]) -> List[int]:
        """Tier each insight 0-3 for the value gate (see ``gi.value_gate``).

        One call for the whole episode. Raises on failure — the gate fails open and owns the
        decision to keep everything, so this must not paper over a bad response.
        """
        if not insights:
            return []
        if not self._summarization_initialized:
            raise RuntimeError("OpenAI summarization not initialized for classify_insights")

        import json as _json

        from ...prompts.store import render_prompt

        listing = "\n".join(f"i{idx}: {text}" for idx, text in enumerate(insights))
        user_prompt = render_prompt("openai/insight_value_gate/v1", insights=listing)
        gate_max_tokens = max(
            config_constants.GI_INSIGHT_TOKENS_FLOOR,
            len(insights) * config_constants.GI_VALUE_GATE_TOKENS_EACH,
        )

        def _call() -> str:
            response = self._chat_create(
                model=self.summary_model,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.0,
                **self._token_kwarg(gate_max_tokens),
                response_format={"type": "json_object"},
                _lf_stage="gi",
            )
            # 2026-08-24 cost audit: value-gate calls emitted no llm_cost event.
            # Inside _call so a guardrail re-roll's second call is also counted.
            self._emit_stage_cost(
                stage="gi", capability="gi", model=self.summary_model, response=response
            )
            content = (response.choices[0].message.content or "").strip()
            return _insight_salvage.strip_json_fence(content)

        def _validate(content: str) -> None:
            _guardrails.check_chat_response(
                content, service=self._TELEMETRY_PROVIDER, expect_json=True
            )
            _json.loads(content)  # the real parse — a truncated JSON must trigger the re-roll

        # ADR-148: one bounded in-place re-roll on a transient invalid value-gate JSON before
        # raising. The gate fails open on a persistent bad response (owns keep-everything).
        content = _guardrails.structured_call_with_reroll(
            _call, _validate, service=self._TELEMETRY_PROVIDER, max_reroll=1
        )
        tiers = _json.loads(content)
        # Preserve input order; a missing id keeps the insight (tier 3) rather than dropping it.
        return [int(tiers.get(f"i{idx}", 3)) for idx in range(len(insights))]

    def detect_hosts(
        self,
        feed_title: str | None,
        feed_description: str | None,
        feed_authors: list[str] | None = None,
    ) -> Set[str]:
        """Detect host names from feed-level metadata using OpenAI API.

        Args:
            feed_title: Feed title
            feed_description: Optional feed description
            feed_authors: Optional list of author names from RSS feed (preferred source)

        Returns:
            Set of detected host names
        """
        if not self._speaker_detection_initialized:
            raise RuntimeError(
                "OpenAIProvider speaker detection not initialized. Call initialize() first."
            )

        # Prefer RSS author tags if available
        if feed_authors:
            return set(feed_authors)

        # Otherwise, use OpenAI API to detect hosts from feed metadata
        # This is a simplified version - could be enhanced with dedicated prompt
        if not feed_title:
            return set()

        try:
            # Use detect_speakers with empty known_hosts to detect hosts
            speakers, detected_hosts, _, _ = self.detect_speakers(
                episode_title=feed_title,
                episode_description=feed_description,
                known_hosts=set(),
            )
            return detected_hosts
        except Exception as exc:
            logger.warning(
                "Failed to detect hosts from feed metadata: %s", format_exception_for_log(exc)
            )
            return set()

    def detect_speakers(
        self,
        episode_title: str,
        episode_description: str | None,
        known_hosts: Set[str],
        pipeline_metrics: metrics.Metrics | None = None,
    ) -> Tuple[list[str], Set[str], bool, bool]:
        """Detect speaker names from episode metadata using OpenAI API.

        Args:
            episode_title: Episode title
            episode_description: Optional episode description
            known_hosts: Set of known host names (for context)

        Returns:
            Tuple of:
            - List of detected speaker names (hosts + guests)
            - Set of detected host names (subset of known_hosts)
            - Success flag (True if detection succeeded)
            - used_defaults: True if default names were returned (e.g. on failure)

        Raises:
            ValueError: If detection fails or API key is invalid
            RuntimeError: If provider is not initialized
        """
        # If auto_speakers is disabled, return defaults without requiring initialization
        if not self.cfg.auto_speakers:
            logger.debug("Auto-speakers disabled, detection failed")
            return DEFAULT_SPEAKER_NAMES.copy(), set(), False, True

        if not self._speaker_detection_initialized:
            raise RuntimeError(
                "OpenAIProvider speaker detection not initialized. Call initialize() first."
            )

        logger.debug("Detecting speakers via OpenAI API for episode: %s", episode_title[:50])

        try:
            # Build prompt using prompt_store
            user_prompt = self._build_speaker_detection_prompt(
                episode_title, episode_description, known_hosts
            )

            # Get system prompt from prompt_store
            from ...prompts.store import render_prompt

            system_prompt_name = (
                getattr(self.cfg, f"{self._CONFIG_NS}_speaker_system_prompt", None)
                or "openai/ner/system_ner_v1"
            )
            system_prompt = render_prompt(system_prompt_name)

            # Call OpenAI API with retry
            from ...utils.provider_metrics import (
                _safe_openai_retryable,
                retry_with_metrics,
            )

            response = retry_with_metrics(
                lambda: self._chat_create(
                    model=self.speaker_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.speaker_temperature,
                    **self._token_kwarg(300),
                    response_format={"type": "json_object"},
                    # #1323: this is a SMALL structured call — ~870 prompt tokens, 300 max
                    # completion, measured at 0.1 s against vLLM. Without this it inherits
                    # `summarization_timeout / 3` = 400 s, so one socket that stops answering
                    # stalls the feed for 400 s (up to ~20 min across the retries below). A
                    # thread dump of a wedged run shows exactly that: blocked in
                    # `_receive_response_headers` while the same server answered a fresh
                    # request in 116 ms. Bounding it turns the hang into a retry.
                    timeout=get_small_chat_call_timeout(self.cfg),
                    _lf_stage="speaker_detection",
                ),
                max_retries=2,
                initial_delay=1.0,
                max_delay=30.0,
                retryable_exceptions=_safe_openai_retryable(),
            )

            # 2026-08-24 cost audit: speaker detection emitted no llm_cost event —
            # invisible in cost telemetry (only the metrics.json aggregate below knew).
            self._emit_stage_cost(
                stage="speaker_detection",
                capability="speaker_detection",
                model=self.speaker_model,
                response=response,
            )

            response_text = response.choices[0].message.content
            if not response_text:
                logger.warning("OpenAI API returned empty response")
                return DEFAULT_SPEAKER_NAMES.copy(), set(), False, True

            # Parse JSON response
            speakers, detected_hosts, success = self._parse_speakers_from_response(
                response_text, known_hosts
            )

            logger.debug(
                "OpenAI speaker detection completed: %d speakers, %d hosts, success=%s",
                len(speakers),
                len(detected_hosts),
                success,
            )

            # Track LLM call metrics if available
            if pipeline_metrics is not None and hasattr(response, "usage"):
                input_tokens = response.usage.prompt_tokens if response.usage else 0
                output_tokens = response.usage.completion_tokens if response.usage else 0
                # Compute cost for the aggregate so metrics.json reflects speaker
                # detection spend (previously silently 0 in total_stage_cost_usd).
                sd_cost: Optional[float] = None
                if input_tokens > 0 or output_tokens > 0:
                    from ...workflow.helpers import calculate_provider_cost

                    sd_cost = calculate_provider_cost(
                        cfg=self.cfg,
                        provider_type=self._TELEMETRY_PROVIDER,
                        capability="speaker_detection",
                        model=self.speaker_model,
                        prompt_tokens=int(input_tokens),
                        completion_tokens=int(output_tokens),
                    )
                pipeline_metrics.record_llm_speaker_detection_call(
                    input_tokens, output_tokens, cost_usd=sd_cost
                )

            return speakers, detected_hosts, success, False

        except json.JSONDecodeError as exc:
            logger.error(
                "Failed to parse OpenAI API JSON response: %s", format_exception_for_log(exc)
            )
            logger.debug(
                "Response text: %s", response_text if "response_text" in locals() else "N/A"
            )
            return DEFAULT_SPEAKER_NAMES.copy(), set(), False, True
        except Exception as exc:
            logger.error("OpenAI API error in speaker detection: %s", format_exception_for_log(exc))
            from podcast_scraper.exceptions import ProviderAuthError, ProviderRuntimeError

            # Handle OpenAI-specific error types
            error_msg = str(exc).lower()
            exc_type_name = type(exc).__name__
            if (
                "api key" in error_msg
                or "authentication" in error_msg
                or "permission" in error_msg
                or "401" in error_msg
                or "unauthorized" in error_msg
                or exc_type_name == "AuthenticationError"
            ):
                raise ProviderAuthError(
                    message=f"OpenAI authentication failed: {format_exception_for_log(exc)}",
                    provider=f"{type(self).__name__}/SpeakerDetection",
                    suggestion=(
                        "Check your OPENAI_API_KEY environment variable or config setting. "
                        "Verify the key is valid and has not expired."
                    ),
                ) from exc
            elif "quota" in error_msg or "rate limit" in error_msg or "429" in error_msg:
                raise ProviderRuntimeError(
                    message=f"OpenAI rate limit exceeded: {format_exception_for_log(exc)}",
                    provider=f"{type(self).__name__}/SpeakerDetection",
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"OpenAI speaker detection failed: {format_exception_for_log(exc)}",
                    provider=f"{type(self).__name__}/SpeakerDetection",
                ) from exc

    def extract_quotes(
        self,
        transcript: str,
        insight_text: str,
        **kwargs: Any,
    ) -> List[Any]:
        """Extract candidate quote span that supports the insight (GIL QA via LLM)."""
        if not self._summarization_initialized:
            raise RuntimeError(
                "OpenAIProvider summarization not initialized. Call initialize() first."
            )
        if not (transcript and insight_text):
            return []
        from ...gi.grounding import QuoteCandidate, resolve_llm_quote_span
        from ..common.evidence_prompts import render_extract_quote_prompt

        system, user = render_extract_quote_prompt(
            "openai",
            self.clip_transcript_to_budget(
                transcript,
                response_tokens=config_constants.GI_QUOTE_RESPONSE_TOKENS,
                label="staged-quote transcript",
            ),
            insight_text,
            None,  # already fitted by COUNT above; no char budget to re-apply on top
        )
        try:
            from ...utils.provider_metrics import (
                _safe_openai_retryable,
                apply_gil_evidence_llm_call_metrics,
                merge_gil_evidence_call_metrics_on_failure,
                ProviderCallMetrics,
                retry_with_metrics,
            )

            call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name(self._TELEMETRY_PROVIDER)
            call_metrics.set_breaker_config_from_cfg(self.cfg)
            pm = kwargs.get("pipeline_metrics")

            def _make_api_call():
                return self._chat_create(
                    model=self.summary_model,
                    # RFC-115: transcript-first so the per-insight quote calls share one cached
                    # prefix (system was empty; the transcript is embedded verbatim in `user`).
                    messages=self._build_stage_messages(
                        transcript=transcript, system_prompt=system, user_prompt=user
                    ),
                    temperature=0.0,
                    **self._token_kwarg(config_constants.GI_QUOTE_RESPONSE_TOKENS),
                    _lf_stage="extract_quotes",
                )

            try:
                response = retry_with_metrics(
                    _make_api_call,
                    max_retries=3,
                    initial_delay=1.0,
                    max_delay=30.0,
                    retryable_exceptions=_safe_openai_retryable(),
                    metrics=call_metrics,
                )
            except Exception:
                merge_gil_evidence_call_metrics_on_failure(call_metrics, pm)
                raise
            in_tok, out_tok = _openai_chat_usage_tokens(response)
            apply_gil_evidence_llm_call_metrics(
                call_metrics,
                pm,
                in_tok,
                out_tok,
                # 2026-08-24 cost audit: without cfg/provider_type/model the helper's
                # B4 emission block silently skips — this call was invisible to cost
                # telemetry (the bundled variant below always passed them).
                cfg=self.cfg,
                provider_type=self._TELEMETRY_PROVIDER,
                model=self.summary_model,
                stage="extract_quotes",
            )
            content = (response.choices[0].message.content or "").strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            obj = json.loads(content)
            quotes_raw = obj.get("quotes") or []
            if isinstance(quotes_raw, str):
                quotes_raw = [quotes_raw]
            # Backward compat: fall back to single quote_text
            if not quotes_raw:
                qt = (obj.get("quote_text") or "").strip()
                if qt:
                    quotes_raw = [qt]
            results_q: list = []
            for qt_str in quotes_raw:
                qt_clean = str(qt_str).strip()
                if not qt_clean:
                    continue
                resolved = resolve_llm_quote_span(transcript, qt_clean)
                if resolved is None:
                    continue
                r_start, r_end, r_verbatim = resolved
                results_q.append(
                    QuoteCandidate(
                        char_start=r_start,
                        char_end=r_end,
                        text=r_verbatim,
                        qa_score=1.0,
                    )
                )
            # Deduplicate: LLMs sometimes return the same quote multiple times
            seen_texts: set = set()
            deduped: list = []
            for q in results_q:
                if q.text not in seen_texts:
                    seen_texts.add(q.text)
                    deduped.append(q)
            results_q = deduped
            return results_q
        except Exception as e:
            logger.debug("OpenAI extract_quotes failed: %s", e, exc_info=True)
            return []

    def extract_quotes_bundled(
        self,
        transcript: str,
        insight_texts: List[str],
        **kwargs: Any,
    ) -> Dict[int, List[Any]]:
        """Bundle ``extract_quotes`` across all insights into one OpenAI call (#698 Layer A).

        Mirrors :meth:`GeminiProvider.extract_quotes_bundled`. Same shared prompt, same
        parser, same QuoteCandidate output shape — only the SDK glue differs. See
        ``providers/common/bundled_prompts.py`` for the prompt contract.
        """
        if not self._summarization_initialized:
            raise RuntimeError(
                "OpenAIProvider summarization not initialized. Call initialize() first."
            )
        if not transcript:
            return {idx: [] for idx in range(len(insight_texts))}
        if not insight_texts:
            return {}

        from ...gi.grounding import QuoteCandidate, resolve_llm_quote_span
        from ...providers.common.bundle_extract_parser import (
            BundleExtractParseError,
            BundleOutputBudgetExceeded,
            parse_bundled_extract_response,
        )
        from ...providers.common.bundled_prompts import (
            extract_quotes_bundled_max_tokens,
            EXTRACT_QUOTES_BUNDLED_SYSTEM,
            extract_quotes_bundled_user,
        )
        from ...utils.provider_metrics import (
            _safe_openai_retryable,
            apply_gil_evidence_llm_call_metrics,
            merge_gil_evidence_call_metrics_on_failure,
            ProviderCallMetrics,
            retry_with_metrics,
        )

        system = EXTRACT_QUOTES_BUNDLED_SYSTEM
        # The reply budget has to be known BEFORE the clip, because it is what the clip has to
        # leave room for (#2050). It used to be computed thirty lines further down, which is how
        # a 50,000-char clip and a reply budget scaled by insight count ended up sized in total
        # ignorance of each other and of the window they both had to fit.
        max_out = extract_quotes_bundled_max_tokens(len(insight_texts))
        # RFC-115: relocate the CLIPPED transcript (the exact string the builder embeds) so the
        # quote stage shares one cached prefix with summary/GI/KG. Now that every stage derives
        # its clip from the same served window, episodes share the prefix far more often — the
        # old note here ("the clip can differ from those stages' window") described the bug.
        clipped = self.clip_transcript_to_budget(
            (transcript or "").strip(), response_tokens=max_out, label="bundled-quote transcript"
        )
        user = extract_quotes_bundled_user(clipped, insight_texts)
        messages = self._build_stage_messages(
            transcript=clipped, system_prompt=system, user_prompt=user
        )

        call_metrics = ProviderCallMetrics()
        call_metrics.set_provider_name(self._TELEMETRY_PROVIDER)
        call_metrics.set_breaker_config_from_cfg(self.cfg)
        pm = kwargs.get("pipeline_metrics")

        # Constrain decoding to JSON. This call asked for a nested JSON document in free-form
        # text while the summary / GI / value-gate calls in this same class all set
        # ``response_format`` — the bundled quote path was the one that did not.
        #
        # Measured 2026-09-10 over the Batch B deepen (81 episodes, 13 bundled-parse failures):
        # 8 had ``finish_reason=stop`` with output far under the token cap — 3x missing ','
        # delimiter, 3x "Expecting value" (all at column 5), 1x bad ':' delimiter, and one that
        # emitted a COMPLETE document at char 4320 and then wrote 4319 more ("Extra data").
        # NONE was an unterminated string, so none was truncation; the remaining 5 were
        # ``finish_reason=length`` and are addressed by the chunk-size bound instead.
        # Guarded exactly as the calls above: the o1 / o3 / gpt-5 families reject
        # ``response_format`` alongside a completion-token limit.
        _uses_completion_tokens = self.summary_model.startswith(("o1", "o3", "gpt-5"))

        def _make_api_call() -> Any:
            call_kwargs: Dict[str, Any] = {
                "model": self.summary_model,
                "messages": messages,
                "temperature": 0.0,
                **self._token_kwarg(max_out),
            }
            if not _uses_completion_tokens:
                call_kwargs["response_format"] = self._bundled_quote_response_format(
                    len(insight_texts), max_out
                )
            return self._chat_create(**call_kwargs, _lf_stage="extract_quotes")

        try:
            response = retry_with_metrics(
                _make_api_call,
                max_retries=3,
                initial_delay=1.0,
                max_delay=30.0,
                retryable_exceptions=_safe_openai_retryable(),
                metrics=call_metrics,
            )
        except Exception:
            merge_gil_evidence_call_metrics_on_failure(call_metrics, pm)
            raise
        in_tok, out_tok = _openai_chat_usage_tokens(response)
        apply_gil_evidence_llm_call_metrics(
            call_metrics,
            pm,
            in_tok,
            out_tok,
            cfg=self.cfg,
            provider_type=self._TELEMETRY_PROVIDER,
            model=self.summary_model,
            stage="extract_quotes",
        )

        content = (response.choices[0].message.content or "").strip()
        try:
            parsed = parse_bundled_extract_response(content, expected_count=len(insight_texts))
        except BundleExtractParseError as exc:
            # WARNING, not debug. This is the evidence stack: when it fails, insights lose their
            # grounding quotes, and under `gi_require_grounding: true` an ungrounded insight can
            # be dropped outright. Four failures in ~11 episodes in the 2026-08-16 acceptance run
            # produced nothing at INFO — the only trace was the failover wrapper reporting
            # "Primary error: invalid JSON", with no way to tell a budget cutoff from bad JSON.
            finish_reason = None
            try:
                finish_reason = getattr(response.choices[0], "finish_reason", None)
            except Exception:  # pragma: no cover - diagnostics must not mask the parse failure
                pass
            logger.warning(
                "extract_quotes_bundled parse FAILED: %s | finish_reason=%s "
                "output_tokens=%s/%s insights=%s. %s",
                exc,
                finish_reason,
                out_tok,
                max_out,
                len(insight_texts),
                (
                    "the reply repeats itself -> a decoding loop, not a budget too small"
                    if _is_decoding_loop(content)
                    else (
                        "finish_reason and the truncation diagnosis agree -> raise the output "
                        "budget for this call"
                        if exc.truncation_suspected
                        else "not budget-shaped -> suspect the prompt or the model, not max_tokens"
                    )
                ),
            )
            # #1893: keep the REPLY, not just the fact that it failed. The log line above cannot
            # distinguish a decoding loop from genuine over-generation, and those need opposite
            # fixes. See _capture_bundle_failure.
            _capture_bundle_failure(
                cfg=self.cfg,
                content=content,
                error=str(exc),
                finish_reason=finish_reason,
                out_tok=out_tok,
                max_out=max_out,
                insight_texts=list(insight_texts or []),
                model=self.summary_model,
                prompt_chars=sum(len(str(m.get("content") or "")) for m in messages),
                in_tok=in_tok,
            )
            # ``finish_reason == "length"`` is the one signal that PROVES the budget ran out
            # rather than the model writing bad JSON. Re-raise as the caller-recoverable type
            # so the fallback wrapper leaves it alone and the bisect in
            # ``_maybe_prefetch_bundled_candidates`` retries the halves on THIS model.
            #
            # Without this, prod (2026-08-30, prod_dgx_full) failed here at 2560/2560 with ten
            # insights and the wrapper quietly answered from llama3.1:8b instead — seven times
            # across three of eight episodes, with the bisect never running once.
            if finish_reason == "length":
                # #1970: `finish_reason == "length"` PROVES the reply was cut off rather than the
                # model writing bad JSON. Count it, so "this episode shipped insights with zero
                # quotes" can be correlated with a measured budget cutoff instead of guessed at.
                if pm is not None:
                    try:
                        pm.gi_quote_extraction_truncated_events += 1
                    except Exception:  # pragma: no cover - telemetry must not mask the failure
                        pass
                if not _is_decoding_loop(content):
                    raise BundleOutputBudgetExceeded(
                        f"extract_quotes_bundled output budget exhausted: "
                        f"{out_tok}/{max_out} tokens for {len(insight_texts)} insights",
                        content_length=exc.content_length,
                        error_position=exc.error_position,
                        truncation_suspected=True,
                    ) from exc
                # A decoding loop, not a budget too small: a smaller batch over the same
                # transcript loops the same way (8 -> 4 -> 2 all failed on 2026-10-02), so
                # bisecting only spends another full budget per half. Keep what closed before
                # the loop and stop; insights left without a quote take the per-insight path.
                parsed = _salvage_closed_quotes(content, len(insight_texts))
                logger.warning(
                    "extract_quotes_bundled: decoding loop (%s); kept %d closed quote(s) for "
                    "%d of %d insight(s), not retrying smaller batches",
                    _repetition_signal(content)[1][:60],
                    sum(len(v) for v in parsed.values()),
                    len(parsed),
                    len(insight_texts),
                )
            else:
                raise

        out: Dict[int, List[Any]] = {}
        for idx in range(len(insight_texts)):
            quote_strings = parsed.get(idx, [])
            seen: set = set()
            candidates: List[Any] = []
            for qt_str in quote_strings:
                qt_clean = str(qt_str).strip()
                if not qt_clean:
                    continue
                resolved = resolve_llm_quote_span(transcript, qt_clean)
                if resolved is None:
                    continue
                r_start, r_end, r_verbatim = resolved
                if r_verbatim in seen:
                    continue
                seen.add(r_verbatim)
                candidates.append(
                    QuoteCandidate(
                        char_start=r_start,
                        char_end=r_end,
                        text=r_verbatim,
                        qa_score=1.0,
                    )
                )
            out[idx] = candidates
        return out

    def generate_insights(
        self,
        text: str,
        episode_title: Optional[str] = None,
        max_insights: int = 5,
        params: Optional[Dict[str, Any]] = None,
        pipeline_metrics: Optional[Any] = None,
    ) -> List[str]:
        """Generate a list of short insight statements from transcript (GIL).

        Uses openai/insight_extraction/v2 prompt; parses response as one insight per line.
        Returns empty list on failure; the episode then honestly has no insights.
        """
        if not self._summarization_initialized:
            raise RuntimeError(
                "OpenAIProvider summarization not initialized. Call initialize() first."
            )

        from ...prompts.store import render_prompt

        max_insights = max(1, min(int(max_insights), config_constants.GI_MAX_INSIGHTS_CEILING))
        insight_max_tokens = max(
            config_constants.GI_INSIGHT_TOKENS_FLOOR,
            max_insights * config_constants.GI_INSIGHT_TOKENS_EACH,
        )
        # Truncate transcript for context (e.g. ~100k chars) to avoid token limits
        # Honour the configured temperature. This was hardcoded to 0.3 in every provider,
        # so evals could not pin it and the pipeline was not reproducible.
        insight_temperature = _insight_salvage.resolve_insight_temperature(self.cfg, "openai")
        text_slice = (text or "").strip()
        # #2050: derive the clip from the window this deployment actually serves, not from a
        # literal. 120,000 chars is ~34,000 tokens at the 3.5 chars/token prod ratio — ALREADY
        # over the DGX's 32,768-token window, so this site was handing the server a prompt it
        # had to reject. A wider model (Gemini ~1M) was equally wrongly clipped downward.
        _fitted = self.clip_transcript_to_budget(
            text_slice, response_tokens=insight_max_tokens, label="insight-extraction transcript"
        )
        if len(_fitted) < len(text_slice):
            text_slice = _fitted + "\n\n[Transcript truncated.]"

        try:
            # The prompt decides what an insight IS, so it is a tuned parameter like any
            # other. Hardcoding it made the extraction prompt the only part of this stage that
            # could not be A/B tested — and left the v3 speech-act prompt reachable by ollama alone.
            prompt_version = str(getattr(self.cfg, "gi_insight_prompt_version", "v2") or "v2")
            user_prompt = render_prompt(
                f"openai/insight_extraction/{prompt_version}",
                transcript=text_slice,
                title=episode_title or "",
                max_insights=max_insights,
            )
            system_prompt = render_prompt("openai/insight_extraction/system_v1")
            response = self._chat_create(
                model=self.insight_model,
                # RFC-115: transcript-first so the cleaned transcript caches across this episode's
                # stages (text_slice is rendered verbatim into user_prompt).
                messages=self._build_stage_messages(
                    transcript=text_slice, system_prompt=system_prompt, user_prompt=user_prompt
                ),
                temperature=insight_temperature,
                **self._token_kwarg(insight_max_tokens),
                _lf_stage="gi",
            )
            in_tok, out_tok = _openai_chat_usage_tokens(response)
            # Cost computed up-front so it can be emitted on both happy-path
            # and guardrail-violation branches. ADR-100 cost-attribution.
            gi_cost: Optional[float] = None
            if in_tok is not None and out_tok is not None:
                from ...workflow.helpers import calculate_provider_cost

                gi_cost = _openai_response_cost_usd(response) or calculate_provider_cost(
                    cfg=self.cfg,
                    provider_type=self._TELEMETRY_PROVIDER,
                    capability="gi",
                    model=self.insight_model,
                    prompt_tokens=int(in_tok),
                    completion_tokens=int(out_tok),
                )
                if pipeline_metrics is not None and hasattr(pipeline_metrics, "record_llm_gi_call"):
                    pipeline_metrics.record_llm_gi_call(in_tok, out_tok, cost_usd=gi_cost)

            # ONE call must produce ONE cost event. The truncation branch below emits with
            # ``triggered_guardrail=True`` and then falls through to the normal emit, so a
            # truncated GI call was billed TWICE: measured 2026-08-31 on prod, two truncated
            # calls turned 35,601 real output tokens into 65,601 reported. Every spend figure
            # and per-stage share derived from ``llm_cost`` was inflated by exactly the
            # truncated calls — i.e. wrong in the direction that hides a problem.
            _gi_cost_emitted = False

            def _emit_gi_cost(*, triggered_guardrail: bool = False) -> None:
                nonlocal _gi_cost_emitted
                if in_tok is None or out_tok is None or _gi_cost_emitted:
                    return
                _gi_cost_emitted = True
                try:
                    from ...workflow.cost_monitoring import emit_llm_cost_event

                    emit_llm_cost_event(
                        self.cfg,
                        provider=self._TELEMETRY_PROVIDER,
                        stage="gi",
                        model=self.insight_model,
                        estimated_cost_usd=float(gi_cost or 0.0),
                        prompt_tokens=int(in_tok),
                        completion_tokens=int(out_tok),
                        triggered_guardrail=triggered_guardrail,
                        served_model=getattr(response, "model", None),
                        response=response,
                    )
                except Exception:  # noqa: BLE001 - defensive; never block GI on telemetry
                    pass

            content = (response.choices[0].message.content or "").strip()
            finish_reason = response.choices[0].finish_reason
            # Response-shape guardrail (ADR-100): GI is fail-up. Cost emitted
            # in BOTH branches; raw GuardrailViolation re-raised so the outer
            # `except Exception: return []` does NOT swallow it.
            try:
                _guardrails.check_chat_response(
                    content, service=self._TELEMETRY_PROVIDER, finish_reason=finish_reason
                )
            except _guardrails.GuardrailViolation as gv:
                _emit_gi_cost(triggered_guardrail=True)
                # A truncated LINE LIST is recoverable: the cut lands in the final line and
                # every earlier one is intact. Re-raising here loses the whole episode to the
                # whole-batch loss — 40 good insights discarded because the 41st was clipped.
                salvaged = _insight_salvage.salvage_truncated_lines(gv, content, pipeline_metrics)
                if salvaged is None:
                    raise
                content = salvaged
            _emit_gi_cost()
            lines = [
                line.strip()
                for line in content.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            # Drop common list prefixes: "1. ", "2) ", "- ", "* "
            cleaned = []
            for line in lines:
                s = line.strip()
                if not s:
                    continue
                if len(s) >= 2 and s[0].isdigit() and s[1] in ".)":
                    s = s[2:].strip()
                if s.startswith("- ") or s.startswith("* "):
                    s = s[2:].strip()
                if s:
                    cleaned.append(s)
            _insight_salvage.record_overgeneration(pipeline_metrics, len(cleaned), max_insights)
            return _insight_salvage.take_within_ceiling(cleaned, max_insights)
        except _guardrails.GuardrailViolation:
            # ADR-100: GI is fail-up. Propagate so FallbackAware can route to
            # the configured fallback provider. Do NOT mask as empty insights.
            raise
        except Exception as e:
            logger.debug("OpenAI generate_insights failed: %s", e, exc_info=True)
            return []

    def score_entailment(
        self,
        premise: str,
        hypothesis: str,
        **kwargs: Any,
    ) -> float:
        """Score entailment of hypothesis given premise (GIL NLI via LLM). 0–1."""
        if not self._summarization_initialized:
            raise RuntimeError(
                "OpenAIProvider summarization not initialized. Call initialize() first."
            )
        if not (premise and hypothesis):
            return 0.0
        from ..common.evidence_prompts import render_entailment_prompt

        system, user = render_entailment_prompt("openai", premise, hypothesis)
        try:
            from ...utils.provider_metrics import (
                _safe_openai_retryable,
                apply_gil_evidence_llm_call_metrics,
                merge_gil_evidence_call_metrics_on_failure,
                ProviderCallMetrics,
                retry_with_metrics,
            )

            call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name(self._TELEMETRY_PROVIDER)
            call_metrics.set_breaker_config_from_cfg(self.cfg)
            pm = kwargs.get("pipeline_metrics")

            def _make_api_call():
                return self._chat_create(
                    model=self.summary_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    **self._token_kwarg(10),
                    _lf_stage="score_entailment",
                )

            try:
                response = retry_with_metrics(
                    _make_api_call,
                    max_retries=3,
                    initial_delay=1.0,
                    max_delay=30.0,
                    retryable_exceptions=_safe_openai_retryable(),
                    metrics=call_metrics,
                )
            except Exception:
                merge_gil_evidence_call_metrics_on_failure(call_metrics, pm)
                raise
            in_tok, out_tok = _openai_chat_usage_tokens(response)
            apply_gil_evidence_llm_call_metrics(
                call_metrics,
                pm,
                in_tok,
                out_tok,
                # Same 2026-08-24 fix as extract_quotes: args required for emission.
                cfg=self.cfg,
                provider_type=self._TELEMETRY_PROVIDER,
                model=self.summary_model,
                stage="score_entailment",
            )
            content = (response.choices[0].message.content or "0").strip()
            # Take first number
            for part in content.replace(",", " ").split():
                try:
                    v = float(part)
                    return max(0.0, min(1.0, v))
                except ValueError:
                    continue
            return 0.0
        except Exception as e:
            logger.debug("OpenAI score_entailment failed: %s", e, exc_info=True)
            return 0.0

    def score_entailment_bundled(
        self,
        pairs: List[Tuple[str, str]],
        chunk_size: int = 15,
        **kwargs: Any,
    ) -> Dict[int, float]:
        """Bundle ``score_entailment`` across many pairs (#698 Layer B).

        Mirrors :meth:`GeminiProvider.score_entailment_bundled`. Chunks at
        ``chunk_size`` (default 15) and issues one OpenAI call per chunk.
        """
        if not self._summarization_initialized:
            raise RuntimeError(
                "OpenAIProvider summarization not initialized. Call initialize() first."
            )
        if not pairs:
            return {}
        chunk_size = max(1, int(chunk_size))
        out: Dict[int, float] = {}
        pm = kwargs.get("pipeline_metrics")
        for chunk_start in range(0, len(pairs), chunk_size):
            chunk = pairs[chunk_start : chunk_start + chunk_size]
            chunk_scores = self._score_entailment_bundled_chunk(
                chunk_pairs=chunk, pipeline_metrics=pm
            )
            for local_idx, score in chunk_scores.items():
                out[chunk_start + local_idx] = score
            if pm is not None and hasattr(pm, "gi_evidence_score_entailment_bundled_pairs_total"):
                pm.gi_evidence_score_entailment_bundled_pairs_total += len(chunk)
        return out

    def summarize_bundled(
        self,
        text: str,
        episode_title: Optional[str] = None,
        episode_description: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        pipeline_metrics: metrics.Metrics | None = None,
        call_metrics: Any | None = None,
    ) -> Dict[str, Any]:
        """One completion: semantic transcript clean + JSON title/bullets (Issue #477).

        Returns the same ``summary`` shape as :meth:`summarize` (JSON string
        with ``title``, ``summary``, and ``bullets``).
        """
        if not self._summarization_initialized:
            raise RuntimeError(
                "OpenAIProvider summarization not initialized. Call initialize() first."
            )

        from ...prompts.store import get_prompt_metadata, render_prompt
        from ...utils.provider_metrics import (
            _safe_openai_retryable,
            ProviderCallMetrics,
            retry_with_metrics,
        )

        max_out = int(getattr(self.cfg, "llm_bundled_max_output_tokens", 16384) or 16384)
        # Route through the single token-kwarg helper so provider caps (e.g. DeepSeek's 8192
        # hard limit) apply on the bundled paths too — not just the o1/o3/gpt-5 rename (#1504).
        _token_kwarg: Dict[str, Any] = self._token_kwarg(max_out)

        tmpl_kwargs = dict(self.cfg.summary_prompt_params or {})
        system_prompt = render_prompt(
            "openai/summarization/bundled_clean_summary_system_v1",
            **tmpl_kwargs,
        )
        user_prompt = render_prompt(
            "openai/summarization/bundled_clean_summary_user_v1",
            transcript=text,
            title=episode_title or "",
            **tmpl_kwargs,
        )
        from ..ml.diarization.transcript_naming import with_bundled_field

        user_prompt = with_bundled_field(user_prompt, params)

        if call_metrics is None:
            call_metrics = ProviderCallMetrics()
        call_metrics.set_provider_name(self._TELEMETRY_PROVIDER)
        call_metrics.set_breaker_config_from_cfg(self.cfg)

        def _make_api_call() -> Any:
            kwargs: Dict[str, Any] = {
                "model": self.summary_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.summary_temperature,
                **_token_kwarg,
            }
            if not self.summary_model.startswith(("o1", "o3", "gpt-5")):
                kwargs["response_format"] = {"type": "json_object"}
            if self.summary_seed is not None:
                kwargs["seed"] = self.summary_seed
            return self._chat_create(**kwargs, _lf_stage="summarization")

        try:
            response = retry_with_metrics(
                _make_api_call,
                max_retries=3,
                initial_delay=1.0,
                max_delay=30.0,
                retryable_exceptions=_safe_openai_retryable(),
                metrics=call_metrics,
            )
        except Exception:
            call_metrics.finalize()
            raise

        call_metrics.finalize()
        warn_if_truncated(
            response.choices[0].finish_reason,
            "openai",
            "summarize_bundled",
        )
        raw = (response.choices[0].message.content or "").strip()
        finish_reason = response.choices[0].finish_reason

        # Extract tokens up-front so the cost event can fire on both the
        # happy path and the GuardrailViolation path (ADR-100).
        input_tokens = None
        output_tokens = None
        if hasattr(response, "usage") and response.usage:
            pt = getattr(response.usage, "prompt_tokens", None)
            ct = getattr(response.usage, "completion_tokens", None)
            input_tokens = int(pt) if isinstance(pt, (int, float)) else 0
            output_tokens = int(ct) if isinstance(ct, (int, float)) else 0
            if input_tokens > 0 or output_tokens > 0:
                call_metrics.set_tokens(input_tokens, output_tokens)

        cost: Optional[float] = None
        if input_tokens is not None:
            from ...workflow.helpers import calculate_provider_cost

            cost = _openai_response_cost_usd(response) or calculate_provider_cost(
                cfg=self.cfg,
                provider_type=self._TELEMETRY_PROVIDER,
                capability="summarization",
                model=self.summary_model,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
            )

        def _record_cost(*, triggered_guardrail: bool = False) -> None:
            if input_tokens is None:
                return
            from ...utils.provider_metrics import record_provider_call_cost

            record_provider_call_cost(
                call_metrics,
                cost,
                cfg=self.cfg,
                provider_type=self._TELEMETRY_PROVIDER,
                capability="summarization",
                model=self.summary_model,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                triggered_guardrail=triggered_guardrail,
                response=response,
            )

        # Response-shape guardrail (ADR-100, #1003) — empty content, thinking-
        # prose markers, finish_reason=length. Cost emitted in BOTH branches
        # for paid-but-rejected attribution.
        try:
            _guardrails.check_chat_response(
                raw, service=self._TELEMETRY_PROVIDER, finish_reason=finish_reason
            )
        except _guardrails.GuardrailViolation:
            _record_cost(triggered_guardrail=True)
            raise

        # The call is paid whether or not its JSON is usable, so it is counted before the shape
        # checks below can raise (2026-10-23 cost audit: a bad answer was never billed).
        _record_cost()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Bundled response is not valid JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError("Bundled JSON must be an object")
        summary_prose = data.get("summary")
        bullets = data.get("bullets")
        if not isinstance(summary_prose, str) or not summary_prose.strip():
            raise ValueError("Bundled JSON missing non-empty summary string")
        if not isinstance(bullets, list) or not bullets:
            raise ValueError("Bundled JSON missing non-empty bullets list")

        if pipeline_metrics is not None and input_tokens is not None and output_tokens is not None:
            pipeline_metrics.record_llm_bundled_clean_summary_call(
                input_tokens, output_tokens, cost_usd=cost
            )

        prompt_metadata = {
            "system": get_prompt_metadata(
                "openai/summarization/bundled_clean_summary_system_v1",
                params=tmpl_kwargs,
            ),
            "user": get_prompt_metadata(
                "openai/summarization/bundled_clean_summary_user_v1",
                params={
                    **tmpl_kwargs,
                    "transcript": text[:100] + "..." if len(text) > 100 else text,
                },
            ),
        }

        return {
            "summary": raw,
            "summary_short": None,
            "metadata": {
                "model": self.summary_model,
                "provider": "openai",
                "bundled": True,
                "max_output_tokens": max_out,
                "prompts": prompt_metadata,
            },
        }

    def summarize_extraction_bundled(
        self,
        text: str,
        *,
        episode_title: Optional[str] = None,
        episode_description: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        pipeline_metrics: "metrics.Metrics | None" = None,
        call_metrics: Any | None = None,
    ) -> Any:
        """Single-call extraction bundle: insights + topics + entities (#643).

        Companion to :meth:`summarize_bundled` for the 2-call pipeline:
        bundled summary/bullets + bundled extraction. Omits summary/title/bullets
        from the prompt so the model can spend its budget on KG quality.

        Returns:
            :class:`MegaBundleResult` with empty title/summary/bullets and
            populated insights/topics/entities.
        """
        if not self._summarization_initialized:
            raise RuntimeError(
                "OpenAIProvider summarization not initialized. Call initialize() first."
            )

        from ...prompting.megabundle import build_extraction_bundle_prompt
        from ...utils.provider_metrics import (
            _safe_openai_retryable,
            ProviderCallMetrics,
            retry_with_metrics,
        )
        from ..common.megabundle_parser import parse_extraction_bundle_response

        max_out = int(
            (params or {}).get("max_tokens")
            or getattr(self.cfg, "llm_bundled_max_output_tokens", 16384)
            or 16384
        )
        language = getattr(self.cfg, "language", "en") or None
        system_prompt, user_prompt = build_extraction_bundle_prompt(
            text,
            language=language,
            # transcript naming's speakers field (transcript_naming.PARAMS_LABELS)
            speaker_labels=(params or {}).get("speaker_naming_labels"),
            cache_transcript_prefix=self._cache_transcript_prefix,
            max_transcript_chars=config_constants.transcript_budget_chars(
                getattr(self, "max_context_tokens", None), response_tokens=int(max_out)
            ),
        )

        _uses_completion_tokens = self.summary_model.startswith(("o1", "o3", "gpt-5"))
        # Route through the single token-kwarg helper so provider caps (e.g. DeepSeek's 8192
        # hard limit) apply on the bundled paths too — not just the o1/o3/gpt-5 rename (#1504).
        _token_kwarg: Dict[str, Any] = self._token_kwarg(max_out)

        if call_metrics is None:
            call_metrics = ProviderCallMetrics()
        call_metrics.set_provider_name(self._TELEMETRY_PROVIDER)
        call_metrics.set_breaker_config_from_cfg(self.cfg)

        def _make_api_call() -> Any:
            kwargs: Dict[str, Any] = {
                "model": self.summary_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.0,
                **_token_kwarg,
            }
            if not _uses_completion_tokens:
                kwargs["response_format"] = {"type": "json_object"}
            if self.summary_seed is not None:
                kwargs["seed"] = self.summary_seed
            return self._chat_create(**kwargs, _lf_stage="summarization")

        try:
            resp = retry_with_metrics(
                _make_api_call,
                max_retries=3,
                initial_delay=1.0,
                max_delay=30.0,
                retryable_exceptions=_safe_openai_retryable(),
                metrics=call_metrics,
            )
        except Exception:
            call_metrics.finalize()
            raise
        call_metrics.finalize()

        raw_text = (resp.choices[0].message.content or "").strip() or "{}"
        if hasattr(resp, "usage") and resp.usage is not None:
            try:
                call_metrics.set_tokens(
                    int(getattr(resp.usage, "prompt_tokens", 0) or 0),
                    int(getattr(resp.usage, "completion_tokens", 0) or 0),
                )
            except (TypeError, ValueError):
                pass

        _record_openai_summarization_call(
            resp,
            pipeline_metrics,
            cfg=self.cfg,
            model=self.summary_model,
            provider_type=self._TELEMETRY_PROVIDER,
        )
        from ...utils.provider_metrics import record_response_cost

        record_response_cost(
            self.cfg,
            resp,
            provider_type=self._TELEMETRY_PROVIDER,
            capability="summarization",
            stage="extraction_bundle",
            model=self.summary_model,
            call_metrics=call_metrics,
        )
        return parse_extraction_bundle_response(raw_text)

    def summarize_mega_bundled(
        self,
        text: str,
        *,
        episode_title: Optional[str] = None,
        episode_description: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        pipeline_metrics: "metrics.Metrics | None" = None,
        call_metrics: Any | None = None,
    ) -> Any:
        """Single-call mega-bundle: summary + bullets + insights + topics + entities (#643).

        #632 research flagged OpenAI as "not tier-1" because KG/entity quality
        regressed in mega-bundle mode on fixture transcripts. #646 real-episode
        validation revisits this claim.
        """
        if not self._summarization_initialized:
            raise RuntimeError(
                "OpenAIProvider summarization not initialized. Call initialize() first."
            )

        from ...prompting.megabundle import build_megabundle_prompt
        from ...utils.provider_metrics import (
            _safe_openai_retryable,
            ProviderCallMetrics,
            retry_with_metrics,
        )
        from ..common.megabundle_parser import parse_megabundle_response

        max_out = int(
            (params or {}).get("max_tokens")
            or getattr(self.cfg, "llm_bundled_max_output_tokens", 16384)
            or 16384
        )
        language = getattr(self.cfg, "language", "en") or None
        system_prompt, user_prompt = build_megabundle_prompt(
            text,
            language=language,
            # transcript naming's speakers field (transcript_naming.PARAMS_LABELS)
            speaker_labels=(params or {}).get("speaker_naming_labels"),
            cache_transcript_prefix=self._cache_transcript_prefix,
            max_transcript_chars=config_constants.transcript_budget_chars(
                getattr(self, "max_context_tokens", None), response_tokens=int(max_out)
            ),
        )

        _uses_completion_tokens = self.summary_model.startswith(("o1", "o3", "gpt-5"))
        # Route through the single token-kwarg helper so provider caps (e.g. DeepSeek's 8192
        # hard limit) apply on the bundled paths too — not just the o1/o3/gpt-5 rename (#1504).
        _token_kwarg: Dict[str, Any] = self._token_kwarg(max_out)

        if call_metrics is None:
            call_metrics = ProviderCallMetrics()
        call_metrics.set_provider_name(self._TELEMETRY_PROVIDER)
        call_metrics.set_breaker_config_from_cfg(self.cfg)

        def _make_api_call() -> Any:
            kwargs: Dict[str, Any] = {
                "model": self.summary_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.0,
                **_token_kwarg,
            }
            if not _uses_completion_tokens:
                kwargs["response_format"] = {"type": "json_object"}
            if self.summary_seed is not None:
                kwargs["seed"] = self.summary_seed
            return self._chat_create(**kwargs, _lf_stage="summarization")

        try:
            resp = retry_with_metrics(
                _make_api_call,
                max_retries=3,
                initial_delay=1.0,
                max_delay=30.0,
                retryable_exceptions=_safe_openai_retryable(),
                metrics=call_metrics,
            )
        except Exception:
            call_metrics.finalize()
            raise
        call_metrics.finalize()

        raw_text = (resp.choices[0].message.content or "").strip() or "{}"
        if hasattr(resp, "usage") and resp.usage is not None:
            try:
                call_metrics.set_tokens(
                    int(getattr(resp.usage, "prompt_tokens", 0) or 0),
                    int(getattr(resp.usage, "completion_tokens", 0) or 0),
                )
            except (TypeError, ValueError):
                pass

        _record_openai_summarization_call(
            resp,
            pipeline_metrics,
            cfg=self.cfg,
            model=self.summary_model,
            provider_type=self._TELEMETRY_PROVIDER,
        )
        from ...utils.provider_metrics import record_response_cost

        record_response_cost(
            self.cfg,
            resp,
            provider_type=self._TELEMETRY_PROVIDER,
            capability="summarization",
            stage="summarization",
            model=self.summary_model,
            call_metrics=call_metrics,
        )
        return parse_megabundle_response(raw_text)
