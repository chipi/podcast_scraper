"""OllamaProvider's GI and naming methods (E04 split line, S1).

Moved verbatim out of ``OllamaProvider`` (``ollama_provider``), which inherits this
class: insight extraction, grounding and its quotes, entailment scoring, insight
classification, speaker and host detection, and the bundled summary calls that carry
naming fields. They leave public with GI and naming in the core release; the provider
keeps its call surface.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ...models import Episode
else:
    from ... import models

    Episode = models.Episode  # type: ignore[assignment]
from ... import config_constants
from ...utils.log_redaction import format_exception_for_log
from ...utils.provider_metadata import warn_if_truncated
from ...workflow import metrics
from .. import guardrails as _guardrails, insight_salvage as _insight_salvage
from ..common.transcript_cache import openai_style_messages as _openai_style_messages
from ..ml.speaker_detection import DEFAULT_SPEAKER_NAMES
from ._ollama_provider_calls import _flatten_json_speaker_names, _ollama_openai_chat_extra_kwargs

# The provider module's logger (one logger object): records keep their logger name.
logger = logging.getLogger("podcast_scraper.providers.ollama.ollama_provider")


# The old inline wording, kept only as the fallback if the template is missing. Do not tune here —
# tune the template, which is what the calibration harness measures.
_ENTAILMENT_FALLBACK = (
    "You rate how much the premise supports the hypothesis. "
    "Reply with ONLY a number between 0 and 1 (0=not at all, 1=fully supports)."
)


def _json_object_from_response(text: str) -> str:
    """Pull the JSON object out of a chat reply.

    qwen3.5 reasons before answering, and the reasoning is full of digits and braces. Scoping past
    any ``</think>`` block first is what stops us parsing the model's scratch work as the answer.
    """
    body = text or ""
    marker = body.rfind("</think>")
    if marker != -1:
        body = body[marker + len("</think>") :]
    start = body.find("{")
    end = body.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object in response")
    return body[start : end + 1]


# Roughly 3.5 characters per token for English prose. Used only to keep the transcript inside the
# context window we were actually configured with, instead of a hardcoded guess.
_CHARS_PER_TOKEN = 3.5


# Headroom for the instructions, the insight, and the model's own output.
_EXTRACT_QUOTE_RESERVE_TOKENS = 2000


def _transcript_budget_chars(num_ctx: int) -> int:
    """How much transcript actually fits in the configured context window.

    The old code truncated to a hardcoded 50 000 characters. Real episodes here are 67 000-78 000,
    so **the last third of every episode was invisible to quote extraction** — an insight drawn
    from it could never be grounded, no matter how good the model or the gate. With num_ctx=32768
    the true budget is ~107 000 characters, and nothing needs truncating at all.
    """
    usable = max(0, int(num_ctx) - _EXTRACT_QUOTE_RESERVE_TOKENS)
    return int(usable * _CHARS_PER_TOKEN)


def _render_extract_quote_prompt(transcript: str, insight: str, num_ctx: int) -> "tuple[str, str]":
    """``(system, user)`` for GIL quote extraction, from the ollama prompt template.

    The old inline prompt had three defects, each silently costing evidence: it truncated the
    transcript to 50 000 chars (below our episode length); its system and user messages disagreed
    about the output shape (``{"quotes": [...]}`` vs "quote_text only"); and it embedded a copyable
    example string, which is exactly what local models reproduce verbatim (#1179).
    """
    from ...prompts.store import render_prompt

    budget = _transcript_budget_chars(num_ctx)
    text = transcript.strip()
    if len(text) > budget:
        logger.warning(
            "transcript %d chars exceeds the %d-char context budget; quote extraction will not "
            "see the tail of this episode",
            len(text),
            budget,
        )
        text = text[:budget]

    try:
        return "", render_prompt(
            "ollama/evidence/extract_quote/v1",
            transcript=text,
            insight=insight.strip(),
        )
    except Exception as exc:  # noqa: BLE001 — a missing template must not break grounding
        logger.warning("ollama extract_quote template unavailable (%s); using inline fallback", exc)
        return (
            "Extract short verbatim quotes from the transcript that support the insight. "
            'Reply with ONLY a JSON object with a single key "quotes" (a list of exact strings).',
            f"Transcript:\n{text}\n\nInsight: {insight.strip()}",
        )


def _render_entailment_prompt(premise: str, hypothesis: str) -> "tuple[str, str]":
    """``(system, user)`` for the GIL entailment gate, from the ollama prompt template.

    The wording IS the gate. Strict textual entailment ("does the premise support the hypothesis")
    is not the question the pipeline means — a quote can be excellent evidence for an insight
    without logically entailing it — and asking it strictly cost 60% of the evidence a trusted
    annotator had accepted (#1179). Keeping this in a template is what lets it be calibrated.
    """
    from ...prompts.store import render_prompt

    try:
        rendered = render_prompt(
            "ollama/evidence/entailment/v1",
            premise=premise.strip(),
            hypothesis=hypothesis.strip(),
        )
        return "", rendered
    except Exception as exc:  # noqa: BLE001 — a missing template must not break grounding
        logger.warning("ollama entailment template unavailable (%s); using inline fallback", exc)
        return (
            _ENTAILMENT_FALLBACK,
            f"Premise: {premise.strip()}\n\nHypothesis: {hypothesis.strip()}",
        )


class OllamaProviderGiNamingMethods:
    """OllamaProvider's GI and naming methods; ``self`` is the provider."""

    def _build_speaker_detection_prompt(
        self, episode_title: str, episode_description: str | None, known_hosts: Set[str]
    ) -> str:
        """Build user prompt for speaker detection using prompt_store."""
        from ...prompts.store import render_prompt

        # Try model-specific prompt first, fallback to generic
        if self.cfg.ollama_speaker_user_prompt:
            user_prompt_name = self.cfg.ollama_speaker_user_prompt
        else:
            user_prompt_name = self._get_model_specific_prompt_path(
                self.speaker_model,
                "ner",
                "guest_host_v1",
                "ollama/ner/guest_host_v1",
            )
        template_params = {
            "episode_title": episode_title,
            "episode_description": episode_description or "",
            "known_hosts": ", ".join(sorted(known_hosts)) if known_hosts else "",
        }
        template_params.update(self.cfg.ner_prompt_params)
        user_prompt = render_prompt(user_prompt_name, **template_params)
        return user_prompt

    def _parse_speakers_from_response(
        self, response_text: str, known_hosts: Set[str]
    ) -> Tuple[list[str], Set[str], bool]:
        """Parse speaker names from Ollama API response."""
        # The model's raw answer, for the episode's naming decision trace (#2276).
        self.last_speaker_detection_raw = response_text
        try:
            data = json.loads(response_text)
            if isinstance(data, dict):
                speakers_raw = data.get("speakers", [])
                hosts_flat = _flatten_json_speaker_names(data.get("hosts", []))
                hosts = set(hosts_flat)
                guests_flat = _flatten_json_speaker_names(data.get("guests", []))
                if speakers_raw:
                    all_speakers = _flatten_json_speaker_names(speakers_raw)
                else:
                    all_speakers = list(hosts) + guests_flat
                return all_speakers, hosts, True
        except json.JSONDecodeError:
            if response_text.strip().startswith("{"):
                return DEFAULT_SPEAKER_NAMES.copy(), set(), False
            pass

        # Fallback: parse from plain text
        speakers = []
        for line in response_text.strip().split("\n"):
            for name in line.split(","):
                name = name.strip().strip("-").strip("*").strip()
                if name and len(name) > 1:
                    speakers.append(name)
        detected_hosts = set(s for s in speakers if s in known_hosts)
        return speakers, detected_hosts, len(speakers) > 0

    def _score_entailment_bundled_chunk(
        self,
        chunk_pairs: List[Tuple[str, str]],
        pipeline_metrics: Optional[Any],
    ) -> Dict[int, float]:
        """One bundled NLI Ollama call."""
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
            openai_compatible_chat_usage_tokens,
            ProviderCallMetrics,
            retry_with_metrics,
        )

        system = SCORE_ENTAILMENT_BUNDLED_SYSTEM
        user = score_entailment_bundled_user(chunk_pairs)
        call_metrics = ProviderCallMetrics()
        call_metrics.set_provider_name("ollama")
        max_out = score_entailment_bundled_max_tokens(len(chunk_pairs))

        def _make_api_call() -> Any:
            return self.client.chat.completions.create(
                model=self.summary_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=max_out,
                **_ollama_openai_chat_extra_kwargs(
                    self.summary_model, num_ctx=self.summary_num_ctx
                ),
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
        in_tok, out_tok = openai_compatible_chat_usage_tokens(response)
        apply_gil_evidence_llm_call_metrics(
            call_metrics,
            pipeline_metrics,
            in_tok,
            out_tok,
            cfg=self.cfg,
            provider_type="ollama",
            model=self.summary_model,
            stage="score_entailment",
        )

        content = (response.choices[0].message.content or "").strip()
        try:
            return parse_bundled_nli_response(content, expected_count=len(chunk_pairs))
        except BundleNliParseError as exc:
            logger.debug("Ollama score_entailment_bundled parse failed: %s", exc)
            raise

    def classify_insights(self, insights: List[str]) -> List[int]:
        """Tier each insight 0-3 for the value gate (see ``gi.value_gate``).

        One call for the whole episode. Raises on failure — the gate fails open and owns the
        decision to keep everything, so this must not paper over a bad response.
        """
        if not insights:
            return []
        if not self._summarization_initialized:
            raise RuntimeError("Ollama summarization not initialized for classify_insights")

        import json as _json

        from ...prompts.store import render_prompt

        listing = "\n".join(f"i{idx}: {text}" for idx, text in enumerate(insights))
        user_prompt = render_prompt("ollama/insight_value_gate/v1", insights=listing)
        response = self.client.chat.completions.create(
            model=self.summary_model,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.0,
            max_tokens=max(
                config_constants.GI_INSIGHT_TOKENS_FLOOR,
                len(insights) * config_constants.GI_VALUE_GATE_TOKENS_EACH,
            ),
            **_ollama_openai_chat_extra_kwargs(self.summary_model),
        )
        from ...utils.provider_metrics import record_response_cost

        record_response_cost(
            self.cfg,
            response,
            provider_type="ollama",
            capability="gi",
            stage="gi",
            model=self.summary_model,
        )
        content = (response.choices[0].message.content or "").strip()
        _guardrails.check_chat_response(content, service="ollama")
        tiers = _json.loads(_json_object_from_response(content))
        # Preserve input order; a missing id keeps the insight (tier 3) rather than dropping it.
        return [int(tiers.get(f"i{idx}", 3)) for idx in range(len(insights))]

    def detect_hosts(
        self,
        feed_title: str | None,
        feed_description: str | None,
        feed_authors: list[str] | None = None,
    ) -> Set[str]:
        """Detect host names from feed-level metadata using Ollama API.

        Args:
            feed_title: Feed title (can be None)
            feed_description: Optional feed description
            feed_authors: Optional list of author names from RSS feed (preferred source)

        Returns:
            Set of detected host names
        """
        if not self._speaker_detection_initialized:
            raise RuntimeError(
                "OllamaProvider speaker detection not initialized. Call initialize() first."
            )

        # Prefer RSS author tags if available (like OpenAI)
        if feed_authors:
            return set(feed_authors)

        # Otherwise, use Ollama API to detect hosts from feed metadata
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
        """Detect speaker names from episode metadata using Ollama API.

        Args:
            episode_title: Episode title
            episode_description: Optional episode description
            known_hosts: Set of known host names (for context)
            pipeline_metrics: Optional metrics tracker

        Returns:
            Tuple of:
            - List of detected speaker names (hosts + guests)
            - Set of detected host names (subset of known_hosts)
            - Success flag (True if detection succeeded)
            - used_defaults: True if default names were returned (e.g. on failure)

        Raises:
            ValueError: If detection fails
            RuntimeError: If provider is not initialized
        """
        # If auto_speakers is disabled, return defaults without requiring initialization
        if not self.cfg.auto_speakers:
            logger.debug("Auto-speakers disabled, detection failed")
            return DEFAULT_SPEAKER_NAMES.copy(), set(), False, True

        if not self._speaker_detection_initialized:
            raise RuntimeError(
                "OllamaProvider speaker detection not initialized. Call initialize() first."
            )

        logger.debug("Detecting speakers via Ollama API for episode: %s", episode_title[:50])

        try:
            # Build prompt using prompt_store
            user_prompt = self._build_speaker_detection_prompt(
                episode_title, episode_description, known_hosts
            )

            # Get system prompt from prompt_store
            from ...prompts.store import render_prompt

            # Try model-specific prompt first, fallback to generic
            if self.cfg.ollama_speaker_system_prompt:
                system_prompt_name = self.cfg.ollama_speaker_system_prompt
            else:
                system_prompt_name = self._get_model_specific_prompt_path(
                    self.speaker_model,
                    "ner",
                    "system_ner_v1",
                    "ollama/ner/system_ner_v1",
                )
            system_prompt = render_prompt(system_prompt_name)

            # Call Ollama API (OpenAI-compatible format) with retry
            from ...utils.provider_metrics import retry_with_metrics

            logger.info(
                "Calling Ollama API for speaker detection with model: '%s' "
                "(exact name being sent to Ollama)",
                self.speaker_model,
            )

            response = retry_with_metrics(
                lambda: self.client.chat.completions.create(
                    model=self.speaker_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.speaker_temperature,
                    max_tokens=300,
                    response_format={"type": "json_object"},
                    **_ollama_openai_chat_extra_kwargs(self.speaker_model),
                ),
                max_retries=2,
                initial_delay=1.0,
                max_delay=30.0,
                retryable_exceptions=(Exception,),
            )
            from ...utils.provider_metrics import record_response_cost

            record_response_cost(
                self.cfg,
                response,
                provider_type="ollama",
                capability="speaker_detection",
                stage="speaker_detection",
                model=self.speaker_model,
            )

            response_text = response.choices[0].message.content
            if not response_text:
                logger.warning("Ollama API returned empty response")
                return DEFAULT_SPEAKER_NAMES.copy(), set(), False, True

            # Parse JSON response
            speakers, detected_hosts, success = self._parse_speakers_from_response(
                response_text, known_hosts
            )

            logger.debug(
                "Ollama speaker detection completed: %d speakers, %d hosts, success=%s",
                len(speakers),
                len(detected_hosts),
                success,
            )

            # Track LLM call metrics if available (Ollama doesn't report usage, but track anyway)
            if pipeline_metrics is not None and hasattr(response, "usage"):
                input_tokens = response.usage.prompt_tokens if response.usage else 0
                output_tokens = response.usage.completion_tokens if response.usage else 0
                sd_cost: Optional[float] = None
                if input_tokens > 0 or output_tokens > 0:
                    from ...workflow.helpers import calculate_provider_cost

                    sd_cost = calculate_provider_cost(
                        cfg=self.cfg,
                        provider_type="ollama",
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
                "Failed to parse Ollama API JSON response: %s", format_exception_for_log(exc)
            )
            return DEFAULT_SPEAKER_NAMES.copy(), set(), False, True
        except Exception as exc:
            logger.error("Ollama API error in speaker detection: %s", format_exception_for_log(exc))
            from podcast_scraper.exceptions import ProviderRuntimeError

            raise ProviderRuntimeError(
                message=f"Ollama speaker detection failed: {format_exception_for_log(exc)}",
                provider="OllamaProvider/SpeakerDetection",
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
                "OllamaProvider summarization not initialized. Call initialize() first."
            )
        if not (transcript and insight_text):
            return []
        import json

        from ...gi.grounding import QuoteCandidate, resolve_llm_quote_span

        system, user = _render_extract_quote_prompt(transcript, insight_text, self.summary_num_ctx)
        try:
            from ...utils.provider_metrics import (
                _safe_openai_retryable,
                apply_gil_evidence_llm_call_metrics,
                merge_gil_evidence_call_metrics_on_failure,
                openai_compatible_chat_usage_tokens,
                ProviderCallMetrics,
                retry_with_metrics,
            )

            call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("ollama")
            pm = kwargs.get("pipeline_metrics")

            def _make_api_call():
                return self.client.chat.completions.create(
                    model=self.summary_model,
                    messages=_openai_style_messages(  # RFC-115 transcript-first
                        transcript, system, user, enabled=self._cache_transcript_prefix
                    ),
                    temperature=0.0,
                    max_tokens=config_constants.GI_QUOTE_RESPONSE_TOKENS,
                    **_ollama_openai_chat_extra_kwargs(self.summary_model),
                )

            try:
                response = retry_with_metrics(
                    _make_api_call,
                    max_retries=3,
                    initial_delay=1.0,
                    max_delay=30.0,
                    retryable_exceptions=_safe_openai_retryable(),
                    metrics=call_metrics,
                    error_context="ollama_local",
                )
            except Exception:
                merge_gil_evidence_call_metrics_on_failure(call_metrics, pm)
                raise
            in_tok, out_tok = openai_compatible_chat_usage_tokens(response)
            apply_gil_evidence_llm_call_metrics(
                call_metrics,
                pm,
                in_tok,
                out_tok,
                cfg=self.cfg,
                provider_type="ollama",
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
            logger.debug("Ollama extract_quotes failed: %s", e, exc_info=True)
            return []

    def extract_quotes_bundled(
        self,
        transcript: str,
        insight_texts: List[str],
        **kwargs: Any,
    ) -> Dict[int, List[Any]]:
        """Bundle ``extract_quotes`` (#698 Layer A) — Ollama (OpenAI-compat client)."""
        if not self._summarization_initialized:
            raise RuntimeError(
                "OllamaProvider summarization not initialized. Call initialize() first."
            )
        if not transcript:
            return {idx: [] for idx in range(len(insight_texts))}
        if not insight_texts:
            return {}

        from ...gi.grounding import QuoteCandidate, resolve_llm_quote_span
        from ...providers.common.bundle_extract_parser import (
            BundleExtractParseError,
            parse_bundled_extract_response,
        )
        from ...providers.common.bundled_prompts import (
            extract_quotes_bundled_max_tokens,
            EXTRACT_QUOTES_BUNDLED_SYSTEM,
            extract_quotes_bundled_user,
            transcript_clip,
        )
        from ...utils.provider_metrics import (
            _safe_openai_retryable,
            apply_gil_evidence_llm_call_metrics,
            merge_gil_evidence_call_metrics_on_failure,
            openai_compatible_chat_usage_tokens,
            ProviderCallMetrics,
            retry_with_metrics,
        )

        system = EXTRACT_QUOTES_BUNDLED_SYSTEM
        # #2050: the reply budget is computed first because it is what the clip must leave room
        # for. Both used to be sized in ignorance of each other and of the served window.
        max_out = extract_quotes_bundled_max_tokens(len(insight_texts))
        # RFC-115: relocate the CLIPPED transcript (exact embedded string) for prefix caching.
        clipped = transcript_clip(
            transcript,
            max_chars=config_constants.transcript_budget_chars(
                getattr(self, "max_context_tokens", None),
                response_tokens=max_out,
            ),
        )
        user = extract_quotes_bundled_user(clipped, insight_texts)
        messages = _openai_style_messages(
            clipped, system, user, enabled=self._cache_transcript_prefix
        )
        call_metrics = ProviderCallMetrics()
        call_metrics.set_provider_name("ollama")
        pm = kwargs.get("pipeline_metrics")

        def _make_api_call() -> Any:
            return self.client.chat.completions.create(
                model=self.summary_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.0,
                max_tokens=max_out,
                **_ollama_openai_chat_extra_kwargs(
                    self.summary_model, num_ctx=self.summary_num_ctx
                ),
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
        in_tok, out_tok = openai_compatible_chat_usage_tokens(response)
        apply_gil_evidence_llm_call_metrics(
            call_metrics,
            pm,
            in_tok,
            out_tok,
            cfg=self.cfg,
            provider_type="ollama",
            model=self.summary_model,
            stage="extract_quotes",
        )

        content = (response.choices[0].message.content or "").strip()
        try:
            parsed = parse_bundled_extract_response(content, expected_count=len(insight_texts))
        except BundleExtractParseError as exc:
            logger.debug("Ollama extract_quotes_bundled parse failed: %s", exc)
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

        Uses ollama/insight_extraction/v2 prompt; parses response as one insight per line.
        Returns empty list on failure; the episode then honestly has no insights.
        """
        if not self._summarization_initialized:
            raise RuntimeError(
                "OllamaProvider summarization not initialized. Call initialize() first."
            )

        from ...prompts.store import render_prompt

        max_insights = max(1, min(int(max_insights), config_constants.GI_MAX_INSIGHTS_CEILING))
        insight_max_tokens = max(
            config_constants.GI_INSIGHT_TOKENS_FLOOR,
            max_insights * config_constants.GI_INSIGHT_TOKENS_EACH,
        )
        # Honour the configured temperature. This was hardcoded to 0.3 in every provider,
        # so evals could not pin it and the pipeline was not reproducible.
        insight_temperature = _insight_salvage.resolve_insight_temperature(self.cfg, "ollama")
        text_slice = (text or "").strip()
        # #2050: derive the clip from the window this deployment actually serves, not from a
        # literal. 120,000 chars is ~34,000 tokens at the 3.5 chars/token prod ratio — ALREADY
        # over the DGX's 32,768-token window, so this site was handing the server a prompt it
        # had to reject. A wider model (Gemini ~1M) was equally wrongly clipped downward.
        _budget = config_constants.transcript_budget_chars(
            getattr(self, "max_context_tokens", None),
            response_tokens=insight_max_tokens,
        )
        if _budget and len(text_slice) > _budget:
            text_slice = text_slice[:_budget] + "\n\n[Transcript truncated.]"

        try:
            # The prompt decides what an insight IS, so it is a tuned parameter like any other.
            # Hardcoding it made the extraction prompt the only part of this stage that could not
            # be A/B tested.
            prompt_version = str(getattr(self.cfg, "gi_insight_prompt_version", "v2") or "v2")
            user_prompt = render_prompt(
                f"ollama/insight_extraction/{prompt_version}",
                transcript=text_slice,
                title=episode_title or "",
                max_insights=max_insights,
            )
            system_prompt = render_prompt("ollama/insight_extraction/system_v1")
            response = self.client.chat.completions.create(
                model=self.summary_model,
                messages=_openai_style_messages(  # type: ignore[arg-type]  # RFC-115
                    text_slice, system_prompt, user_prompt, enabled=self._cache_transcript_prefix
                ),
                temperature=insight_temperature,
                max_tokens=insight_max_tokens,
                **_ollama_openai_chat_extra_kwargs(self.summary_model),
            )
            content = (response.choices[0].message.content or "").strip()

            # ADR-100 cost-attribution: emit llm_cost in both branches.
            in_tok_gi = None
            out_tok_gi = None
            if hasattr(response, "usage") and response.usage:
                pt = getattr(response.usage, "prompt_tokens", None)
                ct = getattr(response.usage, "completion_tokens", None)
                in_tok_gi = int(pt) if isinstance(pt, (int, float)) else None
                out_tok_gi = int(ct) if isinstance(ct, (int, float)) else None
            gi_cost: Optional[float] = None
            if in_tok_gi is not None and out_tok_gi is not None:
                from ...workflow.helpers import calculate_provider_cost

                gi_cost = calculate_provider_cost(
                    cfg=self.cfg,
                    provider_type="ollama",
                    capability="gi",
                    model=self.summary_model,
                    prompt_tokens=in_tok_gi,
                    completion_tokens=out_tok_gi,
                )

            def _emit_gi_cost(*, triggered_guardrail: bool = False) -> None:
                if in_tok_gi is None or out_tok_gi is None:
                    return
                try:
                    from ...workflow.cost_monitoring import emit_llm_cost_event

                    emit_llm_cost_event(
                        self.cfg,
                        provider="ollama",
                        stage="gi",
                        model=self.summary_model,
                        estimated_cost_usd=float(gi_cost or 0.0),
                        prompt_tokens=in_tok_gi,
                        completion_tokens=out_tok_gi,
                        triggered_guardrail=triggered_guardrail,
                    )
                except Exception:  # noqa: BLE001
                    pass

            # ADR-100: GI is fail-up. Cost emitted in BOTH branches.
            try:
                _guardrails.check_chat_response(content, service="ollama")
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
            # ADR-100: GI is fail-up. Propagate so FallbackAware routes.
            raise
        except Exception as e:
            logger.debug("Ollama generate_insights failed: %s", e, exc_info=True)
            return []

    def score_entailment(
        self,
        premise: str,
        hypothesis: str,
        **kwargs: Any,
    ) -> float:
        """Score entailment of hypothesis given premise (GIL NLI via LLM). 0–1.

        The prompt comes from ``ollama/evidence/entailment`` rather than being hardcoded here.
        That is not tidying: the wording *is* the gate. Asked for strict textual entailment (the
        old inline wording), qwen3.5:35b accepted only 40% of the evidence a trusted annotator had
        accepted, and the corpus grounded 13.3% of its insights against the cloud's 91.3%. Asked
        the question the pipeline actually means — "is this quote EVIDENCE for this insight?" — it
        accepts 95%. The template is calibrated against gemini's own judgements
        (scripts/eval/score/entailment_calibration_v1.py); a hardcoded string cannot be.
        """
        if not self._summarization_initialized:
            raise RuntimeError(
                "OllamaProvider summarization not initialized. Call initialize() first."
            )
        if not (premise and hypothesis):
            return 0.0

        system, user = _render_entailment_prompt(premise, hypothesis)
        try:
            from ...utils.provider_metrics import (
                _safe_openai_retryable,
                apply_gil_evidence_llm_call_metrics,
                merge_gil_evidence_call_metrics_on_failure,
                openai_compatible_chat_usage_tokens,
                ProviderCallMetrics,
                retry_with_metrics,
            )

            call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("ollama")
            pm = kwargs.get("pipeline_metrics")

            def _make_api_call():
                return self.client.chat.completions.create(
                    model=self.summary_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    max_tokens=10,
                    **_ollama_openai_chat_extra_kwargs(self.summary_model),
                )

            try:
                response = retry_with_metrics(
                    _make_api_call,
                    max_retries=3,
                    initial_delay=1.0,
                    max_delay=30.0,
                    retryable_exceptions=_safe_openai_retryable(),
                    metrics=call_metrics,
                    error_context="ollama_local",
                )
            except Exception:
                merge_gil_evidence_call_metrics_on_failure(call_metrics, pm)
                raise
            in_tok, out_tok = openai_compatible_chat_usage_tokens(response)
            apply_gil_evidence_llm_call_metrics(
                call_metrics,
                pm,
                in_tok,
                out_tok,
                cfg=self.cfg,
                provider_type="ollama",
                model=self.summary_model,
                stage="score_entailment",
            )
            content = (response.choices[0].message.content or "0").strip()
            for part in content.replace(",", " ").split():
                try:
                    v = float(part)
                    return max(0.0, min(1.0, v))
                except ValueError:
                    continue
            return 0.0
        except Exception as e:
            logger.debug("Ollama score_entailment failed: %s", e, exc_info=True)
            return 0.0

    def score_entailment_bundled(
        self,
        pairs: List[Tuple[str, str]],
        chunk_size: int = 15,
        **kwargs: Any,
    ) -> Dict[int, float]:
        """Bundle ``score_entailment`` (#698 Layer B) — Ollama."""
        if not self._summarization_initialized:
            raise RuntimeError(
                "OllamaProvider summarization not initialized. Call initialize() first."
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
        """One completion: semantic transcript clean + JSON title/summary/bullets (Issue #477).

        Returns the same ``summary`` shape as :meth:`summarize` (JSON string
        with ``title``, ``summary``, and ``bullets``).
        """
        if not self._summarization_initialized:
            raise RuntimeError(
                "OllamaProvider summarization not initialized. " "Call initialize() first."
            )

        from ...prompts.store import get_prompt_metadata, render_prompt
        from ...utils.provider_metrics import (
            _safe_openai_retryable,
            ProviderCallMetrics,
            retry_with_metrics,
        )

        max_out = int(getattr(self.cfg, "llm_bundled_max_output_tokens", 16384) or 16384)

        # #2050 / advisor M2: the bundled path renders its own prompt and never reaches
        # `_build_summarization_prompts`, so it needs its own clip. Ollama truncates silently
        # rather than 400ing, and this is prod's summary failover tier.
        _budget = config_constants.transcript_budget_chars(
            getattr(self, "max_context_tokens", None), response_tokens=int(max_out)
        )
        if _budget is not None and len(text or "") > _budget:
            logger.warning(
                "ollama: bundled-summary transcript %d chars exceeds the %d-char budget; "
                "clipping (#2050)",
                len(text),
                _budget,
            )
            text = text[:_budget]
        tmpl_kwargs = dict(self.cfg.summary_prompt_params or {})
        system_prompt = render_prompt(
            "ollama/summarization/bundled_clean_summary_system_v1",
            **tmpl_kwargs,
        )
        user_prompt = render_prompt(
            "ollama/summarization/bundled_clean_summary_user_v1",
            transcript=text,
            title=episode_title or "",
            **tmpl_kwargs,
        )
        from ..ml.diarization.transcript_naming import with_bundled_field

        user_prompt = with_bundled_field(user_prompt, params)

        if call_metrics is None:
            call_metrics = ProviderCallMetrics()
        call_metrics.set_provider_name("ollama")

        def _make_api_call() -> Any:
            return self.client.chat.completions.create(
                model=self.summary_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.summary_temperature,
                max_tokens=max_out,
                response_format={"type": "json_object"},
                **_ollama_openai_chat_extra_kwargs(
                    self.summary_model, num_ctx=self.summary_num_ctx
                ),
            )

        try:
            response = retry_with_metrics(
                _make_api_call,
                max_retries=3,
                initial_delay=1.0,
                max_delay=30.0,
                retryable_exceptions=_safe_openai_retryable(),
                metrics=call_metrics,
                error_context="ollama_local",
            )
        except Exception:
            call_metrics.finalize()
            raise

        call_metrics.finalize()
        warn_if_truncated(
            response.choices[0].finish_reason,
            "ollama",
            "summarize_bundled",
        )
        raw = (response.choices[0].message.content or "").strip()

        # Token + cost capture up-front so cost emits in both branches (ADR-100).
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

            cost = calculate_provider_cost(
                cfg=self.cfg,
                provider_type="ollama",
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
                provider_type="ollama",
                capability="summarization",
                model=self.summary_model,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                triggered_guardrail=triggered_guardrail,
            )

        if pipeline_metrics is not None and input_tokens is not None and output_tokens is not None:
            pipeline_metrics.record_llm_bundled_clean_summary_call(
                input_tokens, output_tokens, cost_usd=cost
            )

        # Response-shape guardrail (ADR-099/100, #999/#1003).
        # Note: this fires BEFORE the JSON-parse block below; record the
        # guardrail violation as a Path D parse-failure kind so a sweep that
        # trips the guardrail still surfaces in metrics_report.md.
        try:
            _guardrails.check_chat_response(raw, service="ollama")
        except _guardrails.GuardrailViolation:
            if pipeline_metrics is not None and hasattr(
                pipeline_metrics, "record_llm_bundled_parse_failure"
            ):
                pipeline_metrics.record_llm_bundled_parse_failure("guardrail_violation")
            _record_cost(triggered_guardrail=True)
            raise

        # #912 Path D: bump the bundled-parse-failure counter on every parse
        # failure (per-kind) so future autoresearch sweeps can't silently
        # crown a flaky bundled candidate. Guards with hasattr/getattr so
        # production callers without an eval-side ``pipeline_metrics`` are
        # unaffected (Metrics only flows in from the autoresearch harness).
        def _record_parse_failure(kind: str) -> None:
            if pipeline_metrics is not None and hasattr(
                pipeline_metrics, "record_llm_bundled_parse_failure"
            ):
                pipeline_metrics.record_llm_bundled_parse_failure(kind)

        # The call is paid whether or not its JSON is usable, so it is counted before the shape
        # checks below can raise (2026-10-23 cost audit: a bad answer was never billed).
        _record_cost()

        try:
            data = json.loads(raw, strict=False)
        except json.JSONDecodeError as exc:
            _record_parse_failure("not_valid_json")
            raise ValueError(f"Bundled response is not valid JSON: {exc}") from exc

        if not isinstance(data, dict):
            _record_parse_failure("not_an_object")
            raise ValueError("Bundled JSON must be an object")
        summary_prose = data.get("summary")
        bullets = data.get("bullets")
        if not isinstance(summary_prose, str) or not summary_prose.strip():
            _record_parse_failure("missing_summary")
            raise ValueError("Bundled JSON missing non-empty summary string")
        if not isinstance(bullets, list) or not bullets:
            _record_parse_failure("missing_bullets")
            raise ValueError("Bundled JSON missing non-empty bullets list")

        prompt_metadata = {
            "system": get_prompt_metadata(
                "ollama/summarization/" "bundled_clean_summary_system_v1",
                params=tmpl_kwargs,
            ),
            "user": get_prompt_metadata(
                "ollama/summarization/" "bundled_clean_summary_user_v1",
                params={
                    **tmpl_kwargs,
                    "transcript": (text[:100] + "..." if len(text) > 100 else text),
                },
            ),
        }

        return {
            "summary": raw,
            "summary_short": None,
            "metadata": {
                "model": self.summary_model,
                "provider": "ollama",
                "bundled": True,
                "max_output_tokens": max_out,
                "prompts": prompt_metadata,
            },
        }
