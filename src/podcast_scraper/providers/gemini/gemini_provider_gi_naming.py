"""GeminiProvider's GI and naming methods (E04 split line, S1).

Moved verbatim out of ``GeminiProvider`` (``gemini_provider``), which inherits this
class: insight extraction, grounding and its quotes, entailment scoring, insight
classification, speaker and host detection, and the bundled summary calls that carry
naming fields. They leave public with GI and naming in the core release; the provider
keeps its call surface.
"""

from __future__ import annotations

import json
import logging
from typing import Any, cast, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from ... import config_constants

if TYPE_CHECKING:
    from ...models import Episode
else:
    from ... import models

    Episode = models.Episode  # type: ignore[assignment]
from ...utils.log_redaction import format_exception_for_log
from ...workflow import metrics
from .. import guardrails as _guardrails, insight_salvage as _insight_salvage
from ..ml.speaker_detection import DEFAULT_SPEAKER_NAMES
from ._gemini_provider_calls import (
    _gemini_finish_reason,
    _merge_generate_content_config,
    _record_gemini_llm_call,
)

# The provider module's logger (one logger object): records keep their logger name.
logger = logging.getLogger("podcast_scraper.providers.gemini.gemini_provider")


class GeminiProviderGiNamingMethods:
    """GeminiProvider's GI and naming methods; ``self`` is the provider."""

    def _build_speaker_detection_prompt(
        self, episode_title: str, episode_description: str | None, known_hosts: Set[str]
    ) -> str:
        """Build user prompt for speaker detection using prompt_store."""
        from ...prompts.store import render_prompt

        user_prompt_name = self.cfg.gemini_speaker_user_prompt
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
        """Parse speaker names from Gemini API response."""
        # The model's raw answer, for the episode's naming decision trace (#2276).
        self.last_speaker_detection_raw = response_text
        try:
            data = json.loads(response_text)
            if isinstance(data, dict):
                speakers = data.get("speakers", [])
                hosts = set(data.get("hosts", []))
                guests = data.get("guests", [])
                all_speakers = list(hosts) + guests if not speakers else speakers
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
        """One bundled NLI Gemini call for up to ``chunk_size`` pairs.

        Returns scores for the chunk indexed locally (``range(len(chunk_pairs))``).
        On any provider/parser failure, raises so the caller can decide on whole-
        batch fallback. Counted as one ``score_entailment`` substage call.
        """
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
            _safe_gemini_retryable,
            apply_gil_evidence_llm_call_metrics,
            gemini_generate_usage_tokens,
            merge_gil_evidence_call_metrics_on_failure,
            ProviderCallMetrics,
            retry_with_metrics,
        )

        system = SCORE_ENTAILMENT_BUNDLED_SYSTEM
        user = score_entailment_bundled_user(chunk_pairs)

        call_metrics = ProviderCallMetrics()
        call_metrics.set_provider_name("gemini")
        call_metrics.set_breaker_config_from_cfg(self.cfg)

        max_out = score_entailment_bundled_max_tokens(len(chunk_pairs))
        generation_config = _merge_generate_content_config(
            self.summary_model,
            {
                "temperature": 0.0,
                "max_output_tokens": max_out,
                "system_instruction": system,
            },
        )

        def _make_api_call() -> Any:
            return self.client.models.generate_content(
                model=self.summary_model,
                contents=user,
                config=cast(Any, generation_config),
            )

        try:
            response = retry_with_metrics(
                _make_api_call,
                **self._gemini_retry_params(
                    {"stage": "gemini_gil_score_entailment_bundled"},
                    pipeline_metrics=pipeline_metrics,
                ),
                retryable_exceptions=_safe_gemini_retryable(),
                metrics=call_metrics,
            )
        except Exception:
            merge_gil_evidence_call_metrics_on_failure(call_metrics, pipeline_metrics)
            raise
        in_tok, out_tok = gemini_generate_usage_tokens(response)
        apply_gil_evidence_llm_call_metrics(
            call_metrics,
            pipeline_metrics,
            in_tok,
            out_tok,
            cfg=self.cfg,
            provider_type="gemini",
            model=self.summary_model,
            stage="score_entailment",
        )

        content = response.text if hasattr(response, "text") else str(response)
        try:
            return parse_bundled_nli_response(content, expected_count=len(chunk_pairs))
        except BundleNliParseError as exc:
            logger.debug("Gemini score_entailment_bundled parse failed: %s", exc)
            raise

    def classify_insights(self, insights: List[str]) -> List[int]:
        """Tier each insight 0-3 for the value gate (see ``gi.value_gate``).

        One call for the whole episode. Raises on failure — the gate fails open, and it owns the
        decision to keep everything, so this must not silently return a wrong-length list.
        """
        if not insights:
            return []
        if not self._summarization_initialized:
            raise RuntimeError("Gemini summarization not initialized for classify_insights")

        import json as _json

        from ...prompts.store import render_prompt

        listing = "\n".join(f"i{idx}: {text}" for idx, text in enumerate(insights))
        prompt = render_prompt("gemini/insight_value_gate/v1", insights=listing)
        generation_config = _merge_generate_content_config(
            self.summary_model,
            {
                "temperature": 0.0,
                "max_output_tokens": max(
                    config_constants.GI_INSIGHT_TOKENS_FLOOR,
                    len(insights) * config_constants.GI_VALUE_GATE_TOKENS_EACH,
                ),
                "response_mime_type": "application/json",
            },
        )
        response = self.client.models.generate_content(
            model=self.summary_model,
            contents=prompt,
            config=cast(Any, generation_config),
        )
        from ...utils.provider_metrics import record_response_cost

        record_response_cost(
            self.cfg,
            response,
            provider_type="gemini",
            capability="gi",
            stage="gi",
            model=self.summary_model,
        )
        content = (getattr(response, "text", "") or "").strip()
        content = _insight_salvage.strip_json_fence(content)
        _guardrails.check_chat_response(
            content,
            service="gemini",
            finish_reason=_gemini_finish_reason(response),
            expect_json=True,
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
        """Detect host names from feed-level metadata using Gemini API.

        Args:
            feed_title: Feed title (can be None)
            feed_description: Optional feed description
            feed_authors: Optional list of author names from RSS feed (preferred source)

        Returns:
            Set of detected host names
        """
        if not self._speaker_detection_initialized:
            raise RuntimeError(
                "GeminiProvider speaker detection not initialized. Call initialize() first."
            )

        # Prefer RSS author tags if available (like OpenAI)
        if feed_authors:
            return set(feed_authors)

        # Otherwise, use Gemini API to detect hosts from feed metadata
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
        """Detect speaker names from episode metadata using Gemini API.

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
            ValueError: If detection fails or API key is invalid
            RuntimeError: If provider is not initialized
        """
        # If auto_speakers is disabled, return defaults without requiring initialization
        if not self.cfg.auto_speakers:
            logger.debug("Auto-speakers disabled, detection failed")
            return DEFAULT_SPEAKER_NAMES.copy(), set(), False, True

        if not self._speaker_detection_initialized:
            raise RuntimeError(
                "GeminiProvider speaker detection not initialized. Call initialize() first."
            )

        logger.debug("Detecting speakers via Gemini API for episode: %s", episode_title[:50])

        try:
            # Build prompt using prompt_store
            user_prompt = self._build_speaker_detection_prompt(
                episode_title, episode_description, known_hosts
            )

            # Get system prompt from prompt_store
            from ...prompts.store import render_prompt

            system_prompt_name = self.cfg.gemini_speaker_system_prompt or "gemini/ner/system_ner_v1"
            system_prompt = render_prompt(system_prompt_name)

            # Call Gemini API using new Client API
            # Request JSON response
            # Use dict format for generation config (more compatible with SDK versions)
            # Note: system_instruction is part of config in new API
            generation_config = _merge_generate_content_config(
                self.speaker_model,
                {
                    "temperature": self.speaker_temperature,
                    "max_output_tokens": 300,
                    "response_mime_type": "application/json",
                    "system_instruction": system_prompt,
                },
            )

            from ...utils.provider_metrics import (
                _safe_gemini_retryable,
                retry_with_metrics,
            )

            response = retry_with_metrics(
                lambda: self.client.models.generate_content(
                    model=self.speaker_model,
                    contents=user_prompt,
                    config=cast(Any, generation_config),
                ),
                **self._gemini_retry_params(
                    {"stage": "gemini_speaker"},
                    pipeline_metrics=pipeline_metrics,
                ),
                retryable_exceptions=_safe_gemini_retryable(),
            )
            from ...utils.provider_metrics import record_response_cost

            record_response_cost(
                self.cfg,
                response,
                provider_type="gemini",
                capability="speaker_detection",
                stage="speaker_detection",
                model=self.speaker_model,
            )

            response_text = response.text if hasattr(response, "text") else str(response)
            if not response_text:
                logger.warning("Gemini API returned empty response")
                return DEFAULT_SPEAKER_NAMES.copy(), set(), False, True

            # Parse JSON response
            speakers, detected_hosts, success = self._parse_speakers_from_response(
                response_text, known_hosts
            )

            logger.debug(
                "Gemini speaker detection completed: %d speakers, %d hosts, success=%s",
                len(speakers),
                len(detected_hosts),
                success,
            )

            # Track LLM call metrics if available
            # Note: Check Gemini SDK for usage information structure
            if pipeline_metrics is not None and hasattr(response, "usage_metadata"):
                usage = response.usage_metadata
                # Safely extract token counts, handling Mock objects in tests
                input_tokens = getattr(usage, "prompt_token_count", 0)
                output_tokens = getattr(usage, "candidates_token_count", 0)
                # Convert to int (handles Mock objects in tests)
                try:
                    input_tokens = int(input_tokens) if input_tokens is not None else 0
                except (TypeError, ValueError):
                    input_tokens = 0
                try:
                    output_tokens = int(output_tokens) if output_tokens is not None else 0
                except (TypeError, ValueError):
                    output_tokens = 0
                sd_cost: Optional[float] = None
                if input_tokens > 0 or output_tokens > 0:
                    from ...workflow.helpers import calculate_provider_cost

                    sd_cost = calculate_provider_cost(
                        cfg=self.cfg,
                        provider_type="gemini",
                        capability="speaker_detection",
                        model=self.speaker_model,
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                    )
                pipeline_metrics.record_llm_speaker_detection_call(
                    input_tokens, output_tokens, cost_usd=sd_cost
                )

            return speakers, detected_hosts, success, False

        except json.JSONDecodeError as exc:
            logger.error(
                "Failed to parse Gemini API JSON response: %s", format_exception_for_log(exc)
            )
            return DEFAULT_SPEAKER_NAMES.copy(), set(), False, True
        except Exception as exc:
            logger.error("Gemini API error in speaker detection: %s", format_exception_for_log(exc))
            from podcast_scraper.exceptions import (
                ProviderAuthError,
                ProviderRuntimeError,
            )

            # Handle Gemini-specific error types
            error_msg = str(exc).lower()
            if "api key" in error_msg or "authentication" in error_msg or "permission" in error_msg:
                raise ProviderAuthError(
                    message=f"Gemini authentication failed: {format_exception_for_log(exc)}",
                    provider="GeminiProvider/SpeakerDetection",
                    suggestion="Check your GEMINI_API_KEY environment variable or config setting",
                ) from exc
            elif "quota" in error_msg or "rate limit" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Gemini rate limit exceeded: {format_exception_for_log(exc)}",
                    provider="GeminiProvider/SpeakerDetection",
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            elif "invalid" in error_msg and "model" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Gemini invalid model: {format_exception_for_log(exc)}",
                    provider="GeminiProvider/SpeakerDetection",
                    suggestion="Check gemini_speaker_model configuration",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"Gemini speaker detection failed: {format_exception_for_log(exc)}",
                    provider="GeminiProvider/SpeakerDetection",
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
                "GeminiProvider summarization not initialized. Call initialize() first."
            )
        if not (transcript and insight_text):
            return []
        import json

        from ...gi.grounding import QuoteCandidate, resolve_llm_quote_span
        from ..common.evidence_prompts import render_extract_quote_prompt

        system, user = render_extract_quote_prompt(
            "gemini",
            transcript,
            insight_text,
            config_constants.transcript_budget_chars(
                getattr(self, "max_context_tokens", None),
                response_tokens=config_constants.GI_QUOTE_RESPONSE_TOKENS,
            ),
        )
        try:
            from ...utils.provider_metrics import (
                _safe_gemini_retryable,
                apply_gil_evidence_llm_call_metrics,
                gemini_generate_usage_tokens,
                merge_gil_evidence_call_metrics_on_failure,
                ProviderCallMetrics,
                retry_with_metrics,
            )

            call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("gemini")
            call_metrics.set_breaker_config_from_cfg(self.cfg)
            pm = kwargs.get("pipeline_metrics")

            _contents, _cfg_over = self._gemini_stage_request(  # RFC-115
                transcript, system, user, self.summary_model
            )
            generation_config = _merge_generate_content_config(
                self.summary_model,
                {
                    "temperature": 0.0,
                    "max_output_tokens": config_constants.GI_QUOTE_RESPONSE_TOKENS,
                    **_cfg_over,
                },
            )

            def _make_api_call():
                return self.client.models.generate_content(
                    model=self.summary_model,
                    contents=_contents,
                    config=cast(Any, generation_config),
                )

            try:
                response = retry_with_metrics(
                    _make_api_call,
                    **self._gemini_retry_params(
                        {"stage": "gemini_gil_extract_quotes"},
                        pipeline_metrics=pm,
                    ),
                    retryable_exceptions=_safe_gemini_retryable(),
                    metrics=call_metrics,
                )
            except Exception:
                merge_gil_evidence_call_metrics_on_failure(call_metrics, pm)
                raise
            in_tok, out_tok = gemini_generate_usage_tokens(response)
            apply_gil_evidence_llm_call_metrics(
                call_metrics,
                pm,
                in_tok,
                out_tok,
                cfg=self.cfg,
                provider_type="gemini",
                model=self.summary_model,
                stage="extract_quotes",
            )
            content = response.text if hasattr(response, "text") else str(response)
            content = (content or "").strip()
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
            logger.debug("Gemini extract_quotes failed: %s", e, exc_info=True)
            return []

    def extract_quotes_bundled(
        self,
        transcript: str,
        insight_texts: List[str],
        **kwargs: Any,
    ) -> Dict[int, List[Any]]:
        """Bundle ``extract_quotes`` across all insights into one LLM call (#698 Layer A).

        One Gemini round-trip returns a JSON mapping ``{insight_idx_str: [quote, ...]}``
        covering every insight in ``insight_texts``. Each quote is text-only — char
        offsets are resolved locally via :func:`resolve_llm_quote_span` after parsing,
        same as the staged path so downstream NLI sees identical inputs.

        Returns a dict mapping each input index (``range(len(insight_texts))``) to a
        possibly-empty list of :class:`QuoteCandidate`. Empty list means the bundled
        path produced nothing for that insight; the caller is expected to fall back
        to the per-insight staged extract for those rows.

        On any provider/parser failure the method raises so the caller can record a
        single batched-fallback metric and re-run the staged extract for the whole
        episode (mirrors the ``mega_bundled`` policy in ``metadata_generation``).
        """
        if not self._summarization_initialized:
            raise RuntimeError(
                "GeminiProvider summarization not initialized. Call initialize() first."
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

        from ...utils.provider_metrics import (
            _safe_gemini_retryable,
            apply_gil_evidence_llm_call_metrics,
            gemini_generate_usage_tokens,
            merge_gil_evidence_call_metrics_on_failure,
            ProviderCallMetrics,
            retry_with_metrics,
        )

        call_metrics = ProviderCallMetrics()
        call_metrics.set_provider_name("gemini")
        call_metrics.set_breaker_config_from_cfg(self.cfg)
        pm = kwargs.get("pipeline_metrics")

        # Bundled call may need a larger output budget than the per-insight call.
        _contents, _cfg_over = self._gemini_stage_request(  # RFC-115
            clipped, system, user, self.summary_model
        )
        generation_config = _merge_generate_content_config(
            self.summary_model,
            {
                "temperature": 0.0,
                "max_output_tokens": max_out,
                **_cfg_over,
            },
        )

        def _make_api_call() -> Any:
            return self.client.models.generate_content(
                model=self.summary_model,
                contents=_contents,
                config=cast(Any, generation_config),
            )

        try:
            response = retry_with_metrics(
                _make_api_call,
                **self._gemini_retry_params(
                    {"stage": "gemini_gil_extract_quotes_bundled"},
                    pipeline_metrics=pm,
                ),
                retryable_exceptions=_safe_gemini_retryable(),
                metrics=call_metrics,
            )
        except Exception:
            merge_gil_evidence_call_metrics_on_failure(call_metrics, pm)
            raise
        in_tok, out_tok = gemini_generate_usage_tokens(response)
        apply_gil_evidence_llm_call_metrics(
            call_metrics,
            pm,
            in_tok,
            out_tok,
            cfg=self.cfg,
            provider_type="gemini",
            model=self.summary_model,
            stage="extract_quotes",
        )
        content = response.text if hasattr(response, "text") else str(response)

        try:
            parsed = parse_bundled_extract_response(content, expected_count=len(insight_texts))
        except BundleExtractParseError as exc:
            logger.debug("Gemini extract_quotes_bundled parse failed: %s", exc)
            raise

        # Resolve verbatim spans locally; deduplicate per insight.
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

        Uses gemini/insight_extraction/v2 prompt; parses response as one insight per line.
        Returns empty list on failure; the episode then honestly has no insights.
        """
        if not self._summarization_initialized:
            raise RuntimeError(
                "GeminiProvider summarization not initialized. Call initialize() first."
            )

        from ...prompts.store import render_prompt

        max_insights = max(1, min(int(max_insights), config_constants.GI_MAX_INSIGHTS_CEILING))
        insight_max_tokens = max(
            config_constants.GI_INSIGHT_TOKENS_FLOOR,
            max_insights * config_constants.GI_INSIGHT_TOKENS_EACH,
        )
        # Honour the configured temperature. This was hardcoded to 0.3 in every provider,
        # so evals could not pin it and the pipeline was not reproducible.
        insight_temperature = _insight_salvage.resolve_insight_temperature(self.cfg, "gemini")
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
            # The prompt decides what an insight IS, so it is a tuned parameter like any
            # other. Hardcoding it made the extraction prompt the only part of this stage that
            # could not be A/B tested — and left the v3 speech-act prompt reachable by ollama alone.
            prompt_version = str(getattr(self.cfg, "gi_insight_prompt_version", "v2") or "v2")
            user_prompt = render_prompt(
                f"gemini/insight_extraction/{prompt_version}",
                transcript=text_slice,
                title=episode_title or "",
                max_insights=max_insights,
            )
            system_prompt = render_prompt("gemini/insight_extraction/system_v1")
            _contents, _cfg_over = self._gemini_stage_request(  # RFC-115
                text_slice, system_prompt, user_prompt, self.summary_model
            )
            generation_config = _merge_generate_content_config(
                self.summary_model,
                {
                    "temperature": insight_temperature,
                    "max_output_tokens": insight_max_tokens,
                    **_cfg_over,
                },
            )
            response = self.client.models.generate_content(
                model=self.summary_model,
                contents=_contents,
                config=cast(Any, generation_config),
            )
            _record_gemini_llm_call(
                response,
                pipeline_metrics,
                recorder_name="record_llm_gi_call",
                cfg=self.cfg,
                model=self.summary_model,
            )
            content = response.text if hasattr(response, "text") else str(response)
            content = (content or "").strip()

            # ADR-100 cost-attribution: emit llm_cost in both branches.
            in_tok_gi = None
            out_tok_gi = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage_gi = response.usage_metadata
                in_raw = getattr(usage_gi, "prompt_token_count", 0)
                out_raw = getattr(usage_gi, "candidates_token_count", 0)
                try:
                    in_tok_gi = int(in_raw) if in_raw is not None else None
                except (TypeError, ValueError):
                    in_tok_gi = None
                try:
                    out_tok_gi = int(out_raw) if out_raw is not None else None
                except (TypeError, ValueError):
                    out_tok_gi = None
            gi_cost: Optional[float] = None
            if in_tok_gi is not None and out_tok_gi is not None:
                from ...workflow.helpers import calculate_provider_cost

                gi_cost = calculate_provider_cost(
                    cfg=self.cfg,
                    provider_type="gemini",
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
                        provider="gemini",
                        stage="gi",
                        model=self.summary_model,
                        estimated_cost_usd=float(gi_cost or 0.0),
                        prompt_tokens=in_tok_gi,
                        completion_tokens=out_tok_gi,
                        triggered_guardrail=triggered_guardrail,
                        served_model=getattr(response, "model", None),
                        response=response,
                    )
                except Exception:  # noqa: BLE001
                    pass

            # ADR-100: GI is fail-up. Cost emitted in BOTH branches.
            try:
                _guardrails.check_chat_response(
                    content, service="gemini", finish_reason=_gemini_finish_reason(response)
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
            logger.debug("Gemini generate_insights failed: %s", e, exc_info=True)
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
                "GeminiProvider summarization not initialized. Call initialize() first."
            )
        if not (premise and hypothesis):
            return 0.0
        from ..common.evidence_prompts import render_entailment_prompt

        system, user = render_entailment_prompt("gemini", premise, hypothesis)
        try:
            from ...utils.provider_metrics import (
                _safe_gemini_retryable,
                apply_gil_evidence_llm_call_metrics,
                gemini_generate_usage_tokens,
                merge_gil_evidence_call_metrics_on_failure,
                ProviderCallMetrics,
                retry_with_metrics,
            )

            call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("gemini")
            call_metrics.set_breaker_config_from_cfg(self.cfg)
            pm = kwargs.get("pipeline_metrics")

            generation_config = _merge_generate_content_config(
                self.summary_model,
                {
                    "temperature": 0.0,
                    "max_output_tokens": 10,
                    "system_instruction": system,
                },
            )

            def _make_api_call():
                return self.client.models.generate_content(
                    model=self.summary_model,
                    contents=user,
                    config=cast(Any, generation_config),
                )

            try:
                response = retry_with_metrics(
                    _make_api_call,
                    **self._gemini_retry_params(
                        {"stage": "gemini_gil_score_entailment"},
                        pipeline_metrics=pm,
                    ),
                    retryable_exceptions=_safe_gemini_retryable(),
                    metrics=call_metrics,
                )
            except Exception:
                merge_gil_evidence_call_metrics_on_failure(call_metrics, pm)
                raise
            in_tok, out_tok = gemini_generate_usage_tokens(response)
            apply_gil_evidence_llm_call_metrics(
                call_metrics,
                pm,
                in_tok,
                out_tok,
                cfg=self.cfg,
                provider_type="gemini",
                model=self.summary_model,
                stage="score_entailment",
            )
            content = response.text if hasattr(response, "text") else str(response)
            content = (content or "0").strip()
            for part in content.replace(",", " ").split():
                try:
                    v = float(part)
                    return max(0.0, min(1.0, v))
                except ValueError:
                    continue
            return 0.0
        except Exception as e:
            logger.debug("Gemini score_entailment failed: %s", e, exc_info=True)
            return 0.0

    def score_entailment_bundled(
        self,
        pairs: List[Tuple[str, str]],
        chunk_size: int = 15,
        **kwargs: Any,
    ) -> Dict[int, float]:
        """Bundle ``score_entailment`` across many (premise, hypothesis) pairs (#698 Layer B).

        ``pairs`` is the full list of pairs across all insights × all candidates that
        passed the QA threshold. The method chunks at ``chunk_size`` (default 15) and
        issues one Gemini call per chunk, returning a dict mapping each pair index in
        ``range(len(pairs))`` to its NLI score in [0, 1]. Pairs that the model didn't
        return a usable score for are absent from the result — the dispatcher decides
        whether to fall back to per-pair staged calls or drop the candidate.

        Each successful chunk increments the ``score_entailment`` substage cost bucket
        (Phase 1 plumbing) so the autoresearch matrix can attribute the bundled
        savings versus the staged path. A whole-method failure raises so the caller
        can record a fallback metric and re-run the staged NLI loop.
        """
        if not self._summarization_initialized:
            raise RuntimeError(
                "GeminiProvider summarization not initialized. Call initialize() first."
            )
        if not pairs:
            return {}

        chunk_size = max(1, int(chunk_size))
        out: Dict[int, float] = {}
        pm = kwargs.get("pipeline_metrics")
        for chunk_start in range(0, len(pairs), chunk_size):
            chunk = pairs[chunk_start : chunk_start + chunk_size]
            chunk_scores = self._score_entailment_bundled_chunk(
                chunk_pairs=chunk,
                pipeline_metrics=pm,
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
        """One completion: semantic transcript clean + JSON title/bullets (Issue #477)."""
        if not self._summarization_initialized:
            raise RuntimeError(
                "GeminiProvider summarization not initialized. Call initialize() first."
            )

        from ...prompts.store import get_prompt_metadata, render_prompt
        from ...utils.provider_metrics import (
            _safe_gemini_retryable,
            ProviderCallMetrics,
            retry_with_metrics,
        )

        max_out = int(getattr(self.cfg, "llm_bundled_max_output_tokens", 16384) or 16384)
        tmpl_kwargs = dict(self.cfg.summary_prompt_params or {})
        system_prompt = render_prompt(
            "gemini/summarization/bundled_clean_summary_system_v1",
            **tmpl_kwargs,
        )
        user_prompt = render_prompt(
            "gemini/summarization/bundled_clean_summary_user_v1",
            transcript=text,
            title=episode_title or "",
            **tmpl_kwargs,
        )
        from ..ml.diarization.transcript_naming import with_bundled_field

        user_prompt = with_bundled_field(user_prompt, params)

        if call_metrics is None:
            call_metrics = ProviderCallMetrics()
        call_metrics.set_provider_name("gemini")
        call_metrics.set_breaker_config_from_cfg(self.cfg)

        def _make_api_call() -> Any:
            generation_config = _merge_generate_content_config(
                self.summary_model,
                {
                    "temperature": self.summary_temperature,
                    "max_output_tokens": max_out,
                    "response_mime_type": "application/json",
                    "system_instruction": system_prompt,
                },
            )
            return self.client.models.generate_content(
                model=self.summary_model,
                contents=user_prompt,
                config=cast(Any, generation_config),
            )

        try:
            response = retry_with_metrics(
                _make_api_call,
                **self._gemini_retry_params(
                    {
                        "stage": "gemini_bundled_clean_summary",
                        "episode_title": (episode_title or "")[:200],
                    },
                    pipeline_metrics=pipeline_metrics,
                ),
                retryable_exceptions=_safe_gemini_retryable(),
                metrics=call_metrics,
            )
        except Exception:
            call_metrics.finalize()
            raise

        call_metrics.finalize()

        raw = (response.text if hasattr(response, "text") else str(response) or "").strip()

        # Token + cost capture up-front so cost emits in both branches (ADR-100).
        input_tokens = None
        output_tokens = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            input_tokens_raw = getattr(usage, "prompt_token_count", 0)
            output_tokens_raw = getattr(usage, "candidates_token_count", 0)
            try:
                input_tokens = int(input_tokens_raw) if input_tokens_raw is not None else None
            except (TypeError, ValueError):
                input_tokens = None
            try:
                output_tokens = int(output_tokens_raw) if output_tokens_raw is not None else None
            except (TypeError, ValueError):
                output_tokens = None
            if input_tokens is not None and output_tokens is not None:
                call_metrics.set_tokens(input_tokens, output_tokens)

        cost: Optional[float] = None
        if input_tokens is not None:
            from ...workflow.helpers import calculate_provider_cost

            cost = calculate_provider_cost(
                cfg=self.cfg,
                provider_type="gemini",
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
                provider_type="gemini",
                capability="summarization",
                model=self.summary_model,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                triggered_guardrail=triggered_guardrail,
                response=response,  # RFC-115: surfaces cached_content tokens in llm_cost
            )

        # Response-shape guardrail (ADR-100, #1003) on bundled output.
        try:
            _guardrails.check_chat_response(
                raw, service="gemini", finish_reason=_gemini_finish_reason(response)
            )
        except _guardrails.GuardrailViolation:
            _record_cost(triggered_guardrail=True)
            raise

        # The call is paid whether or not its JSON is usable, so it is counted before the shape
        # checks below can raise (2026-10-23 cost audit: a bad answer was never billed).
        _record_cost()

        try:
            data = json.loads(raw, strict=False)
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
                "gemini/summarization/bundled_clean_summary_system_v1",
                params=tmpl_kwargs,
            ),
            "user": get_prompt_metadata(
                "gemini/summarization/bundled_clean_summary_user_v1",
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
                "provider": "gemini",
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

        Companion to :meth:`summarize_bundled` for the 2-call pipeline.
        """
        if not self._summarization_initialized:
            raise RuntimeError(
                "GeminiProvider summarization not initialized. Call initialize() first."
            )

        from ...prompting.megabundle import build_extraction_bundle_prompt
        from ...utils.provider_metrics import (
            _safe_gemini_retryable,
            ProviderCallMetrics,
            retry_with_metrics,
        )
        from ..common.megabundle_parser import parse_extraction_bundle_response
        from ..common.output_tokens import cloud_structured_max_output_tokens

        max_out = int(
            (params or {}).get("max_tokens")
            or getattr(self.cfg, "llm_bundled_max_output_tokens", 16384)
            or 16384
        )
        max_out = cloud_structured_max_output_tokens(self.cfg, max_out)
        language = getattr(self.cfg, "language", "en") or None
        system_prompt, user_prompt = build_extraction_bundle_prompt(
            text,
            language=language,
            # transcript naming's speakers field (transcript_naming.PARAMS_LABELS)
            speaker_labels=(params or {}).get("speaker_naming_labels"),
            max_transcript_chars=config_constants.transcript_budget_chars(
                getattr(self, "max_context_tokens", None), response_tokens=int(max_out)
            ),
        )

        if call_metrics is None:
            call_metrics = ProviderCallMetrics()
        call_metrics.set_provider_name("gemini")
        call_metrics.set_breaker_config_from_cfg(self.cfg)

        def _make_api_call() -> Any:
            generation_config = _merge_generate_content_config(
                self.summary_model,
                {
                    "temperature": 0.0,
                    "max_output_tokens": max_out,
                    "response_mime_type": "application/json",
                    "system_instruction": system_prompt,
                },
            )
            return self.client.models.generate_content(
                model=self.summary_model,
                contents=user_prompt,
                config=cast(Any, generation_config),
            )

        try:
            resp = retry_with_metrics(
                _make_api_call,
                **self._gemini_retry_params(
                    {
                        "stage": "gemini_extraction_bundle",
                        "episode_title": (episode_title or "")[:200],
                    },
                    pipeline_metrics=pipeline_metrics,
                ),
                retryable_exceptions=_safe_gemini_retryable(),
                metrics=call_metrics,
            )
        except Exception:
            call_metrics.finalize()
            raise
        call_metrics.finalize()

        raw_text = (resp.text if hasattr(resp, "text") else str(resp) or "").strip() or "{}"
        if hasattr(resp, "usage_metadata") and resp.usage_metadata is not None:
            try:
                call_metrics.set_tokens(
                    int(getattr(resp.usage_metadata, "prompt_token_count", 0) or 0),
                    int(getattr(resp.usage_metadata, "candidates_token_count", 0) or 0),
                )
            except (TypeError, ValueError):
                pass

        _record_gemini_llm_call(
            resp,
            pipeline_metrics,
            recorder_name="record_llm_summarization_call",
            cfg=self.cfg,
            model=self.summary_model,
        )
        from ...utils.provider_metrics import record_response_cost

        record_response_cost(
            self.cfg,
            resp,
            provider_type="gemini",
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

        #632 research flagged Gemini as "not tier-1" because KG/entity quality
        regressed in mega-bundle mode on fixture transcripts. #646 real-episode
        validation revisits this claim against production audio. Exposed for
        users who want to trade some quality for ~3× lower cost and ~1.5 s/ep
        latency.
        """
        if not self._summarization_initialized:
            raise RuntimeError(
                "GeminiProvider summarization not initialized. Call initialize() first."
            )

        from ...prompting.megabundle import build_megabundle_prompt
        from ...utils.provider_metrics import (
            _safe_gemini_retryable,
            ProviderCallMetrics,
            retry_with_metrics,
        )
        from ..common.megabundle_parser import parse_megabundle_response
        from ..common.output_tokens import cloud_structured_max_output_tokens

        max_out = int(
            (params or {}).get("max_tokens")
            or getattr(self.cfg, "llm_bundled_max_output_tokens", 16384)
            or 16384
        )
        max_out = cloud_structured_max_output_tokens(self.cfg, max_out)
        language = getattr(self.cfg, "language", "en") or None
        system_prompt, user_prompt = build_megabundle_prompt(
            text,
            language=language,
            # transcript naming's speakers field (transcript_naming.PARAMS_LABELS)
            speaker_labels=(params or {}).get("speaker_naming_labels"),
            max_transcript_chars=config_constants.transcript_budget_chars(
                getattr(self, "max_context_tokens", None), response_tokens=int(max_out)
            ),
        )

        if call_metrics is None:
            call_metrics = ProviderCallMetrics()
        call_metrics.set_provider_name("gemini")
        call_metrics.set_breaker_config_from_cfg(self.cfg)

        def _make_api_call() -> Any:
            generation_config = _merge_generate_content_config(
                self.summary_model,
                {
                    "temperature": 0.0,
                    "max_output_tokens": max_out,
                    "response_mime_type": "application/json",
                    "system_instruction": system_prompt,
                },
            )
            return self.client.models.generate_content(
                model=self.summary_model,
                contents=user_prompt,
                config=cast(Any, generation_config),
            )

        try:
            resp = retry_with_metrics(
                _make_api_call,
                **self._gemini_retry_params(
                    {"stage": "gemini_megabundle"},
                    pipeline_metrics=pipeline_metrics,
                ),
                retryable_exceptions=_safe_gemini_retryable(),
                metrics=call_metrics,
            )
        except Exception:
            call_metrics.finalize()
            raise
        call_metrics.finalize()

        raw_text = (resp.text if hasattr(resp, "text") else str(resp) or "").strip() or "{}"
        if hasattr(resp, "usage_metadata") and resp.usage_metadata is not None:
            try:
                call_metrics.set_tokens(
                    int(getattr(resp.usage_metadata, "prompt_token_count", 0) or 0),
                    int(getattr(resp.usage_metadata, "candidates_token_count", 0) or 0),
                )
            except (TypeError, ValueError):
                pass

        _record_gemini_llm_call(
            resp,
            pipeline_metrics,
            recorder_name="record_llm_summarization_call",
            cfg=self.cfg,
            model=self.summary_model,
        )
        from ...utils.provider_metrics import record_response_cost

        record_response_cost(
            self.cfg,
            resp,
            provider_type="gemini",
            capability="summarization",
            stage="summarization",
            model=self.summary_model,
            call_metrics=call_metrics,
        )
        return parse_megabundle_response(raw_text)
