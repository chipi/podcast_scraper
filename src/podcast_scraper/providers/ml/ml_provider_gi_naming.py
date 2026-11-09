"""MLProvider's GI and naming methods (E04 split line, S1).

Moved verbatim out of ``MLProvider`` (``ml_provider``), which inherits this
class: insight extraction, grounding and its quotes, entailment scoring, insight
classification, speaker and host detection, and the bundled summary calls that carry
naming fields. They leave public with GI and naming in the core release; the provider
keeps its call surface.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ...models import Episode
else:
    from ... import models

    Episode = models.Episode  # type: ignore[assignment]
from ...exceptions import ProviderConfigError
from ...languages import transcription_language
from ...utils.log_redaction import format_exception_for_log
from . import speaker_detection

# The provider module's logger (one logger object): records keep their logger name.
logger = logging.getLogger("podcast_scraper.providers.ml.ml_provider")


class MLProviderGiNamingMethods:
    """MLProvider's GI and naming methods; ``self`` is the provider."""

    def detect_hosts(
        self,
        feed_title: Optional[str],
        feed_description: Optional[str],
        feed_authors: Optional[List[str]] = None,
    ) -> Set[str]:
        """Detect host names from feed-level metadata using spaCy NER.

        Args:
            feed_title: Feed title
            feed_description: Feed description (optional)
            feed_authors: List of author names from RSS feed (optional, preferred source)

        Returns:
            Set of detected host names
        """
        # Ensure model is loaded (only if auto_speakers is enabled)
        if self._spacy_nlp is None:
            if not self.cfg.auto_speakers:
                raise ProviderConfigError(
                    message="Cannot detect speakers: auto_speakers is False",
                    provider="MLProvider/spaCy",
                    config_key="auto_speakers",
                    suggestion="Set auto_speakers=True to enable speaker detection",
                )
            self._initialize_spacy()

        return speaker_detection.detect_hosts_from_feed(
            feed_title=feed_title,
            feed_description=feed_description,
            feed_authors=feed_authors,
            nlp=self._spacy_nlp,
            # The FEED's language, for the same reason `detect_speakers` passes it: a feed title
            # and description stay in the source language even after the transcript is
            # translated, so the naming vocabulary has to be the feed's, not the analysis one.
            language=transcription_language(self.cfg) or "",
        )

    def detect_speakers(
        self,
        episode_title: str,
        episode_description: str | None,
        known_hosts: Set[str],
    ) -> Tuple[list[str], Set[str], bool, bool]:
        """Detect speaker names from episode metadata using spaCy NER.

        Args:
            episode_title: Episode title
            episode_description: Optional episode description
            known_hosts: Set of known host names (for context)

        Returns:
            Tuple of:
            - List of detected speaker names
            - Set of detected host names (subset of known_hosts)
            - Success flag (True if detection succeeded)
            - used_defaults: True if defaults were added to reach min speakers
        """
        # Ensure model is loaded (only if auto_speakers is enabled)
        if self._spacy_nlp is None:
            if not self.cfg.auto_speakers:
                raise ProviderConfigError(
                    message="Cannot detect speakers: auto_speakers is False",
                    provider="MLProvider/spaCy",
                    config_key="auto_speakers",
                    suggestion="Set auto_speakers=True to enable speaker detection",
                )
            self._initialize_spacy()

        speaker_names, detected_hosts_set, detection_succeeded, used_defaults = (
            speaker_detection.detect_speaker_names(  # type: ignore[misc]
                episode_title=episode_title,
                episode_description=episode_description,
                nlp=self._spacy_nlp,
                cfg=self.cfg,
                cached_hosts=known_hosts,
                # S2.14. The EPISODE's language, not "en", and the difference is the whole
                # point: the inputs here are the feed's title and description, which stay in
                # the source language even after the TRANSCRIPT has been translated — nothing
                # translates feed metadata. So English NER over a Spanish title is a live
                # hazard for a translated episode too, and §5.2 measured what it produces:
                # recall 2/2 with precision falling 67% -> 18%. Phantom people, not missing
                # ones, and a phantom name becomes a person in the roster and then the KG.
                text_language=transcription_language(self.cfg),
            )
        )

        return speaker_names, detected_hosts_set, detection_succeeded, used_defaults

    def extract_quotes(
        self,
        transcript: str,
        insight_text: str,
        **kwargs: Any,
    ) -> List[Any]:
        """Extract candidate quote spans that support the insight (GIL QA).

        Uses local extractive QA model (gi_qa_model). Returns list of QuoteCandidate
        (char_start, char_end, text, qa_score).
        """
        from ...gi.grounding import QuoteCandidate
        from . import extractive_qa

        if not (transcript and insight_text):
            return []
        question = f"What evidence supports: {insight_text.strip()}"
        try:
            spans = extractive_qa.answer_candidates(
                context=transcript,
                question=question,
                model_id=getattr(self.cfg, "gi_qa_model", "roberta-squad2"),
                device=getattr(self.cfg, "extractive_qa_device", None),
                window_chars=int(getattr(self.cfg, "gi_qa_window_chars", 0)),
                window_overlap_chars=int(getattr(self.cfg, "gi_qa_window_overlap_chars", 250)),
                top_k=3,
            )
        except Exception as e:
            logger.warning(
                "Extractive QA failed for extract_quotes: %s", format_exception_for_log(e)
            )
            return []
        results: list = []
        for span in spans:
            if span.end > len(transcript):
                continue
            # A QA model answers a question; it does not quote. Its span is the ANSWER — a few
            # words ("Codex", median 40 chars) — and a fragment that short cannot entail a full
            # claim, so the NLI gate rejected every one of them and nothing ever grounded. Expand
            # the span to the sentence containing it: that is the evidence a reader wants to see,
            # and it is what NLI needs as a premise.
            start, end = extractive_qa.expand_span_to_sentence(transcript, span.start, span.end)
            results.append(
                QuoteCandidate(
                    char_start=start,
                    char_end=end,
                    text=transcript[start:end],
                    qa_score=span.score,
                )
            )
        return results

    def generate_insights(
        self,
        text: str,
        episode_title: Optional[str] = None,
        max_insights: int = 5,
        params: Optional[Dict[str, Any]] = None,
        pipeline_metrics: Optional[Any] = None,
    ) -> List[str]:
        """Generate insight statements (GIL). ML provider does not implement this.

        Returns empty list; the episode then honestly has no insights
        and the configured summary_provider is ML-only.
        """
        return []

    def score_entailment(
        self,
        premise: str,
        hypothesis: str,
        **kwargs: Any,
    ) -> float:
        """Score entailment of hypothesis given premise (GIL NLI). 0–1, higher = more entailment."""
        from . import nli_loader

        if not (premise and hypothesis):
            return 0.0
        try:
            return nli_loader.entailment_score(
                premise=premise.strip(),
                hypothesis=hypothesis.strip(),
                model_id=getattr(self.cfg, "gi_nli_model", "nli-deberta-base"),
                device=getattr(self.cfg, "nli_device", None),
            )
        except Exception as e:
            logger.warning("NLI failed for score_entailment: %s", format_exception_for_log(e))
            return 0.0
