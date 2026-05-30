"""spaCy model loading and NER model access."""

from __future__ import annotations

import logging
import subprocess  # nosec B404
import sys
from typing import Any, Optional

from .. import config
from .constants import _VALID_MODEL_NAME_PATTERN, MAX_MODEL_NAME_LENGTH

logger = logging.getLogger(__name__)


def _ensure_spacy_sentence_boundaries(nlp: Any) -> None:
    """Ensure ``doc.sents`` works for downstream code."""
    if "parser" in nlp.pipe_names or "senter" in nlp.pipe_names:
        return
    if "sentencizer" in nlp.pipe_names:
        return
    try:
        nlp.add_pipe("sentencizer")
    except (ValueError, TypeError) as exc:
        logger.debug("Could not add sentencizer to spaCy pipeline: %s", exc)


def _validate_model_name(model_name: str) -> bool:
    """Validate spaCy model name to prevent command injection."""
    if not model_name or len(model_name) > MAX_MODEL_NAME_LENGTH:
        return False
    return bool(_VALID_MODEL_NAME_PATTERN.match(model_name))


def _load_spacy_model(model_name: str) -> Optional[Any]:
    """Load spaCy model, automatically downloading if missing."""
    import spacy  # noqa: F401

    if not _validate_model_name(model_name):
        logger.error("Invalid spaCy model name: %s (contains invalid characters)", model_name)
        return None

    try:
        try:
            nlp = spacy.load(model_name, disable=["parser", "tagger", "lemmatizer"])
            logger.info("Loaded spaCy model (NER only): %s", model_name)
        except (ValueError, KeyError):
            logger.debug(
                "Model %s doesn't support component disabling, loading full pipeline", model_name
            )
            nlp = spacy.load(model_name)
            logger.info("Loaded spaCy model (full pipeline): %s", model_name)
        _ensure_spacy_sentence_boundaries(nlp)
        return nlp
    except OSError:
        logger.debug("spaCy model '%s' not found locally, attempting to download...", model_name)
        try:
            subprocess.run(  # nosec B603
                [sys.executable, "-m", "spacy", "download", model_name],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.debug("Successfully downloaded spaCy model: %s", model_name)
            try:
                nlp = spacy.load(model_name, disable=["parser", "tagger", "lemmatizer"])
                logger.info("Loaded spaCy model (NER only) after download: %s", model_name)
            except (ValueError, KeyError):
                nlp = spacy.load(model_name)
                logger.info("Loaded spaCy model (full pipeline) after download: %s", model_name)
            _ensure_spacy_sentence_boundaries(nlp)
            return nlp
        except subprocess.CalledProcessError as exc:
            logger.error(
                "Failed to download spaCy model '%s': %s. Output: %s",
                model_name,
                exc,
                exc.stderr or exc.stdout or "",
            )
            logger.info("You can manually install with: python -m spacy download %s", model_name)
            return None
        except OSError as exc:
            logger.error(
                "Failed to load spaCy model '%s' after download attempt: %s", model_name, exc
            )
            return None


def get_ner_model(cfg: config.Config) -> Optional[Any]:
    """Get the appropriate spaCy NER model based on configuration."""
    if cfg.dry_run:
        return None

    if not cfg.auto_speakers:
        return None

    model_name = cfg.ner_model
    if not model_name:
        if cfg.language == "en":
            model_name = config.DEFAULT_NER_MODEL
        else:
            logger.debug("No default NER model for language '%s', skipping detection", cfg.language)
            return None

    nlp = _load_spacy_model(model_name)
    if nlp is not None:
        logger.debug("Loaded spaCy model: %s", model_name)

    return nlp
