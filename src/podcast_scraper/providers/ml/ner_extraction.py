"""NER entity extraction for experiment evaluation.

This module provides functions to extract all entity types (not just PERSON)
from text using spaCy models, for use in NER evaluation experiments.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def extract_all_entities(
    text: str,
    nlp: Any,
    labels: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Extract all entities from text using spaCy NER.

    Args:
        text: Text to extract entities from
        nlp: spaCy NLP model (must have NER component)
        labels: Optional list of labels to filter (e.g., ["PERSON", "ORG", "GPE"]).
                If None, extracts all entity types.

    Returns:
        List of entity dicts with keys: start, end, text, label
    """
    if not text or not nlp:
        return []

    try:
        doc = nlp(text)
        entities = []

        for ent in doc.ents:
            # Filter by labels if specified
            if labels and ent.label_ not in labels:
                continue

            entities.append(
                {
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "text": ent.text,
                    "label": ent.label_,
                }
            )

        return entities
    except Exception as exc:
        logger.error(f"Error extracting entities: {exc}", exc_info=True)
        return []
