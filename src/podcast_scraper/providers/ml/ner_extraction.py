"""NER entity extraction for experiment evaluation.

This module provides functions to extract all entity types (not just PERSON)
from text using spaCy models or LLM providers, for use in NER evaluation experiments.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ...utils.log_redaction import format_exception_for_log

logger = logging.getLogger(__name__)

_LLM_NER_SYSTEM = "You extract named entities from text. Return valid JSON only."

_LLM_NER_USER_TEMPLATE = """Extract all person names and organization names from the following text.

Rules:
- PERSON: real people mentioned by name (hosts, guests, experts, public figures)
- ORG: organizations, companies, show/podcast names, institutions
- Return ONLY names that actually appear in the text
- Deduplicate: return each unique name only once
- Do NOT include generic terms like "host", "guest", "speaker"

Return JSON:
{{"persons": ["Name One", "Name Two"], "organizations": ["Org One", "Org Two"]}}

Text:
{text}"""


import re

from ...utils.json_parsing import parse_llm_json, strip_code_fences

# Keep _strip_code_fences as a local alias for backward compatibility
_strip_code_fences = strip_code_fences


def _extract_show_title_from_header(text: str) -> Optional[str]:
    """Extract podcast show title from screenplay-style transcript header.

    Looks for ``# Show Name — Episode`` or ``# Show Name - Episode`` on the
    first line.  Returns the show name or None.
    """
    first_line = text.split("\n", 1)[0].strip()
    m = re.match(r"^#\s+(.+?)\s*[—–\-]\s*Episode", first_line)
    return m.group(1).strip() if m else None


def extract_all_entities(
    text: str,
    nlp: Any,
    labels: Optional[List[str]] = None,
    known_org: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Extract all entities from text using spaCy NER.

    Args:
        text: Text to extract entities from
        nlp: spaCy NLP model (must have NER component)
        labels: Optional list of labels to filter (e.g., ["PERSON", "ORG", "GPE"]).
                If None, extracts all entity types.
        known_org: Optional known ORG name (e.g., podcast/show title from RSS
                   metadata). If provided and not already found by spaCy, every
                   occurrence in the text is added as an ORG entity.  When None,
                   the function attempts to extract the show title from a
                   ``# Show Name — Episode`` header as a fallback.

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

        # Inject known ORG (show title) if not already found by spaCy.
        org_name = known_org or _extract_show_title_from_header(text)
        if org_name and (not labels or "ORG" in labels):
            found_orgs = {e["text"] for e in entities if e["label"] == "ORG"}
            if org_name not in found_orgs:
                # Find all occurrences in transcript and add as ORG entities.
                start = 0
                while True:
                    idx = text.find(org_name, start)
                    if idx == -1:
                        break
                    entities.append(
                        {
                            "start": idx,
                            "end": idx + len(org_name),
                            "text": org_name,
                            "label": "ORG",
                        }
                    )
                    start = idx + len(org_name)
                if org_name not in found_orgs:
                    logger.debug("Injected known ORG '%s' (%d occurrences)", org_name, start)

        return entities
    except Exception as exc:
        logger.error("Error extracting entities: %s", format_exception_for_log(exc), exc_info=True)
        return []


def extract_entities_via_llm(
    text: str,
    provider: Any,
    backend_type: str,
    model: str,
    labels: Optional[List[str]] = None,
    max_input_chars: int = 40000,
) -> List[Dict[str, Any]]:
    """Extract entities from text using an LLM provider's chat API.

    Uses a structured NER prompt and parses JSON response into the same
    entity format as ``extract_all_entities`` (spaCy path).

    Args:
        text: Transcript text to extract entities from.
        provider: Initialized LLM provider (OpenAI, Gemini, Ollama, etc.).
        backend_type: Provider type string ("openai", "gemini", "ollama", etc.).
        model: Model name/ID for the provider.
        labels: Optional list of labels to filter results (default PERSON + ORG).
        max_input_chars: Truncate text to this length to fit context windows.

    Returns:
        List of entity dicts with keys: start, end, text, label.
        Offsets are dummy (0-based) since LLMs return names, not spans.
        Use ``entity_set`` scoring mode (position-agnostic).
    """
    if not text:
        return []

    truncated = text[:max_input_chars]
    user_msg = _LLM_NER_USER_TEMPLATE.format(text=truncated)

    try:
        result_json = _call_llm_ner(provider, backend_type, model, user_msg)
    except Exception as exc:
        logger.error(
            "LLM NER extraction failed: %s",
            format_exception_for_log(exc),
            exc_info=True,
        )
        return []

    want_labels = set(labels) if labels else {"PERSON", "ORG"}
    entities: List[Dict[str, Any]] = []

    if "PERSON" in want_labels:
        for name in result_json.get("persons", result_json.get("PERSON", [])):
            if isinstance(name, str) and name.strip():
                entities.append(
                    {"start": 0, "end": len(name), "text": name.strip(), "label": "PERSON"}
                )

    if "ORG" in want_labels:
        for name in result_json.get("organizations", result_json.get("ORG", [])):
            if isinstance(name, str) and name.strip():
                entities.append(
                    {"start": 0, "end": len(name), "text": name.strip(), "label": "ORG"}
                )

    return entities


def _call_llm_ner(
    provider: Any,
    backend_type: str,
    model: str,
    user_msg: str,
) -> Dict[str, Any]:
    """Dispatch LLM NER call to the appropriate provider API."""
    if backend_type == "openai":
        response = provider.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _LLM_NER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return parse_llm_json(response.choices[0].message.content or "{}")

    elif backend_type == "gemini":
        import google.genai as genai

        response = provider.client.models.generate_content(
            model=model,
            contents=user_msg,
            config=genai.types.GenerateContentConfig(
                temperature=0.0,
                system_instruction=_LLM_NER_SYSTEM,
                response_mime_type="application/json",
            ),
        )
        return parse_llm_json(response.text or "{}")

    elif backend_type == "ollama":
        from openai import OpenAI

        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _LLM_NER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return parse_llm_json(response.choices[0].message.content or "{}")

    elif backend_type == "anthropic":
        response = provider.client.messages.create(
            model=model,
            max_tokens=1024,
            system=_LLM_NER_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0.0,
        )
        return parse_llm_json(response.content[0].text if response.content else "{}")

    elif backend_type == "mistral":
        response = provider.client.chat.complete(
            model=model,
            messages=[
                {"role": "system", "content": _LLM_NER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
        )
        return parse_llm_json(response.choices[0].message.content or "{}")

    elif backend_type in ("deepseek", "grok"):
        response = provider.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _LLM_NER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
        )
        return parse_llm_json(response.choices[0].message.content or "")

    else:
        raise ValueError(f"Unsupported backend for LLM NER: {backend_type}")
