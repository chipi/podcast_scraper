"""LLM-as-judge relevance grading for the hybrid-vs-baseline eval (RFC-057 / Step 2).

Turns the human-grading bottleneck into a spot-check: an LLM grades each (query,
candidate) for relevance (0 irrelevant / 1 related / 2 directly answers), filling a
``JudgmentRecord.relevance`` so ``judged_eval.score_from_judgments`` can produce
discriminating nDCG/recall per backend. A human then validates a small sample.

The module is dependency-free and testable: the actual model call is injected as a
``complete: Callable[[str], str]`` (prompt → response text), so unit tests use a fake
and the script (``eval_hybrid_judged.py auto``) passes a real OpenAI-backed callable.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Sequence

from .judged_eval import JudgmentRecord

Complete = Callable[[str], str]

_GRADE_RANGE = (0, 1, 2)


def build_grading_prompt(record: JudgmentRecord) -> str:
    """Build the grading prompt: query + numbered candidates → ask for a JSON map."""
    lines = [
        "You are grading podcast-corpus search results for relevance.",
        f'Query: "{record.query}"',
        "",
        "Rate EACH result: 0 = irrelevant, 1 = related, 2 = directly answers the query.",
        'Return ONLY a JSON object mapping doc_id to grade, e.g. {"a": 2, "b": 0}.',
        "",
        "Results:",
    ]
    for cand in record.candidates:
        text = (cand.get("text") or "")[:400].replace("\n", " ")
        lines.append(f'- id={cand.get("doc_id")}: {text}')
    return "\n".join(lines)


def parse_grades(response: str, candidate_ids: Sequence[str]) -> Dict[str, int]:
    """Extract a ``{doc_id: grade}`` map from the model's response (robust to prose).

    Only ids in *candidate_ids* are kept; grades are clamped to 0/1/2; unparsable →
    empty (the record is then skipped as having no signal).
    """
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        return {}
    try:
        raw = json.loads(match.group(0))
    except (ValueError, TypeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    valid = set(candidate_ids)
    out: Dict[str, int] = {}
    for doc_id, grade in raw.items():
        if doc_id in valid:
            try:
                g = int(grade)
            except (ValueError, TypeError):
                continue
            out[doc_id] = g if g in _GRADE_RANGE else max(0, min(2, g))
    return out


def grade_record(record: JudgmentRecord, complete: Complete) -> JudgmentRecord:
    """Fill *record.relevance* by asking the model to grade its candidates."""
    ids = [c.get("doc_id") for c in record.candidates]
    try:
        response = complete(build_grading_prompt(record))
    except Exception:  # noqa: BLE001 - a failed grade leaves the record unjudged
        return record
    record.relevance = parse_grades(response, [i for i in ids if i])
    return record


def grade_records(records: Sequence[JudgmentRecord], complete: Complete) -> List[JudgmentRecord]:
    """Grade every record in place; returns the list for convenience."""
    return [grade_record(r, complete) for r in records]
