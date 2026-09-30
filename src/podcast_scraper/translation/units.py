"""Translation units: sentence groups inside a turn (RFC-124 §5.1 / S2.3).

TWO GRANULARITIES, AND COLLAPSING THEM BREAKS BOTH JOBS.

- A **unit** is the translation CONTEXT: a greedily packed group of sentences inside a single
  RFC-123 turn. It exists so the model sees enough around a sentence to translate it well.
- A **sentence** is the ALIGNMENT ATOM: what gets stored, offset and rendered.

RFC-124 §5.1 is explicit about why one ~120-word block cannot serve both. The ad-free builder
drops any segment overlapping an excised range (`adfree_transcript.py`), so with one
pseudo-segment per unit every ad boundary would discard up to ~45 seconds of real speech instead
of one 5-15 s Whisper fragment. And `.en.segments.json` is served as subtitle cues — a 120-word
cue is a paragraph, not a subtitle. **This is the one v1 decision that cannot be cheaply
reversed**: changing alignment granularity later re-translates every episode and invalidates
every provenance block.

UNITS NEVER CROSS A TURN BOUNDARY, because a turn is one speaker's uninterrupted run and
translating across a speaker change would let one person's words be rendered in the grammar of
another's. Backchannel turns get one unit each — they are already flagged rather than merged
(RFC-123 §2.2) precisely so they are not folded into a neighbour's context.

LABELS ARE NEVER IN A UNIT'S PAYLOAD (D-24). The speaker label is carried on the unit for
provenance and re-applied to the English line by the renderer; sending it through the translator
would rename the same person inconsistently between units.

PACKING IS DETERMINISTIC given the same turns and budget — it has to be, because a unit's
identity is its position and any wobble would silently re-key provenance.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: The prompt template's fixed instruction, in tokens. Measured at 74 with the real tokenizer
#: for ``translategemma_v1``.
TEMPLATE_TOKENS = 74

#: Per-sentence cost of the numbered request shape: ``"N. "`` plus the newline.
#:
#: Measured 2 tokens per line with the pessimistic character estimator; 4 is the conservative
#: bound, because Gemma tokenizes digits INDIVIDUALLY — ``"\n123. "`` is about 5 tokens at a
#: three-digit index against 3 at one digit. A reserve that is right for two digits and wrong
#: for three is the H1 failure in a narrower window, so the sentence count is capped below
#: three digits rather than the reserve being grown to cover a case that should not arise.
PER_SENTENCE_NUMBERING_TOKENS = 4

#: Hard cap on sentences per unit, so every index stays two digits and the reserve above is
#: provably sufficient. Also a sanity bound in its own right: the measured real transcript's
#: longest TURN was 41 words, so a 100-sentence unit means diarization merged a rapid exchange
#: into one turn — a shape worth splitting for the model's sake regardless of tokens.
MAX_SENTENCES_PER_UNIT = 99

#: Legacy flat reserve. KEPT ONLY AS A FLOOR for callers that pass nothing, and it is NOT what
#: the packer budgets against any more — a review found why. A flat 128 ignored the per-sentence
#: numbering, so a unit packed to the flat budget could carry ~96 sentences whose numbering added
#: ~290 tokens, and the provider's own precheck (which measures the RENDERED prompt) then refused
#: it. The unit is not oversized by the packer's reckoning, so it is not flagged; it simply fails,
#: withholds the whole episode's render, and fails identically on every retry. Latent only
#: because the measured fixture's longest turn was 41 words.
DEFAULT_OVERHEAD_TOKENS = 128

#: Fallback when no tokenizer is available: the FEWEST chars per token measured over 141 real
#: Spanish units. Pessimistic on purpose — it over-counts, so packing errs toward smaller units.
_PESSIMISTIC_CHARS_PER_TOKEN = 2.2


@dataclass
class UnitSentence:
    """One sentence: the alignment atom, with its span in the SOURCE screenplay."""

    sent_id: str
    text: str
    char_start: int
    char_end: int

    def to_dict(self) -> Dict[str, Any]:
        """One sentence with its char span into the unit's text.

        The span is carried, not recomputed downstream, because the sentence is the alignment
        ATOM while the unit is only context (RFC-124 §5.1) — re-splitting the text later could
        produce different boundaries and silently break the alignment.
        """
        return {
            "sent_id": self.sent_id,
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


@dataclass
class TranslationUnit:
    """One translation request's worth of text, bounded by a turn."""

    unit_id: str
    turn_id: str
    speaker_label: str
    sentences: List[UnitSentence]
    backchannel: bool = False
    #: True when a SINGLE sentence exceeds the budget on its own. It cannot be split further
    #: without breaking the alignment atom, so it is kept whole and flagged: the provider will
    #: refuse it and the episode records a failed unit, which is visible. Silently sending it
    #: would return a translation of its first clause with ``finish_reason: stop`` — measured.
    oversized: bool = False
    content_key: str = field(default="", repr=False)

    @property
    def source_text(self) -> str:
        """The payload. Sentences joined by a space — NO speaker label, ever (D-24)."""
        return " ".join(s.text for s in self.sentences)

    @property
    def numbered_source(self) -> str:
        """The payload as numbered lines, for the sentence-aligned request shape (RFC-124 §5.1).

        Kept separate from :attr:`source_text` because the fallback path deliberately sends the
        unit as ONE block when numbered output comes back the wrong length.
        """
        return "\n".join(f"{i}. {s.text}" for i, s in enumerate(self.sentences, start=1))

    def to_dict(self) -> Dict[str, Any]:
        """One translation unit: its sentences, its turn, and whether it is a backchannel.

        `speaker_label` is the anonymous label rather than a resolved name, because units are
        built BEFORE naming (D-34) — a unit that carried a person's name would be asserting an
        identity nothing has decided yet.
        """
        out: Dict[str, Any] = {
            "unit_id": self.unit_id,
            "turn_id": self.turn_id,
            "speaker_label": self.speaker_label,
            "backchannel": self.backchannel,
            "content_key": self.content_key,
            "sentences": [s.to_dict() for s in self.sentences],
        }
        if self.oversized:
            out["oversized"] = True
        return out


def content_key(source_language: str, sentences: Sequence[UnitSentence]) -> str:
    """Hash of ``(source_language, the unit's sentence texts)`` — RFC-124 §5.1b.

    STABLE ACROSS RENUMBERING, which is the whole point. ``unit_id`` is turn-ordinal, so
    anything that renumbers turns changes it; a rename changes ``Label:`` prefix lengths and
    therefore every char offset, but never the unit TEXT. So this key still hits, and a naming
    repair on a translated show costs a re-render instead of a full re-translation — naming
    repair being the most common repair in this corpus.

    Offsets are deliberately NOT in the hash, for exactly that reason.
    """
    h = hashlib.sha256()
    h.update((source_language or "").encode("utf-8"))
    for s in sentences:
        h.update(b"\x00")
        h.update(s.text.encode("utf-8"))
    return h.hexdigest()


def _estimate(text: str) -> int:
    return int(len(text) / _PESSIMISTIC_CHARS_PER_TOKEN) + 1


def pack_units(
    turns: Sequence[Dict[str, Any]],
    *,
    screenplay_text: str,
    source_language: str,
    max_input_tokens: int,
    overhead_tokens: int = DEFAULT_OVERHEAD_TOKENS,
    count_tokens: Optional[Callable[[str], Optional[int]]] = None,
) -> List[TranslationUnit]:
    """Pack a turns artifact's sentences into units. Pure and deterministic.

    THE TEXT COMES FROM THE SCREENPLAY, NOT FROM THE ARTIFACT. ``turns.json`` stores only
    ``sent_id`` and a char span — it deliberately carries no text, so the artifact cannot drift
    from the transcript and nothing is stored twice. Slicing here means a unit's payload IS the
    screenplay at those offsets, by construction, which is the same identity S1.1's invariants
    guarantee per turn. A packer that accepted text alongside offsets could be handed two
    versions of the same sentence and would have no way to notice.

    Args:
        turns: The ``turns`` array of a ``turns.json`` document (RFC-123).
        screenplay_text: The transcript those offsets index. For the source variant this is
            ``<base>.txt``; passing the wrong body silently mis-slices every unit, which is
            why :func:`podcast_scraper.workflow.transcript_resolution.load_transcript` derives
            body and sidecar together.
        source_language: Normalized tag, for the content key.
        max_input_tokens: The MODEL's input context (2048 for TranslateGemma), not the
            server's ``max-model-len`` — those measure different things and the server will
            happily accept a unit the model cannot read.
        overhead_tokens: What the prompt costs beyond the unit's own text.
        count_tokens: Exact counter, normally the provider's ``/tokenize``. ``None`` — or a
            ``None`` return — falls back to a pessimistic character estimate, so packing errs
            toward smaller units rather than oversized ones.

    Returns:
        Units in document order. ``unit_id`` is ``<turn_id>.uNN``, sentences keep their
        ``sent_id`` from the turns artifact.
    """
    # The budget shrinks as a unit gains sentences, because the numbered request shape costs
    # per sentence. Budgeting against a FLAT reserve is what let the packer and the provider
    # disagree about the same unit.
    fixed_overhead = max(int(overhead_tokens), TEMPLATE_TOKENS)

    def budget_for(sentence_count: int) -> int:
        used = fixed_overhead + PER_SENTENCE_NUMBERING_TOKENS * max(1, sentence_count)
        return max(1, int(max_input_tokens) - used)

    def tokens_of(text: str) -> int:
        if count_tokens is not None:
            exact = count_tokens(text)
            if exact is not None:
                return exact
        return _estimate(text)

    units: List[TranslationUnit] = []
    for turn in turns:
        turn_id = str(turn.get("turn_id") or "")
        label = str(turn.get("speaker_label") or "")
        backchannel = bool(turn.get("backchannel"))
        sentences = []
        for s in turn.get("sentences") or []:
            start = int(s.get("char_start") or 0)
            end = int(s.get("char_end") or 0)
            sliced = screenplay_text[start:end]
            if not sliced.strip():
                continue
            sentences.append(
                UnitSentence(
                    sent_id=str(s.get("sent_id") or ""),
                    text=sliced,
                    char_start=start,
                    char_end=end,
                )
            )
        if not sentences:
            continue

        if backchannel:
            # One unit, whatever its sentence count. A backchannel is flagged rather than merged
            # (RFC-123 §2.2) so it is not folded into a neighbour's context; packing it with a
            # neighbour here would undo that on the way to the model.
            groups: List[List[UnitSentence]] = [sentences]
        else:
            groups = []
            current: List[UnitSentence] = []
            current_tokens = 0
            for sent in sentences:
                cost = tokens_of(sent.text)
                if cost > budget_for(1):
                    # A single sentence over budget even alone. It cannot be split without
                    # breaking the alignment atom, so flush what we have and give it its own
                    # unit — where it is FLAGGED rather than silently sent.
                    if current:
                        groups.append(current)
                        current, current_tokens = [], 0
                    groups.append([sent])
                    continue
                # The budget is evaluated for the unit this sentence would CREATE, numbering
                # included, so a unit can never be packed past what the provider will accept.
                if current and (
                    current_tokens + cost > budget_for(len(current) + 1)
                    or len(current) >= MAX_SENTENCES_PER_UNIT
                ):
                    groups.append(current)
                    current, current_tokens = [], 0
                current.append(sent)
                current_tokens += cost
            if current:
                groups.append(current)

        for idx, group in enumerate(groups, start=1):
            oversized = len(group) == 1 and tokens_of(group[0].text) > budget_for(1)
            units.append(
                TranslationUnit(
                    unit_id=f"{turn_id}.u{idx:02d}",
                    turn_id=turn_id,
                    speaker_label=label,
                    sentences=group,
                    backchannel=backchannel,
                    oversized=oversized,
                    content_key=content_key(source_language, group),
                )
            )
    return units


def pack_stats(units: Sequence[TranslationUnit]) -> Dict[str, Any]:
    """What the manifest records about packing — the numbers S2.10 sizes capacity from."""
    sentence_counts = [len(u.sentences) for u in units]
    return {
        "units": len(units),
        "sentences": sum(sentence_counts),
        "backchannel_units": sum(1 for u in units if u.backchannel),
        "oversized_units": sum(1 for u in units if u.oversized),
        "max_sentences_per_unit": max(sentence_counts) if sentence_counts else 0,
    }
