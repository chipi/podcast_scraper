"""Shared helpers for LLM-based KG graph extraction (topics + entities JSON)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Whisper / subword summarizer artifacts at bullet starts (non-exhaustive; conservative).
_KNOWN_ML_BULLET_PREFIX_RES = (re.compile(r"^(?:Unden|Exting|Distingu)-\s*", re.IGNORECASE),)


def strip_known_ml_bullet_prefixes(text: str) -> str:
    """Remove common broken subword prefixes from summary bullets (ML / ASR noise)."""
    s = (text or "").strip()
    for pat in _KNOWN_ML_BULLET_PREFIX_RES:
        s = pat.sub("", s)
    return s.strip()


def build_kg_transcript_system_prompt(max_topics: int, max_entities: int) -> str:
    """System message for transcript-based KG extraction with explicit array caps.

    #1033 — the entity/topic boundary is enforced via worked examples + a
    common-mistakes warning. Empirically (see #112 / EVAL_112) models given
    a "person | organization" rule alone happily emit conceptual nouns
    ("error budgets", "RFCs", "security practices") as orgs. The
    "ONLY proper-noun named entities" rule + the contrast table fix this.
    """
    mt = max(1, min(int(max_topics), 20))
    me = max(1, min(int(max_entities), 50))
    return (
        "You extract a small knowledge-graph fragment from a podcast transcript. "
        "Return ONLY valid JSON (no markdown fences, no commentary) with this shape:\n"
        '{"topics":[{"label":"short topic phrase","description":"optional 1-3 sentences '
        'why this topic matters in this episode"}],"entities":[{"name":"Full Name",'
        '"entity_kind":"person","description":"optional 1-3 sentences on their role '
        'or relevance here"}]}\n'
        "Omit description keys when not useful.\n\n"
        "ENTITY vs TOPIC — CRITICAL DISTINCTION:\n"
        'entity_kind must be exactly one of "person", "organization" or "object". '
        "An ENTITY is a specific, named, proper-noun referent — a real-world "
        "individual, company, brand, podcast, product, place, event or institution "
        "that has a name. If you cannot capitalize it as a proper noun and point to "
        "one specific real-world thing, it is NOT an entity — put it in topics "
        "instead.\n\n"
        "CHOOSING THE KIND:\n"
        '  - "person": a human being.\n'
        '  - "organization": a body of people acting collectively. Test — could it '
        "employ someone or hold a position? A company, university, band or agency "
        "can.\n"
        '  - "object": a named thing that is NEITHER of those — an event, place, '
        "creative work, product, podcast, book or standard.\n"
        "Use object when unsure. NEVER default to person: a wrong person makes a "
        "battle or a podcast appear in the show's list of human voices.\n\n"
        "Correct entity examples (extract these into entities):\n"
        '  - {"name":"Maya","entity_kind":"person"}\n'
        '  - {"name":"Cascadia Alliance","entity_kind":"organization"}\n'
        '  - {"name":"Strava","entity_kind":"organization"}\n'
        '  - {"name":"Singletrack Sessions","entity_kind":"object"}\n'
        '  - {"name":"The Norman Conquest","entity_kind":"object"}\n\n'
        "Common mistakes — these are TOPICS not ENTITIES (do NOT put them in entities):\n"
        '  - "error budgets" → topic (concept), not entity\n'
        '  - "security practices" → topic (concept), not entity\n'
        '  - "RFCs" → topic (concept), not entity\n'
        '  - "trail building" → topic (concept), not entity\n'
        '  - "the host" → not an entity at all (no name)\n'
        '  - "a trail builder" → not an entity at all (no name)\n\n'
        "Quantity guidance: most episodes have 2-8 named entities. If you find "
        "yourself emitting 15+ entities, you are almost certainly labeling topics "
        "or concepts as entities — re-check the distinction above. It is BETTER "
        "to emit 0 entities than to fill the array with conceptual nouns.\n\n"
        "Use one canonical spelling per entity: if a name is spelled inconsistently "
        "in the transcript, pick the single most likely correct spelling and emit it "
        "once — do not also emit the variant as a separate entity. Keep genuinely "
        "different entities (e.g. UPS vs USPS) separate. "
        "Each topic label should be a compact, CANONICAL heading: a stable "
        "noun-phrase naming a concept, prefer about 2–5 words (hard cap 200 "
        "characters). It must read like a subject heading someone would reuse "
        "across episodes — NOT a paraphrase of one sentence from this transcript.\n"
        "GOOD topic labels (canonical concepts):\n"
        '  - "AI security"   - "reward hacking"   - "compute strategy"\n'
        '  - "monetary policy"   - "autonomous delivery"   - "US AI policy"\n'
        "BAD topic labels (NEVER emit — these are sentence fragments / clauses "
        "lifted from the transcript, not headings):\n"
        '  - "attack stemmed from reward"  → use "reward hacking"\n'
        '  - "incident blurs the line"     → use "AI incident attribution"\n'
        '  - "Trump administration is considering" → use "US AI policy"\n'
        '  - "OpenAI\'s unreleased GPT model" → use "unreleased models"\n'
        "Convert every candidate to its canonical concept, or drop it. Avoid long "
        'sentences, comma stacks, leading clauses ("How …", "Why …"), verbs, and '
        "pasting raw transcript lines — put nuance in description instead. "
        "Short stable headings align better across episodes in later tooling. "
        f"Hard limits: at most {mt} objects in topics and at most {me} in entities — "
        "never exceed these array lengths. Order by importance (strongest first); "
        "extras are truncated if you exceed the limit."
    )


_MAX_TOPIC_LABEL_CHARS = 50


#: Words above which a topic label is a PROPOSITION, not a noun phrase — rejected, not truncated.
#:
#: ``_enforce_noun_phrase_label`` cuts an over-long label at ``_MAX_TOPIC_LABEL_CHARS`` and moves
#: the tail into ``description``. For a slightly wordy noun phrase that is right. For a sentence it
#: is actively harmful: truncation does not discard a bad topic, it DISGUISES one. A real DGX
#: pipeline run produced
#:
#:     "Product development in frontier AI requires building for model capabilities two…"
#:
#: which became the label "Product development in frontier AI requires" — indistinguishable in
#: shape from a real topic, unique to its episode, and therefore permanently unclusterable. All 48
#: topics across six episodes were this. Downstream every one inflates the singleton rate and
#: pollutes clustering, co-occurrence and trending.
#:
#: 8 words matches ``kg.filters._TOPIC_MAX_RAW_WORDS`` deliberately — the read-time guard that
#: cleans corpora extracted before this check existed. Extraction stops the pollution at source;
#: the read-time guard handles what is already on disk. Neither replaces the other.
_MAX_TOPIC_LABEL_WORDS = 8


def _is_proposition_not_a_topic(label: str) -> bool:
    """True when *label* is a sentence rather than a noun phrase.

    Word count only, deliberately: this runs on raw provider output before any normalization, so
    it must not depend on the slug, the description, or anything a later stage computes.
    """
    return len(label.split()) > _MAX_TOPIC_LABEL_WORDS


def _enforce_noun_phrase_label(label: str) -> Optional[Tuple[str, Optional[str]]]:
    """Enforce noun-phrase shape on a topic label. ``None`` means "not a topic — drop it".

    If *label* exceeds ``_MAX_TOPIC_LABEL_CHARS``, split at a word boundary: the short part becomes
    the label, the overflow is returned separately (caller appends it to ``description``).

    Returns ``(label, overflow_or_None)``, or ``None`` when the input is a PROPOSITION rather than
    a noun phrase — see :func:`_is_proposition_not_a_topic`.

    The ``None`` return is deliberate rather than a silent pass-through. A first version of this
    check lived at the two call sites in ``llm_extract`` only, and the two in ``kg/pipeline`` — the
    ones the real provider path actually uses — kept truncating. A fresh ingest still produced
    eight truncated propositions and the drop never even logged. Making the REJECTION part of this
    function's contract means a caller cannot accidentally skip it: the type changes, so every site
    has to decide what to do, and mypy fails the ones that do not.
    """
    if _is_proposition_not_a_topic(label):
        return None
    if len(label) <= _MAX_TOPIC_LABEL_CHARS:
        return label, None
    # Cut at word boundary
    cut = label[:_MAX_TOPIC_LABEL_CHARS].rsplit(" ", 1)[0]
    if not cut:
        cut = label[:_MAX_TOPIC_LABEL_CHARS]
    overflow = label[len(cut) :].strip()
    return cut.rstrip(), overflow if overflow else None


def _truncate_kg_description(text: Optional[str], limit: int = 2000) -> Optional[str]:
    s = (text or "").strip()
    if not s:
        return None
    return s[:limit] if len(s) > limit else s


#: Reply budget for the KG extraction call, mirrored from the providers' ``_token_kwarg(2048)``.
KG_RESPONSE_TOKENS = 2048


#: Display names are capped at the same 500 chars the raw path used.
_ENTITY_NAME_MAX_CHARS = 500


def clean_entity_display_name(name: Optional[str]) -> str:
    """Strip extractor punctuation debris from an entity's DISPLAY name (#2055).

    The KG has a documented asymmetry (`kg/pipeline.py:511`): a node's ID is slugified, so
    ``"lukasz kaiser)"`` and ``"lukasz kaiser"`` both become ``person:lukasz-kaiser`` — but the
    LABEL was stored raw. `_dedupe_nodes_by_id` keeps the FIRST node's label, so whichever
    extraction happened to run first decided whether users saw ``Lukasz Kaiser`` or
    ``Lukasz Kaiser)``. Observed in prod: ``Sophia Dew)``, ``Ben Mildenhall)``, ``Lukasz Kaiser)``.

    The debris is almost always an UNBALANCED paren, from a parenthetical alias whose other half
    was consumed upstream. So only unbalanced edges are stripped:

        "Lukasz Kaiser)"            -> "Lukasz Kaiser"
        "(Lukasz Kaiser"            -> "Lukasz Kaiser"
        "Bell Labs (Murray Hill)"   -> unchanged, the parens are balanced and meaningful

    Deliberately conservative about everything else. Real names carry punctuation that a
    broad-brush strip would destroy — ``Jean-Luc Picard``, ``O'Brien``, ``J.R.R. Tolkien``,
    ``will.i.am``, ``Sam Altman, Jr.`` — and silently rewriting a person's name is a worse bug
    than the stray bracket this fixes.
    """
    text = (name or "").strip()
    if not text:
        return ""
    # Repeat: "(Lukasz Kaiser))" needs two passes, and each pass must re-check balance.
    for _ in range(4):
        before = text
        opens, closes = text.count("("), text.count(")")
        if closes > opens and text.endswith(")"):
            text = text[:-1].strip()
        elif opens > closes and text.startswith("("):
            text = text[1:].strip()
        elif opens == closes and text.startswith("(") and text.endswith(")"):
            # Wholly parenthesised — the parens are wrapping, not part of the name.
            inner = text[1:-1].strip()
            if inner and "(" not in inner and ")" not in inner:
                text = inner
        if text == before:
            break
    # Collapse internal whitespace so "Aaron   Levie" and "Aaron Levie" are one person.
    text = " ".join(text.split())
    # A "name" with no letter or digit left in it is punctuation, not a person. Returning it would
    # put an entity called ")" or "()" on a person card; the caller drops an empty name.
    if not any(ch.isalnum() for ch in text):
        return ""
    return text[:_ENTITY_NAME_MAX_CHARS]


def truncate_transcript_for_kg(text: str, limit: Optional[int] = None) -> str:
    """Trim a transcript to what the CALLER's deployment can hold (#2050).

    ``limit`` comes from the provider's context budget. ``None`` means the window is not known —
    nothing declared one, no server to ask — and the transcript is returned uncut so the server's
    own 400 can name the real limit.

    This used to default to ``120000``, applied identically by all six providers regardless of the
    window each was talking to: ~34,000 tokens at the 3.5 chars/token real corpus transcripts
    exhibit, which does not fit a 32,768-token DGX at all, while clipping a 1M-token Gemini to a
    fraction of what it could hold.
    """
    text_slice = (text or "").strip()
    if limit is not None and len(text_slice) > limit:
        return text_slice[:limit] + "\n\n[Transcript truncated.]"
    return text_slice


def build_kg_user_prompt(
    transcript: str,
    title: str,
    max_topics: int,
    max_entities: int,
    prompt_version: str = "v4",
    # v5: #1035 NER pre-pass (extends v4 with optional candidate-spans block).
    # v4: entity canonicalization (#851). v3: noun-phrase (#590).
    ner_entity_hints: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Render shared Jinja prompt for KG extraction.

    Args:
        transcript: Cleaned transcript text (the LLM sees this verbatim).
        title: Episode title (rendered as ``Episode title:`` header).
        max_topics: Hard cap for the topics array.
        max_entities: Hard cap for the entities array.
        prompt_version: Template version to render (``v4`` or ``v5``).
            ``v5`` opts into the #1035 NER pre-pass — when chosen, the
            caller MUST also pass ``ner_entity_hints``.
        ner_entity_hints: Optional list of ``{"text": str, "label": "PERSON"
            | "ORG"}`` dicts produced by ``kg.ner_prepass.extract_kg_ner_hints``.
            Ignored by ``v4`` (which doesn't reference the variable).
            Rendered as a candidate-list block by ``v5``. ``None`` or empty
            list under ``v5`` renders the template without the block —
            falls back to LLM-from-scratch extraction.
    """
    from ..prompts.store import render_prompt

    return render_prompt(
        f"shared/kg_graph_extraction/{prompt_version}",
        transcript=transcript,
        title=title or "",
        max_topics=max_topics,
        max_entities=max_entities,
        ner_entity_hints=ner_entity_hints or [],
    )


def _strip_json_fence(raw: str) -> str:
    content = (raw or "").strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return content


#: The three entity kinds. A person, a body of people, or a NAMED THING that is neither.
#:
#: ``object`` is the catch-all, added 2026-09-13 (#2057). Before it existed the vocabulary was
#: person|organization and the normaliser was two branches — five organisation synonyms, then
#: ``return "person"`` — so ``event``, ``podcast``, ``show``, ``place``, ``book``, ``film``,
#: ``product``, ``concept`` and a MISSING kind all became people. Measured on prod ``top_people``
#: 2026-09-13: 7 of the corpus's top 40 "voices" were not people, and the #1 voice, with 2,720
#: grounded insights, was the Norman Conquest.
#:
#: Forcing those into ``organization`` instead would only move the pollution: a battle is not a
#: company, and "top organizations" would inherit what "top voices" is being cleaned of. The
#: model needed a third bucket, so it has one.
ENTITY_KIND_PERSON = "person"
ENTITY_KIND_ORGANIZATION = "organization"
ENTITY_KIND_OBJECT = "object"
ENTITY_KINDS = (ENTITY_KIND_PERSON, ENTITY_KIND_ORGANIZATION, ENTITY_KIND_OBJECT)

#: Words for an actual human being.
_PERSON_KINDS = frozenset({"person", "people", "individual", "human", "speaker", "guest", "host"})

#: Words for a BODY OF PEOPLE acting collectively. The test is "could it employ someone or hold a
#: position?" — a company, university or band can; a podcast episode or a battle cannot.
_ORGANIZATION_KINDS = frozenset(
    {
        "organization",
        "organisation",
        "org",
        "company",
        "corporation",
        "institution",
        "institute",
        "agency",
        "foundation",
        "university",
        "college",
        "school",
        "publisher",
        "network",
        "studio",
        "label",
        "band",
        "team",
        "group",
        "firm",
        "startup",
        "nonprofit",
        "ngo",
        "government",
        "ministry",
        "department",
        "committee",
        "party",
        "union",
        "club",
    }
)

#: Words for a NAMED THING that is neither a person nor a body of people: events, places,
#: creative works, products, concepts. Listed rather than inferred so the mapping is reviewable —
#: but the list is not load-bearing, because anything unlisted lands here too.
_OBJECT_KINDS = frozenset(
    {
        "object",
        "thing",
        "event",
        "conflict",
        "war",
        "battle",
        "place",
        "location",
        "gpe",
        "country",
        "city",
        "region",
        "facility",
        "work",
        "work_of_art",
        "book",
        "film",
        "movie",
        "album",
        "song",
        "podcast",
        "show",
        "program",
        "programme",
        "series",
        "publication",
        "magazine",
        "newspaper",
        "paper",
        "product",
        "platform",
        "brand",
        "technology",
        "tool",
        "concept",
        "theory",
        "law",
        "language",
        "standard",
        "protocol",
        "award",
        "document",
    }
)


def _normalize_entity_kind(kind: Optional[str]) -> str:
    """Map the extractor's ``entity_kind`` onto :data:`ENTITY_KINDS`.

    ``person`` is returned ONLY when the extractor says so. It is never the fallback — defaulting
    an untrusted value to the most specific, most user-visible type is backwards, and doing exactly
    that is what put an 11th-century military campaign at the top of "top voices" with 2,720
    grounded insights.

    Everything unrecognised — including an ABSENT kind — becomes ``object``. Absence is not
    evidence of personhood; it is evidence of nothing, and ``object`` is the bucket for "a named
    thing we cannot place more precisely". That keeps the entity (no data loss, unlike dropping
    it) while keeping it out of the person and organization surfaces.

    Unrecognised values are logged, so drift between this map and the extraction prompt shows up
    as a countable number rather than as silent misclassification.
    """
    raw = (kind or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw in _PERSON_KINDS:
        return ENTITY_KIND_PERSON
    if raw in _ORGANIZATION_KINDS:
        return ENTITY_KIND_ORGANIZATION
    if raw in _OBJECT_KINDS:
        return ENTITY_KIND_OBJECT
    if not raw:
        logger.info(
            "kg: entity_kind absent — classifying as object, not person (#2057). The extraction "
            "prompt asks for it on every entity, so this counts prompt non-compliance."
        )
    else:
        logger.warning(
            "kg: unrecognised entity_kind %r — classifying as object (#2057). If this appears "
            "often, the extraction prompt and this map have drifted apart.",
            raw,
        )
    return ENTITY_KIND_OBJECT


def _parse_topic_items(raw_topics: Any) -> List[Dict[str, str]]:
    """Parse raw topic items with noun-phrase label enforcement.

    Sentence-shaped labels are DROPPED, not truncated — see ``_MAX_TOPIC_LABEL_WORDS``. Drops are
    logged rather than silent: a provider emitting propositions instead of topics is a quality
    signal about the RUN (it is what a degraded fallback tier does), and an episode whose topics
    all vanish must be attributable to that rather than looking like an episode about nothing.
    """
    out: List[Dict[str, str]] = []
    _dropped_propositions: List[str] = []
    if not isinstance(raw_topics, list):
        return out
    for item in raw_topics:
        if isinstance(item, str) and item.strip():
            enforced = _enforce_noun_phrase_label(item.strip())
            if enforced is None:
                _dropped_propositions.append(item.strip())
                continue
            label, overflow = enforced
            row: Dict[str, str] = {"label": label}
            if overflow:
                row["description"] = overflow
            out.append(row)
        elif isinstance(item, dict):
            lab = item.get("label") or item.get("name") or item.get("topic")
            if not isinstance(lab, str) or not lab.strip():
                continue
            enforced = _enforce_noun_phrase_label(lab.strip())
            if enforced is None:
                _dropped_propositions.append(lab.strip())
                continue
            label, overflow = enforced
            row = {"label": label}
            desc_parts: List[str] = []
            if overflow:
                desc_parts.append(overflow)
            desc = item.get("description")
            if isinstance(desc, str):
                td = _truncate_kg_description(desc)
                if td:
                    desc_parts.append(td)
            if desc_parts:
                row["description"] = ". ".join(desc_parts)
            out.append(row)
    if _dropped_propositions:
        # Loud on purpose. A provider emitting propositions instead of noun phrases is a signal
        # about the RUN, not about the episode — it is what a degraded fallback tier does — and an
        # episode whose topics all vanish must be attributable to that rather than reading as an
        # episode about nothing. Logged at WARNING with a sample so the cause is in the run log.
        logger.warning(
            "kg: dropped %d topic label(s) that were propositions, not noun phrases "
            "(> %d words); sample=%r",
            len(_dropped_propositions),
            _MAX_TOPIC_LABEL_WORDS,
            _dropped_propositions[:3],
        )
    return out


def parse_kg_graph_response(
    raw: str,
    *,
    max_topics: Optional[int] = None,
    max_entities: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Parse model output into {\"topics\": [...], \"entities\": [...]} or None.

    When ``max_topics`` / ``max_entities`` are set, lists are truncated after parsing
    (defense in depth alongside pipeline caps).
    """
    content = _strip_json_fence(raw)
    if not content:
        return None
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}\s*$", content)
        if not m:
            logger.debug("KG JSON parse failed: not valid JSON")
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            logger.debug("KG JSON parse failed after brace extract")
            return None

    if not isinstance(obj, dict):
        return None

    raw_topics = obj.get("topics")
    raw_entities = obj.get("entities")
    topics_out = _parse_topic_items(raw_topics)

    entities_out: List[Dict[str, str]] = []
    if isinstance(raw_entities, list):
        for item in raw_entities:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("label")
            if not isinstance(name, str):
                continue
            # Guard on the CLEANED name, not the raw one (#2055): ")" and "()" are non-empty raw
            # but clean to nothing, and an entity with an empty label lands on a person card.
            name = clean_entity_display_name(name)
            if not name:
                continue
            ek_raw = item.get("entity_kind")
            ek_in = ek_raw if isinstance(ek_raw, str) else None
            erow: Dict[str, str] = {
                "name": name,  # already cleaned above (#2055)
                # Never None: an unplaceable kind becomes `object` rather than being dropped or
                # guessed as a person (#2057).
                "entity_kind": _normalize_entity_kind(ek_in),
            }
            edesc = item.get("description")
            if isinstance(edesc, str):
                ed = _truncate_kg_description(edesc)
                if ed:
                    erow["description"] = ed
            entities_out.append(erow)

    if max_topics is not None and max_topics >= 1:
        topics_out = topics_out[: int(max_topics)]
    if max_entities is not None and max_entities >= 1:
        entities_out = entities_out[: int(max_entities)]

    if not topics_out and not entities_out:
        return None
    return {"topics": topics_out, "entities": entities_out}


def resolve_kg_model_id(provider: Any, params: Optional[Dict[str, Any]]) -> str:
    """Pick model id for KG call: params[\"kg_extraction_model\"] or provider.summary_model."""
    if params and params.get("kg_extraction_model"):
        return str(params["kg_extraction_model"])
    return str(getattr(provider, "summary_model", "") or "unknown")
