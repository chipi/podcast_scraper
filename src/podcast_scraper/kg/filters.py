"""#652 Part B — deterministic post-extraction validators for KG topics + entities.

Two filters that run on the final topic/entity lists regardless of source
(``provider``, prefilled from mega/extraction bundle):

1. Topic normalizer — lowercases-strips, trims to ≤ 4 tokens, drops leading
   and medial stopwords, dedupes near-matches within an episode via
   normalized-form equality. Keeps first-occurrence order.

2. Entity-kind repair — maintains a curated ``KNOWN_ORGS`` set seeded from
   the 100-ep `my-manual-run4` corpus; forces ``kind=org`` for exact-match
   only. Source-agnostic — fixes both LLM-assigned kind errors and spaCy
   NER label mistakes. False negatives (missing an org) strictly preferred
   over false positives (wrongly overriding). Anything not in the list:
   the model's / NER's answer governs.
"""

from __future__ import annotations

import difflib
import re
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Topic normalizer (Finding 2)
# ---------------------------------------------------------------------------

_TOPIC_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "of",
        "for",
        "vs",
        "to",
        "on",
        "at",
        "by",
        "from",
        "with",
    }
)

# Bumped from 4 → 6 after #652 stabilization. Real-corpus audit showed many
# genuine multi-word topics like "AI ethics and public perception" or "global
# oil supply chain" that get mangled at 4 tokens.
_TOPIC_MAX_TOKENS = 6

# Strip punctuation EXCEPT '&' and '-' (preserves "P&I", "AT&T", "ai-agents").
# Apostrophes are handled separately so "China's" → "chinas" not "china s".
_PUNCTUATION_RE = re.compile(r"[^\w\s\-&]")
_APOSTROPHE_RE = re.compile(r"'")
_MULTI_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_topic_label(label: str) -> Optional[str]:
    """Return a lower-cased, stopword-stripped, ≤6-token topic label, or None.

    Design (post-#652-stabilization audit on ``my-manual-run4`` 100-ep corpus):

    * Lowercase + collapse whitespace (always).
    * Strip apostrophes WITHOUT inserting a space ("China's" → "chinas").
    * Strip punctuation EXCEPT ``&`` and ``-`` so "P&I", "AT&T", "ai-agents"
      survive.
    * Cap at ≤6 tokens (was 4 — too aggressive; lost meaning on multi-word
      topics like "AI ethics and public perception").
    * Strip leading + trailing stopwords ONLY. Medial stopwords are KEPT
      because removing them destroyed meaning ("International Group of P&I
      Clubs" → "international group p" was a regression).
    * Dedupe via normalized-form equality at the caller.
    """
    if not label:
        return None
    text = label.lower()
    text = _APOSTROPHE_RE.sub("", text)  # "china's" → "chinas" (no orphan)
    text = _PUNCTUATION_RE.sub(" ", text)
    text = _MULTI_WHITESPACE_RE.sub(" ", text).strip()
    if not text:
        return None
    tokens = text.split(" ")
    # Drop leading stopwords.
    while tokens and tokens[0] in _TOPIC_STOPWORDS:
        tokens = tokens[1:]
    # Drop trailing stopwords ("markets of the" → "markets of" → "markets").
    while tokens and tokens[-1] in _TOPIC_STOPWORDS:
        tokens = tokens[:-1]
    # Cap at max tokens AFTER stopword trimming.
    tokens = tokens[:_TOPIC_MAX_TOKENS]
    # NOTE: medial stopwords are intentionally preserved. The previous
    # implementation stripped them too, which destroyed meaning for topics
    # like "personal journeys of dissent" → "personal journeys dissent".
    if not tokens:
        return None
    return " ".join(tokens)


def normalize_topic_labels(labels: Sequence[str]) -> Tuple[List[str], int]:
    """Normalize + dedupe a topic list. Returns ``(normalized, change_count)``.

    ``change_count`` counts every label whose normalized form differs from
    the input OR that was dropped as a near-duplicate. Used to populate the
    ``topics_normalized_count`` metric.
    """
    out: List[str] = []
    seen_normalized: set[str] = set()
    changes = 0
    for raw in labels:
        raw_str = str(raw or "").strip()
        normalized = _normalize_topic_label(raw_str)
        if normalized is None:
            changes += 1
            continue
        if normalized in seen_normalized:
            changes += 1
            continue
        seen_normalized.add(normalized)
        if normalized != raw_str:
            changes += 1
        out.append(normalized)
    return out, changes


# ---------------------------------------------------------------------------
# Entity-kind repair (Finding 7)
# ---------------------------------------------------------------------------

# Curated set seeded from 100-ep `my-manual-run4` observations. False negatives
# (missing an org) strictly preferred over false positives.
KNOWN_ORGS: frozenset[str] = frozenset(
    {
        # Podcasts / shows that NER mis-classifies as people.
        "npr",
        "planet money",
        "the daily",
        "the journal",
        "tomorrow's cure",
        "no priors",
        "invest like the best",
        # Media organisations.
        "wsj",
        "the wall street journal",
        # Sponsor-ad companies from the 100-ep top-cluster analysis.
        "ramp",
        "workos",
        "rogo",
        # Common tech / AI orgs referenced in podcast episodes.
        "openai",
        "anthropic",
        "google",
        "meta",
        "amazon",
        "microsoft",
        "tesla",
        "nvidia",
        "apple",
    }
)


def repair_entity_kind(entities: Sequence[dict]) -> Tuple[List[dict], int]:
    """Force ``kind='org'`` on any entity whose name matches ``KNOWN_ORGS``.

    Returns ``(updated_entities, repaired_count)``. All other fields on each
    entity dict are passed through unchanged.
    """
    out: List[dict] = []
    repaired = 0
    for ent in entities:
        if not isinstance(ent, dict):
            out.append(ent)
            continue
        name = str(ent.get("name") or "").strip().lower()
        current_kind = str(ent.get("kind") or "").strip().lower()
        if name and name in KNOWN_ORGS and current_kind != "org":
            new_ent = dict(ent)
            new_ent["kind"] = "org"
            out.append(new_ent)
            repaired += 1
        else:
            out.append(ent)
    return out, repaired


# ---------------------------------------------------------------------------
# Entity name consolidation (#851) — within-episode duplicate-spelling merge
# ---------------------------------------------------------------------------
#
# The KG LLM frequently emits the SAME real entity under two spellings in one
# episode (often the transcript's literal form AND the real-world-correct name),
# e.g. "Burne Hobart" + "Byrne Hobart". Topic normalization (#652) dedupes by
# exact normalized-form equality, which cannot collapse spelling variants. This
# is the deterministic safety net: a conservative, entity-type-aware, within-
# episode merge. The extraction prompt (#851 primary) is responsible for the
# CORRECT surviving spelling; this pass only guarantees one node per entity.
#
# Thresholds are deliberately conservative and centralized for later tuning.
# The within-episode scope is itself a safety feature: two genuinely-different
# similar-named people rarely co-occur in one episode. The acronym guard keeps
# UPS ≠ USPS (the confirmed false-merge landmine).

_PERSON_SURNAME_RATIO = 0.82  # surname similarity to call two person names variants
_PERSON_SURNAME_RATIO_RELAXED = 0.75  # relaxed when first names are identical
_PERSON_FIRST_RATIO = 0.70  # first-name similarity floor
_PERSON_OVERALL_RATIO = 0.80  # whole-string similarity floor
_SINGLE_TOKEN_RATIO = 0.90  # single-token names (non-acronym)
_ORG_RATIO = 0.92  # orgs: strict; only very-high similarity merges
_ACRONYM_MAX_LEN = 5  # single token ≤ this length is treated as an acronym


def _clean_entity_name(name: str) -> str:
    """Lowercase, strip punctuation (keep ``&``/``-``), collapse whitespace."""
    text = _APOSTROPHE_RE.sub("", str(name or "").lower())
    text = _PUNCTUATION_RE.sub(" ", text)
    return _MULTI_WHITESPACE_RE.sub(" ", text).strip()


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _is_acronymish(raw: str, clean: str) -> bool:
    """Single short token, or an all-caps short string — never fuzzy-merged."""
    tokens = clean.split()
    if len(tokens) == 1 and len(clean) <= _ACRONYM_MAX_LEN:
        return True
    raw_compact = re.sub(r"[^A-Za-z]", "", str(raw or ""))
    return bool(raw_compact) and raw_compact.isupper() and len(raw_compact) <= _ACRONYM_MAX_LEN + 1


def _normalize_kind(kind: object) -> str:
    return "org" if str(kind or "").strip().lower() in ("org", "organization") else "person"


def _are_entity_variants(a_raw: str, b_raw: str, kind: str) -> bool:
    """Conservative, type-aware test: are two names the same entity, variant-spelled?"""
    a, b = _clean_entity_name(a_raw), _clean_entity_name(b_raw)
    if not a or not b:
        return False
    if a == b:
        return True
    # Acronym / short-token guard — the UPS vs USPS landmine.
    if _is_acronymish(a_raw, a) or _is_acronymish(b_raw, b):
        return False
    ta, tb = a.split(), b.split()
    if kind == "org":
        return _ratio(a, b) >= _ORG_RATIO
    # Persons: token-structure aware (surname is the discriminator).
    if len(ta) >= 2 and len(tb) >= 2 and len(ta) == len(tb):
        fa, fb, sa, sb = ta[0], tb[0], ta[-1], tb[-1]
        first_ok = (
            fa == fb
            or fa.startswith(fb)
            or fb.startswith(fa)
            or _ratio(fa, fb) >= _PERSON_FIRST_RATIO
        )
        if (
            first_ok
            and _ratio(sa, sb) >= _PERSON_SURNAME_RATIO
            and _ratio(a, b) >= _PERSON_OVERALL_RATIO
        ):
            return True
        # Strong corroboration: identical first name + close surname.
        if fa == fb and _ratio(sa, sb) >= _PERSON_SURNAME_RATIO_RELAXED:
            return True
        return False
    if len(ta) == 1 and len(tb) == 1:
        return _ratio(a, b) >= _SINGLE_TOKEN_RATIO
    return False


def _pick_canonical(members: List[Dict]) -> Dict:
    """Pick the surviving entity: longest name (most complete), tie → lexical.

    Backfills missing/empty fields (e.g. ``description``) from the merged-away
    members so consolidation never drops data. The chosen *name* still wins; only
    absent fields are filled. Note this guarantees *a* spelling, not the correct
    one — the extraction prompt (#851 primary) owns correctness, this pass owns
    deduplication.
    """
    chosen = min(
        members, key=lambda e: (-len(str(e.get("name") or "")), str(e.get("name") or "").lower())
    )
    if len(members) == 1:
        return chosen
    merged = dict(chosen)
    for other in members:
        if other is chosen:
            continue
        for key, value in other.items():
            if key == "name":
                continue
            if merged.get(key) in (None, "") and value not in (None, ""):
                merged[key] = value
    return merged


def consolidate_entity_names(entities: Sequence[dict]) -> Tuple[List[dict], int]:
    """Merge within-episode duplicate-spelling entities. Returns ``(entities, merged_count)``.

    Groups same-kind entities into variant clusters (conservative, type-aware) and
    emits one canonical entity per cluster, preserving the canonical dict's fields
    (``entity_kind``, ``description``, …). First-cluster order preserved.
    """
    clusters: List[Dict] = []  # {"kind": str, "names": [str], "members": [dict]}
    passthrough: List[dict] = []
    for ent in entities:
        name = str(ent.get("name") or "").strip() if isinstance(ent, dict) else ""
        if not name:
            passthrough.append(ent)
            continue
        kind = _normalize_kind(ent.get("entity_kind"))
        for cl in clusters:
            if cl["kind"] == kind and any(_are_entity_variants(name, n, kind) for n in cl["names"]):
                cl["names"].append(name)
                cl["members"].append(ent)
                break
        else:
            clusters.append({"kind": kind, "names": [name], "members": [ent]})

    out: List[dict] = []
    merged = 0
    for cl in clusters:
        out.append(_pick_canonical(cl["members"]))
        merged += len(cl["members"]) - 1
    out.extend(passthrough)
    return out, merged


__all__ = [
    "KNOWN_ORGS",
    "consolidate_entity_names",
    "normalize_topic_labels",
    "repair_entity_kind",
]


# ---------------------------------------------------------------------------
# Conversational-filler guard (found by local validation, 2026-09-03)
# ---------------------------------------------------------------------------

#: Podcast boilerplate the extractor emits as Topic nodes.
#:
#: Measured on ``tests/fixtures/viewer-validation-corpus/v3`` — 13 distinct topics, of which
#: **4 are dropped** and 9 kept, re-measured 2026-09-03 after the rules were relaxed to remove
#: false positives:
#:
#:     dropped: welcome-back-to, great-to-be-back, excited-for-this-one, without-the
#:     kept but junk: diversify-or  (the documented accepted miss — see is_filler_topic)
#:
#: (An earlier note here said "6 of 13", named five, and reported 13 -> 8. Those three cannot all
#: be true; they came from a stricter version of the rules that also deleted real topics. This
#: block is the re-derived count, and the numbers above are reproducible from the checked-in
#: fixture.)
#:
#: These are not merely ugly: a Topic node becomes a theme-cluster member, a trending chip, and a
#: followable navigation destination, so "welcome back to" ends up offered to a listener as a
#: storyline.
#:
#: The normalizer above cannot catch them — it trims stopwords and caps tokens, and "welcome back"
#: is a legitimate-looking two-token phrase by those rules. This is a separate, deliberately
#: CONSERVATIVE question: not "is this label tidy" but "is this a subject at all".
#:
#: False negatives (letting filler through) are strictly preferred over false positives (dropping
#: a real topic), same policy as ``KNOWN_ORGS`` above. Every rule here fires only on labels with
#: no content word left to lose.

#: Known podcast boilerplate, matched EXACTLY on the normalized form.
#:
#: An explicit list cannot produce a false positive the way a heuristic can, so it carries the
#: cases where structure alone is not enough. "great to be back" has a content word ("great") and
#: a sensible shape; only knowing it is a greeting distinguishes it from a topic.
_BOILERPLATE_PHRASES: frozenset[str] = frozenset(
    {
        "welcome back",
        "welcome back to",
        "great to be back",
        "good to be back",
        "glad to be here",
        "great to be here",
        "excited for this one",
        "thanks for listening",
        "thanks for having me",
        "thank you for listening",
        "see you next time",
        "until next time",
        "that is all for today",
        "lets get into it",
        "lets dive in",
        "welcome to the show",
        "back to the show",
    }
)

#: Words that cannot carry a topic on their own.
_FUNCTION_WORDS: frozenset[str] = _TOPIC_STOPWORDS | frozenset(
    {
        "i",
        "me",
        "my",
        "we",
        "us",
        "our",
        "you",
        "your",
        "he",
        "she",
        "it",
        "its",
        "they",
        "them",
        "their",
        "this",
        "that",
        "these",
        "those",
        "there",
        "here",
        "who",
        "what",
        "when",
        "where",
        "why",
        "how",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "will",
        "would",
        "can",
        "could",
        "should",
        "may",
        "might",
        "must",
        "if",
        "then",
        "than",
        "so",
        "as",
        "not",
        "no",
        "yes",
        "all",
        "one",
        "just",
        "very",
        "really",
        "back",
        "again",
        "about",
        "into",
        "out",
        "up",
        "down",
        "over",
        "under",
        "more",
        "most",
        "some",
        "any",
        "much",
        "many",
        "without",
        "within",
        "through",
        "during",
        "before",
        "after",
        "because",
    }
)

#: A label ending in one of these is a truncated fragment ("diversify or", "without the").
_FRAGMENT_TAILS: frozenset[str] = frozenset(
    {
        "and",
        "or",
        "but",
        "the",
        "a",
        "an",
        "of",
        "for",
        "to",
        "in",
        "on",
        "at",
        "by",
        "from",
        "with",
        "vs",
        "if",
        "than",
        "as",
        "into",
        "about",
        "over",
        "under",
    }
)

#: Lead words that mark greeting / sign-off / reaction boilerplate rather than subject matter.
#:
#: Trimmed after measuring false positives. "today", "coming", "great", "happy", "wonderful",
#: "stay" and "dont" were all here and all eat real titles: "Coming Out", "Today", "Great
#: Firewall", "Happy Hour", "Stay Interviews". A word that is ordinary English cannot be a
#: boilerplate marker on its own — only words that are almost exclusively conversational openers
#: qualify, and even then rule 3 requires nothing contentful to follow.
_CONVERSATIONAL_LEADS: frozenset[str] = frozenset(
    {
        "welcome",
        "thanks",
        "thank",
        "hello",
        "hi",
        "hey",
        "goodbye",
        "bye",
        "subscribe",
        "excited",
        "glad",
        "pleased",
        "delighted",
        "awesome",
        "join",
        "tune",
        "lets",
    }
)


def _light_tokens(label: str) -> List[str]:
    """Lowercase + de-punctuate ONLY — no stopword trimming.

    The fragment rule has to run before ``_normalize_topic_label``, which strips trailing
    stopwords and therefore destroys the very evidence that marks a fragment: "diversify or"
    becomes "diversify" and "without the" becomes "without", both of which then look like
    ordinary one-word topics.
    """
    text = label.lower()
    text = _APOSTROPHE_RE.sub("", text)
    text = _PUNCTUATION_RE.sub(" ", text)
    return [tok for tok in _MULTI_WHITESPACE_RE.sub(" ", text).strip().split(" ") if tok]


def _label_was_truncated(label: str, topic_id: str) -> bool:
    """True when the slug id carries substantially MORE content than the label.

    Detects truncation by COMPARISON, not by counting slug segments. Counting was wrong and
    deleted real data: ``identity.slugify`` preserves intra-word hyphens, so a legitimately
    hyphenated label inflates the segment count with no truncation at all. Measured false
    positives on the counting rule — both dropped silently from every enrichment surface:

        "Direct-to-consumer e-commerce go-to-market strategies"   5 words -> 9 segments
        "State-of-the-art large-scale machine-learning systems"   4 words -> 9 segments

    That violates this module's own policy that a real topic is never lost to a heuristic.

    The comparison has no such failure mode: if the label was truncated, the id retains content
    the label lost, so the id is materially longer. If it was not, the two describe the same text
    and their lengths track each other however many hyphens are involved.

        truncated:  label "Product development in frontier AI requires"  (42 chars)
                    id    "product-development-in-frontier-ai-requires-building-for-…" (84)
        intact:     label "Direct-to-consumer e-commerce go-to-market strategies"  (52)
                    id    "direct-to-consumer-e-commerce-go-to-market-strategies"  (52)

    The ratio floor is deliberately generous — a slug can legitimately differ from its label
    (punctuation dropped, "&" spelled out) — so this fires only on a real content gap.
    """

    # BOTH sides split on hyphens, or the comparison is meaningless: the slug always has them
    # expanded, so leaving them intact in the label counts one hyphenated compound as one word
    # against several and manufactures the exact false positive this function exists to avoid.
    def _words(text: str) -> list[str]:
        return _light_tokens(text.replace("-", " ").replace("_", " "))

    slug_words = _words(topic_id.split(":", 1)[-1])
    label_words = _words(label)
    if not slug_words or not label_words:
        return False
    # An id carrying half again as many words as its label means content was cut off.
    return len(slug_words) >= len(label_words) + _TRUNCATION_WORD_GAP


#: Extra words the id must carry before the label counts as truncated.
#:
#: 3, not 1: slugging can legitimately add a word or two ("P&I" -> "p and i"), and the guard must
#: not fire on that. The observed truncations were far larger — 11-13 word propositions cut to
#: 5-8 — so a 3-word gap separates them with room to spare.
_TRUNCATION_WORD_GAP = 3


#: Raw word count above which a "topic" is a CLAIM, not a subject.
#:
#: ``_TOPIC_MAX_TOKENS`` truncates to 6 tokens, which is right for a slightly wordy topic and
#: wrong for a sentence: it turns "Ambition must expand because AI tools flatten the translation
#: layers between functions" into "ambition must expand because ai tools", a fragment that looks
#: like a topic and can never match anything in another episode.
#:
#: Observed on a real DGX pipeline run (2026-09-03, Lenny's Podcast). The LLM emitted 11-13 word
#: propositions as Topic labels and every one was truncated into a unique fragment. Note the run
#: had fallen back to the ollama tier because the DGX vLLM was unreachable, so this is what a
#: DEGRADED provider produces — which is exactly when a guard matters, because nothing else
#: notices. Whatever the model, a topic this long is not a subject, and truncating it hides that.
#:
#: 8, not 6: the normalizer's own cap is 6 and real multi-word topics ("AI ethics and public
#: perception", "global oil supply chain") sit well under it, so 8 rejects sentences without
#: touching anything the cap was widened for.
_TOPIC_MAX_RAW_WORDS = 8


def is_filler_topic(label: str, topic_id: str | None = None) -> bool:
    """True when *label* is conversational boilerplate rather than a subject.

    Pass ``topic_id`` when available. The stored ``properties.label`` has ALREADY been truncated
    to ``_TOPIC_MAX_TOKENS``, so a sentence arrives here looking like a tidy six-word topic and the
    evidence of its real length is gone. The slug id keeps it: a KG written from an 11-word
    proposition carries ``topic:ambition-must-expand-because-ai-tools-flatten-the-translation-…``
    beside a label reading "Ambition must expand because AI tools". The id is the only place the
    truncation is visible after the fact.

    Conservative by construction — see the notes above. Returns False for anything it is not
    confident about, so a real topic is never lost to a heuristic.
    """
    if topic_id and _label_was_truncated(label, topic_id):
        return True

    raw = _light_tokens(label)
    if not raw:
        return True

    # 0. A sentence, not a subject. Rejected rather than truncated — see the constant.
    if len(raw) > _TOPIC_MAX_RAW_WORDS:
        return True

    # 1. Truncated fragment: a dangling conjunction/preposition with no object ("regulation and",
    #    "without the"). Checked on the RAW tokens — see _light_tokens.
    #
    #    Requires THREE or more tokens. A two-word label ending in a function word is as likely a
    #    proper title as a truncation — "Down Under", "Inside Out", "Coming Up" — and world
    #    knowledge is the only thing that separates them. Per this module's policy the miss is
    #    preferable: "diversify or" survives, which is cosmetic, rather than deleting "Down Under"
    #    from every surface, which is silent data loss.
    if len(raw) > 2 and raw[-1] in _FRAGMENT_TAILS:
        return True

    normalized = _normalize_topic_label(label)
    if not normalized:
        return True  # nothing survived normalization — it was punctuation or stopwords

    # 1b. Known boilerplate, matched exactly. Checked against the LIGHT form too, because
    #     normalization strips trailing stopwords ("welcome back to" -> "welcome back").
    if normalized in _BOILERPLATE_PHRASES or " ".join(raw) in _BOILERPLATE_PHRASES:
        return True

    tokens = normalized.split(" ")

    # 2. Nothing but function words: "this one", "back again", "without".
    #
    # Only for a SINGLE token or three-plus. A two-word all-function-word label is too often a
    # real proper title — "Down Under", "Coming Out", "Inside Out" — and the cost of dropping one
    # of those (silent data loss on every surface) outweighs the cost of keeping a rare "this one".
    if len(tokens) != 2 and all(tok in _FUNCTION_WORDS for tok in tokens):
        return True

    # 3. Greeting / sign-off / reaction, e.g. "welcome back", "great to be back",
    #    "excited for this one", "thanks for listening". Requires the lead word AND no content
    #    word after it, so "welcome to the machine" or "great firewall" survive.
    if tokens[0] in _CONVERSATIONAL_LEADS:
        rest = [tok for tok in tokens[1:] if tok not in _FUNCTION_WORDS]
        if not rest:
            return True

    return False
