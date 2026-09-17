"""Cross-artifact speaker/role coherence checks (#2062).

The defects behind #2062 all shipped with a green suite because every test in that area asserted
the behaviour of ONE function. None asserted that the FINISHED ARTIFACTS agreed with each other, so
a corpus could claim a host who was never in the episode, hang two identities on one human, or
point a SPOKEN_BY edge at a Person that does not exist, and nothing noticed.

Three files each hold part of the same claim:

* ``metadata.json`` ``content.speakers`` — who the diarization roster HEARD, and in what role;
* ``kg.json`` Person nodes — who those people ARE, with a role the UI renders;
* ``gi.json`` SPOKEN_BY edges — who said each quote.

A single-file check cannot catch a disagreement BETWEEN them, which is the entire bug class. Each
function here returns a list of human-readable violations (empty means coherent) so the same rule
can guard the fixture corpus in CI, a freshly ingested corpus during validation, and production
during a remediation — rather than being re-implemented, slightly differently, in each place.

Advisory by construction: these REPORT, they do not raise and do not mutate. An episode that cost a
transcription should still ship what it has.
"""

from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from ..graph_id_utils import is_bare_speaker_label

#: Roles that assert the person OPENED THEIR MOUTH in this episode.
SPEAKER_ROLES = frozenset({"host", "guest"})

#: Every role the UI knows how to render. Anything else is invisible to filters and cards alike.
KNOWN_ROLES = frozenset({"host", "guest", "mentioned"})

#: How similar two spellings must be to count as one human. ASR mishears names constantly — a
#: roster that heard "skanda eminas" and a graph that says "Skanda Amarnath" describe one person —
#: and a coherence check that called that a violation would report the corpus's most common BENIGN
#: case as its most common defect.
#:
#: 0.70 rather than the resolver's 0.75 because the resolver matches across the WHOLE corpus, where
#: a loose threshold merges strangers, while this matches within ONE episode against a handful of
#: names, where the risk is inverted. Measured on the pairs that actually occur:
#:
#:     skanda amarnath / skanda eminas        0.714   same person (fixture ASR variant)
#:     hanna crebo rediker / hanna krebohticker 0.757  same person (fixture ASR variant)
#:     john smith / john doe                  0.556   different
#:     sarah guo / elad gil                   0.353   different (the real phantom-host case)
#:     ryan knutson / jessica mendoza         0.222   different
#:
#: The nearest true-negative sits 0.16 below the nearest true-positive, so the threshold is not
#: balanced on a knife edge. Revisit it with measurements, not intuition.
FUZZY_THRESHOLD = 0.70


def _fold(name: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace.

    Keeps any letter or digit in ANY script. The previous rule was ``[^a-z0-9]+`` after
    accent-stripping, which deleted every character of an entirely non-Latin name: the fold
    returned ``""``, the empty guard in :func:`same_person` returned False, and
    ``same_person('张川红', '张川红')`` was False. A speaker who cannot match themselves reads as
    "never spoke" on every coherence check — and production carries Round Table China, China Plus,
    ChinaTalk and The Naked Pravda.

    WHAT THIS DOES NOT FIX, since an earlier version of this docstring claimed it did: such a
    speaker is still invisible to m0009's PROMOTION. That path goes through
    :func:`identity.slugify.person_id`, which raises on a name with no ASCII to slug, so
    ``roster_roles`` drops the entry before any folding happens. This fold reaches
    :func:`same_person` and ``voices_in_episode``, not promotion.

    Measured reach of that remaining gap, on the production snapshot: **zero**. Of 13,642 person
    nodes and 4,307 roster entries, 190 names carry a non-ASCII character (141 distinct —
    ``Flávio Bolsonaro``, ``Paul Erdős``, ``Timothée Lacroix``) and every one is Latin-with-
    diacritics and slugs fine; there are no CJK or Cyrillic person names at all. The shows above
    discuss those regions in English. So this is a real hole with no current occupant — worth
    knowing before someone "fixes" it against no evidence, and worth re-measuring before
    concluding it is still empty.

    ``str.isalnum`` is the script-agnostic test; punctuation and symbols still become spaces, so
    the Latin behaviour this was tuned on is unchanged.
    """
    s = unicodedata.normalize("NFKD", str(name or ""))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return " ".join("".join(c if c.isalnum() else " " for c in s).split())


def same_person(a: str, b: str, threshold: float = FUZZY_THRESHOLD) -> bool:
    """True when two spellings plausibly name one human.

    Exact after folding, one a subset of the other's tokens (``"Twiggy"`` vs ``"Twiggy Lawson"``,
    ``"Dr. Adam Rodman"`` vs ``"Adam Rodman"``), or similar enough overall to be an ASR variant.
    """
    fa, fb = _fold(a), _fold(b)
    if not fa or not fb:
        return False
    if fa == fb:
        return True
    ta, tb = set(fa.split()), set(fb.split())
    if ta and tb and (ta <= tb or tb <= ta):
        return True
    return SequenceMatcher(None, fa, fb).ratio() >= threshold


def _persons(kg: Mapping[str, Any]) -> List[dict]:
    return [n for n in (kg.get("nodes") or []) if isinstance(n, dict) and n.get("type") == "Person"]


def _role(node: Mapping[str, Any]) -> str:
    return str((node.get("properties") or {}).get("role") or "").strip().lower()


def _name(node: Mapping[str, Any]) -> str:
    return str((node.get("properties") or {}).get("name") or "").strip()


def roster_names(metadata: Mapping[str, Any]) -> List[str]:
    """Names in ``content.speakers`` — the voices the roster actually placed.

    Since schema 1.2.0 (#2075) the record also lists people a source only NAMED, as
    ``placed: false``. Those are exactly who this check exists to catch, so they are excluded: a
    graph host whose only support is an unplaced entry must still be reported as someone who did
    not speak. A missing flag (pre-1.2.0 artifact) is kept, as before.
    """
    speakers = (metadata.get("content") or {}).get("speakers") or []
    return [
        str(s.get("name") or "").strip()
        for s in speakers
        if isinstance(s, dict) and s.get("name") and s.get("placed") is not False
    ]


def check_speakers_actually_spoke(
    metadata: Mapping[str, Any], kg: Mapping[str, Any], *, label: str = ""
) -> List[str]:
    """Every ``host``/``guest`` in the graph must correspond to a voice the roster heard.

    Host and guest are SPEAKING roles. On the production episodes where diarization named every
    voice it heard, 19.4% of host nodes and 14.3% of guest nodes named someone who did not speak
    (counting every episode gives a larger figure, but a partial roster is SILENT about its
    anonymous voices rather than denying them, so that number is an upper bound) — a co-host who sat
    it out, the show's own name
    as a person, an ASR variant of a real speaker — because the graph was built from the
    pre-diarization hint instead of the roster.

    No roster means no evidence, so nothing is reported: absence of a signal is not a violation.
    """
    spoke = roster_names(metadata)
    if not spoke:
        return []
    out: List[str] = []
    for node in _persons(kg):
        if _role(node) not in SPEAKER_ROLES:
            continue
        who = _name(node)
        if not any(same_person(who, s) for s in spoke):
            out.append(f"{label}{_role(node)}={who!r} never spoke; roster heard {sorted(spoke)}")
    return out


def check_roster_speakers_reach_the_graph(
    metadata: Mapping[str, Any], kg: Mapping[str, Any], *, label: str = ""
) -> List[str]:
    """The converse: a voice the roster NAMED and placed should be in the graph as a speaker.

    This is the #2062 headline from the other direction — 93.2% of roster-named guests never
    reached kg.json, so the corpus had 0.6% guests while the roster named one on 66.9% of episodes.

    NOT part of :func:`check_episode`, deliberately. It assumes every entry in ``content.speakers``
    is a person who belongs in the episode's cast, and that does not hold universally: the committed
    fixture corpus records the ad read as a speaker named "Ad" with ``role="guest"`` on 30 episodes,
    which would make this rule fire 30 times on a corpus that has no speaker defect at all. An ad
    voice is not a guest, but that is a separate argument about what a roster should contain, and a
    coherence guard that is wrong about a third of a corpus is a guard people learn to ignore.

    Call it explicitly where the caller knows the roster holds only real cast — validating a fresh
    ingest, or checking a remediation — rather than as a blanket corpus assertion.
    """
    spoke = roster_names(metadata)
    if not spoke:
        return []
    speakers_in_graph = [_name(n) for n in _persons(kg) if _role(n) in SPEAKER_ROLES]
    out: List[str] = []
    for who in spoke:
        if not any(same_person(who, g) for g in speakers_in_graph):
            out.append(f"{label}roster heard {who!r} but the graph lists no such speaker")
    return out


def check_no_anonymous_speakers(kg: Mapping[str, Any], *, label: str = "") -> List[str]:
    """``SPEAKER_07`` or a bare role word is a placeholder, not someone who can host an episode.

    Delegates to ``graph_id_utils.is_bare_speaker_label`` rather than matching a prefix here: a
    prefix rule flags "Speaker John Knight", who is a real person, and that predicate already
    carries the exact distinction (numbered labels and the role-word set) beside the id BUILDER, so
    the check and the builder cannot drift apart.
    """
    return [
        f"{label}placeholder {_name(n)!r} published as {_role(n)}"
        for n in _persons(kg)
        if _role(n) in SPEAKER_ROLES and is_bare_speaker_label(_name(n))
    ]


def check_roles_are_known(kg: Mapping[str, Any], *, label: str = "") -> List[str]:
    """A role outside the vocabulary renders as nothing and is invisible to every filter."""
    return [
        f"{label}unknown role {_role(n)!r} on {_name(n)!r}"
        for n in _persons(kg)
        if _role(n) and _role(n) not in KNOWN_ROLES
    ]


def check_spoken_by_targets_exist(gi: Mapping[str, Any], *, label: str = "") -> List[str]:
    """A dangling SPOKEN_BY renders an insight whose speaker card is guaranteed empty."""
    ids = {
        n.get("id")
        for n in (gi.get("nodes") or [])
        if isinstance(n, dict) and n.get("type") == "Person"
    }
    return [
        f"{label}SPOKEN_BY {e.get('from')} -> {e.get('to')} (no such Person node)"
        for e in (gi.get("edges") or [])
        if isinstance(e, dict) and e.get("type") == "SPOKEN_BY" and e.get("to") not in ids
    ]


def check_one_quote_one_speaker(gi: Mapping[str, Any], *, label: str = "") -> List[str]:
    """One quote, one mouth.

    A fresh DGX ingest produced 116 SPOKEN_BY edges for 59 quotes: the enrichment pass minted the
    global ``person:twiggy`` while the GI pipeline minted the episode-scoped
    ``person:unresolved-twiggy-{ep}`` for the SAME voice, so every quote hung on both.
    """
    by_quote: Dict[Any, set] = {}
    for e in gi.get("edges") or []:
        if isinstance(e, dict) and e.get("type") == "SPOKEN_BY":
            by_quote.setdefault(e.get("from"), set()).add(e.get("to"))
    return [
        f"{label}quote {q} attributed to {sorted(people)}"
        for q, people in by_quote.items()
        if len(people) > 1
    ]


def check_not_collapsed_onto_one_speaker(
    metadata: Mapping[str, Any], gi: Mapping[str, Any], *, min_quotes: int = 4, label: str = ""
) -> List[str]:
    """The operator-reported symptom, as a rule.

    "I click on insights and there's always the same name listed on all insights." On production
    36.9% of episodes attribute every quote to one person — 6,238 of 18,122 quotes — because
    attribution was sticky across markers it could not recognise. A MULTI-VOICE episode whose every
    attributed quote lands on one person is that signature; a genuine monologue is not, so an
    episode the roster heard as one voice is exempt.

    KNOWN FALSE POSITIVE — measured on a fresh ingest after the fix (2026-09-14). This rule ALSO
    fires on a legitimate shape: an interview where one voice says everything quotable. Talk
    Eastern Europe, "Book Talk: Betrayal" — roster ``[Adam Reichardt host, Luke Harding guest]``,
    diarization cleanly separated them (273 / 118 segments), and all 82 quotes went to the guest.
    Checking every quote's char offset against the transcript's own speaker markers: **81 of 82
    are correctly attributed**. The host asks questions; questions are not claims; claims are what
    become quotes.

    So the 36.9% figure above conflates two populations — the sticky-attribution defect AND
    interview-shaped episodes — and this rule cannot separate them, because the edges alone do not
    say whether the other speaker had anything quotable to say.

    TREAT A HIT AS "look at this episode", not as "attribution is broken". The discriminator that
    would settle it is a quote's char offset against the transcript's markers (the check above),
    which needs the transcript this function is not given. Tightening it that way is worth doing
    and is not done here.
    """
    if len(roster_names(metadata)) < 2:
        return []
    targets = [
        e.get("to")
        for e in (gi.get("edges") or [])
        if isinstance(e, dict) and e.get("type") == "SPOKEN_BY"
    ]
    if len(targets) >= min_quotes and len(set(targets)) == 1:
        return [f"{label}all {len(targets)} attributed quotes -> {targets[0]}"]
    return []


def check_no_show_as_speaker(
    metadata: Mapping[str, Any], kg: Mapping[str, Any], *, label: str = ""
) -> List[str]:
    """A show is not a person who hosts it (#2064).

    The one coherence rule that the roster cannot supply, because the roster is where the defect
    enters: feed host detection seeded the show's own name, so ``content.speakers`` says
    ``host='Africa Tech Summit'`` and every other rule here agrees with it. It is a WRONG entry,
    not a missing one, which is why :func:`check_speakers_actually_spoke` waves it through.

    19 episodes in a 279-episode production sample carry one. The evidence that settles it is the
    feed's own title, which the metadata already holds — so this compares rather than consulting a
    list of words that would need feeding forever.
    """
    feed_title = str((metadata.get("feed") or {}).get("title") or "")
    if not feed_title:
        return []
    from ..speaker_detectors.hosts import names_the_show

    out: List[str] = []
    for node in _persons(kg):
        if _role(node) not in SPEAKER_ROLES:
            continue
        who = _name(node)
        if names_the_show(who, feed_title):
            out.append(f"{label}{_role(node)}={who!r} names the show {feed_title!r}")
    return out


def check_episode(
    metadata: Mapping[str, Any],
    kg: Mapping[str, Any],
    gi: Mapping[str, Any],
    *,
    label: str = "",
) -> List[str]:
    """Every UNIVERSAL coherence rule for one episode, as one list of violations.

    Excludes :func:`check_roster_speakers_reach_the_graph`, which is true of a well-formed cast but
    not of every roster on disk — see its docstring. Call that one explicitly.
    """
    prefix = f"{label}: " if label else ""
    out: List[str] = []
    out += check_speakers_actually_spoke(metadata, kg, label=prefix)
    out += check_no_show_as_speaker(metadata, kg, label=prefix)
    out += check_no_anonymous_speakers(kg, label=prefix)
    out += check_roles_are_known(kg, label=prefix)
    out += check_spoken_by_targets_exist(gi, label=prefix)
    out += check_one_quote_one_speaker(gi, label=prefix)
    out += check_not_collapsed_onto_one_speaker(metadata, gi, label=prefix)
    return out


def check_corpus(episodes: Iterable[Sequence[Any]]) -> List[str]:
    """``check_episode`` over ``(label, metadata, kg, gi)`` tuples."""
    out: List[str] = []
    for label, metadata, kg, gi in episodes:
        out += check_episode(metadata, kg, gi, label=str(label))
    return out
