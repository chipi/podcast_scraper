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


#: Titles someone is ADDRESSED by, never part of the name. The post-nominal half of this problem
#: (``Peter Attia, MD``) already has an owner in ``identity.slugify.canonical_person_name``, which
#: normalises what gets PUBLISHED; these are stripped only for the comparison below, because a
#: title is the one token that can appear on one side of a pair and not the other.
#: DELIBERATELY SHORT. Every token here widens `same_person`, and the widening is not symmetric
#: with the risk: a title that is also a common given name, stage name or surname merges two
#: strangers rather than reuniting one person. `Justice Smith` and `Will Smith` are different
#: actors; `Major Garrett` is a journalist; `Sister Souljah`, `Gen`, `Lady` and `Lord` head stage
#: names. Those are all excluded — the cost of missing them is one unmatched variant, the cost of
#: including them is a wrong identity. Only titles that are vanishingly rare as names survive.
HONORIFIC_PREFIXES = frozenset(
    {
        "dr",
        "mr",
        "mrs",
        "ms",
        "mx",
        "prof",
        "professor",
        "sir",
        "dame",
        "rev",
        "reverend",
        "hon",
        "honorable",
        "honourable",
        "capt",
        "colonel",
        "lieutenant",
        "sgt",
        "sergeant",
        "admiral",
        "senator",
        "ambassador",
        "governor",
    }
)


def _drop_honorifics(folded: str) -> str:
    """*folded* without its leading titles — unchanged when that would leave nothing.

    Only leading tokens are dropped. ``Rev`` and ``Major`` are real surnames, so a title is
    recognised by POSITION as well as spelling, and a name that is nothing but a title
    (``'Professor'`` alone, from a roster that named a voice by its role) keeps its only token
    rather than folding to the empty string, which every caller reads as "no name at all".
    """
    tokens = folded.split()
    while len(tokens) > 1 and tokens[0] in HONORIFIC_PREFIXES:
        tokens.pop(0)
    return " ".join(tokens)


def same_person(a: str, b: str, threshold: float = FUZZY_THRESHOLD) -> bool:
    """True when two spellings plausibly name one human.

    Exact after folding, one a subset of the other's tokens (``"Twiggy"`` vs ``"Twiggy Lawson"``,
    ``"Dr. Adam Rodman"`` vs ``"Adam Rodman"``), or similar enough overall to be an ASR variant.

    A TITLE AND A MISHEARD SPELLING TOGETHER used to defeat all three, though either alone was
    caught: ``"Professor Bruce Lanphier"`` against a graph's ``"Bruce Lanphear"`` fails the subset
    rule (neither token set contains the other once the surnames differ) and then fails the ratio,
    because the extra title lengthens one side enough to drag an otherwise-matching pair under the
    threshold — ``same_person`` was True for that pair without the title and True for the title
    without the misspelling. So the comparison is retried with leading titles dropped. This only
    ever ADDS matches, and m0009 reads a non-match as "this person did not speak", so each one it
    misses demotes a real speaker out of their own episode. Measured on the 2,298-episode
    2026-09-20 production snapshot: of 650 speaking nodes matching no roster entry, this rescues
    exactly 1 (Bruce Lanphear on *Ground Truths*) and makes 0 previously-unambiguous nodes
    ambiguous.
    """
    fa, fb = _fold(a), _fold(b)
    if not fa or not fb:
        return False
    if fa == fb:
        return True
    ta, tb = set(fa.split()), set(fb.split())
    if ta and tb and (ta <= tb or tb <= ta):
        return True
    if SequenceMatcher(None, fa, fb).ratio() >= threshold:
        return True
    ha, hb = _drop_honorifics(fa), _drop_honorifics(fb)
    if (ha, hb) == (fa, fb):
        return False  # no title on either side — nothing the retry could change
    if ha == hb:
        return True
    untitled_a, untitled_b = ha.split(), hb.split()
    # NO SUBSET RULE ON A TITLE-STRIPPED MONONYM. Dropping the title off "Dr. Smith" leaves
    # "smith", and the subset rule reads a mononym as the same human as anyone sharing it — so
    # "Dr. Smith" would become Jane Smith, and "Senator Warren" Elizabeth Warren. The mononym rule
    # exists for a name someone actually goes by ("Twiggy"); a surname left behind by a title is
    # not that, it is the half of a name we just threw the other half away from.
    if len(untitled_a) > 1 and len(untitled_b) > 1:
        sa, sb = set(untitled_a), set(untitled_b)
        if sa <= sb or sb <= sa:
            return True
    return SequenceMatcher(None, ha, hb).ratio() >= threshold


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

    NARROWED 2026-09-26 by the interview-shape exemption below, which removes most of the population
    this rule could not separate. The full discriminator is still a quote's char offset against the
    transcript's markers (the check above), which needs the transcript this function is not given.
    """
    roster = _roster_entries(metadata)
    if len([r for r in roster if r.get("name")]) < 2:
        return []
    targets = [
        e.get("to")
        for e in (gi.get("edges") or [])
        if isinstance(e, dict) and e.get("type") == "SPOKEN_BY"
    ]
    if not (len(targets) >= min_quotes and len(set(targets)) == 1):
        return []

    # THE INTERVIEW-SHAPE EXEMPTION. The false positive documented above is specific: every quote
    # lands on the GUEST while a distinct HOST exists. That is what an interview looks like — the
    # host asks questions, questions are not claims, and claims are what become quotes. On the
    # measured example (Talk Eastern Europe, "Book Talk: Betrayal") 81 of 82 such quotes were
    # CORRECTLY attributed. The sticky-attribution defect this rule exists to catch has the
    # opposite shape: quotes pile onto whoever the marker-blind attribution latched onto first,
    # which on a host-led show is the host. So exempt guest-only, keep reporting host-only.
    #
    # A narrowing, not a silencing: a genuinely broken episode whose quotes happen to land on the
    # guest is still missed, exactly as before. It stops 22 interview-shaped prod episodes being
    # counted as damage, which was inflating every "remaining violations" figure in the #2097 arc.
    sole_target = next(iter(set(targets)))
    role = _roster_role_for_target(sole_target, roster, gi)
    if role == "guest" and any(str(r.get("role") or "").lower() == "host" for r in roster):
        return []
    return [f"{label}all {len(targets)} attributed quotes -> {targets[0]}"]


def _roster_entries(metadata: Mapping[str, Any]) -> List[dict]:
    """``content.speakers`` entries the roster actually PLACED — same filter as ``roster_names``.

    ``content.speakers`` is a LIST of ``{id, name, role, placed, …}``, not a mapping with an
    ``entries`` key; reading it as the latter is a mistake this codebase has already paid for.
    """
    speakers = (metadata.get("content") or {}).get("speakers") or []
    return [
        s
        for s in speakers
        if isinstance(s, dict) and s.get("name") and s.get("placed") is not False
    ]


def _roster_role_for_target(target_id: Any, roster: List[dict], gi: Mapping[str, Any]) -> str:
    """The roster role of the Person a SPOKEN_BY edge points at, or ``""`` when unresolvable.

    The edge carries a graph node id, and the roster keys on name, so this goes through the GI
    Person node to get a name and then matches it with ``same_person`` (the roster may spell it
    differently). Unresolvable returns empty, which keeps the caller REPORTING — an exemption must
    never be granted on a lookup failure.
    """
    name = ""
    for node in gi.get("nodes") or []:
        if isinstance(node, dict) and node.get("id") == target_id:
            name = str((node.get("properties") or {}).get("name") or "").strip()
            break
    if not name:
        return ""
    for entry in roster:
        if same_person(name, str(entry.get("name") or "")):
            return str(entry.get("role") or "").lower()
    return ""


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


# =================================================================================================
# THE ONE-RECORD SYNC CHECK (#2075)
# =================================================================================================
#
# Operator rule 2026-09-17: ONE speaker record per episode (`content.speakers`, schema 1.2.0) and
# every surface that shows people is written from it — the transcript's speaker names (segments),
# who said each quote (gi.json), the people and roles in kg.json, and the operator graph that reads
# both. "Most things in sync and a few not" is the failure the operator named: an episode whose
# surfaces disagree is the defect, however right each file looks on its own.
#
# The checks above predate the record and are advisory about a roster that might be a guess. This
# one judges a RECORD, where each entry already says whether a voice was matched to that person, so
# it can be exact: every surface must agree with the placed entries, and nothing else.


def _labels_of(segments: Any) -> List[str]:
    rows = segments if isinstance(segments, list) else ((segments or {}).get("segments") or [])
    out: List[str] = []
    for r in rows:
        if isinstance(r, dict):
            label = str(r.get("speaker_label") or "").strip()
            if label and label not in out:
                out.append(label)
    return out


def _is_cue_initials(label: str) -> bool:
    """``MG`` — a publisher transcript's cue initials, not a name anyone can be matched to.

    In Moscow's Shadows labels every cue of Mark Galeotti's solo show ``MG``. That is a real
    disagreement with the record (it names no placed person) but not a wrong NAME, and a check that
    failed every one of those episodes would be a check people learn to ignore. Measured on the
    #2075 validation run it is the only label of its kind (3 episodes). Kept narrow on purpose:
    one to three capital letters and nothing else.
    """
    return 1 <= len(label) <= 3 and label.isalpha() and label.isupper()


def _is_display_label(name: str) -> bool:
    """A surface's label for a voice that is not a person: "Unidentified speaker", "Advertisement".

    Written onto quotes and transcript lines so a surface can show SOMETHING for an unnamed voice.
    It claims nobody spoke; it is not a name to match against the record. Read from the roster's own
    table so the two cannot drift. Measured on the validation run: counting these flagged 1,076
    quotes as credited to a person.
    """
    from ..providers.ml.diarization.roster import VOICE_TYPE_LABELS

    return name in set(VOICE_TYPE_LABELS.values()) or is_bare_speaker_label(name)


def placed_people(metadata: Mapping[str, Any], *, legacy_as_placed: bool = False) -> List[dict]:
    """The record's placed entries, or ``[]`` when the artifact carries no record.

    ``legacy_as_placed`` treats every named entry of a pre-1.2.0 roster (no ``placed`` flag) as
    placed. That is only for auditing an existing corpus, and it inherits that roster's errors —
    which is exactly what such an audit wants to surface.
    """
    entries = [
        s
        for s in ((metadata.get("content") or {}).get("speakers") or [])
        if isinstance(s, dict) and str(s.get("name") or "").strip()
    ]
    is_record = any(s.get("placed") is not None for s in entries)
    if is_record:
        return [s for s in entries if s.get("placed") is True]
    return list(entries) if legacy_as_placed else []


def has_speaker_record(metadata: Mapping[str, Any]) -> bool:
    """True when ``content.speakers`` is a 1.2.0 record (its entries carry ``placed``)."""
    return any(
        isinstance(s, dict) and s.get("placed") is not None
        for s in ((metadata.get("content") or {}).get("speakers") or [])
    )


def _sync_cast(
    kg: Mapping[str, Any], placed: List[dict], unplaced: List[str], prefix: str
) -> List[str]:
    """kg.json's host/guest cast against the record's placed and unplaced entries."""
    placed_names = [str(p["name"]) for p in placed]
    cast = [(_name(n), _role(n)) for n in _persons(kg) if _role(n) in SPEAKER_ROLES]
    out: List[str] = []
    for who, role in cast:
        if any(same_person(who, p) for p in placed_names):
            continue
        if any(same_person(who, u) for u in unplaced):
            out.append(f"{prefix}UNPLACED_CAST {role}={who!r} is placed: false in the record")
        else:
            out.append(f"{prefix}CAST_NOT_PLACED {role}={who!r} has no placed voice")
    for p in placed:
        role = str(p.get("role") or "").lower()
        if role in SPEAKER_ROLES and not any(same_person(str(p["name"]), c) for c, _r in cast):
            out.append(f"{prefix}PLACED_NOT_CAST {role}={p['name']!r} is not a speaker in kg.json")
    return out


def _sync_quotes(gi: Mapping[str, Any], placed_names: List[str], prefix: str) -> List[str]:
    """gi.json SPOKEN_BY edges and each quote's own speaker fields, against the placed names."""
    person_name = {
        n.get("id"): str((n.get("properties") or {}).get("name") or "")
        for n in (gi.get("nodes") or [])
        if isinstance(n, dict) and n.get("type") == "Person"
    }
    out: List[str] = []
    edge_by_quote: Dict[Any, Any] = {}
    for e in gi.get("edges") or []:
        if not (isinstance(e, dict) and e.get("type") == "SPOKEN_BY"):
            continue
        edge_by_quote[e.get("from")] = e.get("to")
        who = person_name.get(e.get("to")) or str(e.get("to") or "")
        if who and not any(same_person(who, p) for p in placed_names):
            out.append(f"{prefix}QUOTE_NOT_PLACED {e.get('from')} -> {who!r}")
    for n in gi.get("nodes") or []:
        if not (isinstance(n, dict) and n.get("type") == "Quote"):
            continue
        props = n.get("properties") or {}
        sid, sname = props.get("speaker_id"), props.get("speaker_name")
        edge = edge_by_quote.get(n.get("id"))
        # A quote's own fields only DISAGREE when they say something. The insights view reads
        # `speaker_name` / `speaker_id` first and falls back to the SPOKEN_BY edge when they are
        # empty, so an empty field beside an edge renders the edge's answer — the same answer.
        # Measured: counting empty-beside-edge flagged 4,175 quotes on the validation run that every
        # reader shows identically. What IS a disagreement: a field naming someone the edge does
        # not, or a field naming someone with no edge at all.
        edge_name = person_name.get(edge) if edge else None
        if sid and sid != edge:
            out.append(
                f"{prefix}QUOTE_FIELDS_VS_EDGE {n.get('id')} speaker_id={sid!r} edge={edge!r}"
            )
        elif (
            sname
            and not _is_display_label(str(sname))
            and not (edge_name and same_person(str(sname), edge_name))
        ):
            out.append(
                f"{prefix}QUOTE_FIELDS_VS_EDGE {n.get('id')} speaker_name={sname!r} edge={edge!r}"
            )
    return out


def _sync_labels(
    segments: Any, adfree_segments: Any, placed_names: List[str], prefix: str
) -> List[str]:
    """Transcript speaker labels (the file `/segments` serves: raw first) and raw vs ad-free."""
    out: List[str] = []
    served = segments if segments is not None else adfree_segments
    for lab in _labels_of(served):
        if _is_display_label(lab) or _is_cue_initials(lab):
            continue
        if not any(same_person(lab, p) for p in placed_names):
            out.append(f"{prefix}LABEL_NOT_PLACED transcript names {lab!r}")
    if segments is not None and adfree_segments is not None:
        # NAMED labels only. The ad cutter removes an ad voice's lines, so an anonymous
        # `SPEAKER_00` present in raw and absent in ad-free is the cut working, not a disagreement
        # about who someone is.
        raw = {x for x in _labels_of(segments) if not is_bare_speaker_label(x)}
        ad = {x for x in _labels_of(adfree_segments) if not is_bare_speaker_label(x)}
        if raw != ad:
            out.append(
                f"{prefix}RAW_VS_ADFREE raw-only={sorted(raw - ad)} adfree-only={sorted(ad - raw)}"
            )
    return out


def _sync_diagnostics(diagnostics: Mapping[str, Any], placed: List[dict], prefix: str) -> List[str]:
    """The record against the roster's own diagnostics: same people named, same roles."""
    named = [
        v
        for v in (diagnostics.get("voices") or [])
        if isinstance(v, dict) and v.get("named") and v.get("resolved_name")
    ]
    out: List[str] = []
    for v in named:
        who = str(v["resolved_name"])
        match = [p for p in placed if same_person(who, str(p["name"]))]
        if not match:
            out.append(f"{prefix}RECORD_VS_DIAGNOSTICS diagnostics named {who!r}; not placed")
            continue
        diag_role, rec_role = str(v.get("role") or ""), str(match[0].get("role") or "")
        if diag_role in SPEAKER_ROLES and rec_role != diag_role:
            out.append(
                f"{prefix}RECORD_VS_DIAGNOSTICS {who!r} is {diag_role} in diagnostics, "
                f"{rec_role} in the record"
            )
    for p in placed:
        if not any(same_person(str(p["name"]), str(v["resolved_name"])) for v in named):
            out.append(
                f"{prefix}RECORD_VS_DIAGNOSTICS placed {p['name']!r} is not a named voice "
                "in diagnostics"
            )
    return out


def _sync_split_people(placed: List[dict], prefix: str) -> List[str]:
    """Two placed entries that are one human: a person diarization split and naming did not rejoin.

    Every other rule compares the record with a surface, so a split written CONSISTENTLY to all of
    them — `Elad` guest and `Elad Gil` host in the transcript, the diagnostics, the record and the
    graph — passes each one. This rule reads the record alone.
    """
    from ..providers.ml.diarization.roster import _same_person_on_one_episode

    out: List[str] = []
    names = [" ".join(str(p["name"]).split()) for p in placed]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if _same_person_on_one_episode(a, b):
                out.append(f"{prefix}SPLIT_PERSON {a!r} and {b!r} are placed as two people")
    return out


def _sync_context(context: Mapping[str, Any], placed: List[dict], prefix: str) -> List[str]:
    """``context.json`` ``basic.hosts`` / ``basic.guests`` against the record's placed people.

    The digest is a denormalised copy for readers that do not open ``metadata.json`` (the MCP
    tools). It is written from the record; this is what says it still is.
    """
    raw = context.get("basic")
    basic: Mapping[str, Any] = raw if isinstance(raw, dict) else {}
    out: List[str] = []
    for role in ("host", "guest"):
        want = sorted(str(p["name"]) for p in placed if p.get("role") == role)
        got = sorted(str(n) for n in (basic.get(f"{role}s") or []))
        if want != got:
            out.append(f"{prefix}CONTEXT_VS_RECORD {role}s: context={got} record={want}")
    return out


def check_episode_in_sync(
    metadata: Mapping[str, Any],
    kg: Mapping[str, Any],
    gi: Mapping[str, Any] | None,
    *,
    segments: Any = None,
    adfree_segments: Any = None,
    diagnostics: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
    legacy_as_placed: bool = False,
    label: str = "",
) -> List[str]:
    """Every disagreement between the speaker record and the surfaces written from it.

    Empty means the episode is in sync. Each violation starts with a stable code so a corpus-wide
    count can group them:

    * ``QUOTE_NOT_PLACED`` — a quote is credited (SPOKEN_BY) to someone no voice was matched to
    * ``QUOTE_FIELDS_VS_EDGE`` — a quote's own ``speaker_id`` disagrees with its SPOKEN_BY edge
    * ``CAST_NOT_PLACED`` — kg.json lists a host/guest no voice was matched to
    * ``UNPLACED_CAST`` — the record says ``placed: false``, yet kg.json gives them a speaking role
    * ``PLACED_NOT_CAST`` — a placed host/guest is missing from kg.json's speakers
    * ``LABEL_NOT_PLACED`` — the transcript names a speaker who is not a placed entry
    * ``RAW_VS_ADFREE`` — the raw and ad-free segments name different speakers
    * ``RECORD_VS_DIAGNOSTICS`` — the record and the roster's own diagnostics disagree on who was
      named, or in what role
    * ``SPLIT_PERSON`` — one human is placed as two entries (a diarization split never rejoined)
    * ``CONTEXT_VS_RECORD`` — ``context.json``'s hosts/guests are not the record's placed people
    * ``NO_GRAPH`` — the record places speakers and ``kg.json`` does not exist (re-derive needed,
      as distinct from a graph that exists and omits them, which is ``PLACED_NOT_CAST``)

    Names are compared with :func:`same_person`, so an ASR variant of a placed person is not a
    violation. An artifact with no record reports nothing unless ``legacy_as_placed`` is set; the
    diagnostics rule is skipped in that mode, because a pre-record roster has no flag to compare.
    """
    prefix = f"{label}: " if label else ""
    if not has_speaker_record(metadata) and not legacy_as_placed:
        return []
    placed = placed_people(metadata, legacy_as_placed=legacy_as_placed)
    placed_names = [str(p["name"]) for p in placed]
    unplaced = [
        str(s.get("name"))
        for s in ((metadata.get("content") or {}).get("speakers") or [])
        if isinstance(s, dict) and s.get("placed") is False and s.get("name")
    ]
    out = _sync_split_people(placed, prefix)
    # A MISSING graph is not a disagreeing graph. Comparing against an empty mapping reported every
    # placed person as `PLACED_NOT_CAST` — "the graph omits them" — when the truth is that no graph
    # was ever written. Both are worth reporting, but they are different repairs: one re-runs the
    # roster, the other re-derives the episode. Counting them under one code hides the second, and
    # once the reprocess stages began writing the record (#2075) it would have hidden it at scale.
    if kg:
        out += _sync_cast(kg, placed, unplaced, prefix)
    elif placed:
        out.append(
            f"{prefix}NO_GRAPH the record places {len(placed)} speaker(s) but kg.json is absent"
        )
    if gi:
        out += _sync_quotes(gi, placed_names, prefix)
    out += _sync_labels(segments, adfree_segments, placed_names, prefix)
    if diagnostics and not legacy_as_placed:
        out += _sync_diagnostics(diagnostics, placed, prefix)
    if context and not legacy_as_placed:
        out += _sync_context(context, placed, prefix)
    return out
