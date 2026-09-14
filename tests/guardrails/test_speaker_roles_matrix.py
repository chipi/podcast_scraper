"""The host / guest / contributor case matrix — every row from a defect that actually shipped.

WHY THIS FILE EXISTS. Ten root causes sat behind one operator report ("there's always the same name
listed on all insights", "we never actually show who is the guest"), and the suite was green
throughout. #2058 named the reason: the test pyramid asserted the bug. Each layer's tests proved
its own function behaved as written, and nothing asserted that the finished ARTIFACTS agreed with
each other — which is where every one of the ten lived.

THE TWO RULES THIS FILE IS HELD TO:

1. **Every row names a real incident.** A case invented by imagining what might go wrong does not
   go in. The imagined ones are the ones that turned out to be wrong; the real ones are already
   written down in the commit history of #2065 / #2056.
2. **A capability check is only worth what its removal proof shows.** ``test_guard_capability.py``
   re-runs checks against deliberately disabled guards: a negative test that still passes with its
   guard deleted is decoration, and this arc produced four of those.

   BE PRECISE ABOUT COVERAGE, because overstating it here would be the same failure this file
   exists to catch. Removal proofs exist for **four** properties: the show-name guard needs its
   title, the voice guard must read the GI layer, the voice count must come from the sidecar, and
   the display-name decision must reach both artifacts. They do **not** exist for the one-token
   rule, the regnal-numeral guard, ``rewrite_ids`` role precedence, the exactly-one-speaker rule,
   the provenance validator, or the cache-invalidation token — those are covered by ordinary
   positive and negative cases only, which do not prove the guard is connected.

THE CLASSES, and the incident behind each:

    C1  the wrong SOURCE wins        the graph was handed the pre-diarization hint, not the roster
    C2  a NON-PERSON holds a role    show name / role word / org / mangled multi-name string
    C3  IDENTITY splits or collapses one human two ids; two humans one id
    C4  ATTRIBUTION drifts           sticky speaker across an unrecognised marker
    C5  the GUARD CANNOT SEE         a check unable to observe what it claims — separate file

C5 is the one with no prior coverage and the one every serious miss in this session belonged to.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List, Tuple

import pytest

from podcast_scraper.kg.entity_clusters import _are_xep_variants
from podcast_scraper.kg.speaker_coherence import same_person
from podcast_scraper.workflow.metadata_generation import _speaker_lists_for_graph

pytestmark = pytest.mark.unit


def _roster(*pairs: Tuple[str, str]) -> List[SimpleNamespace]:
    """``content.speakers`` as the objects `_speaker_lists_for_graph` actually reads.

    It uses ``getattr(sp, "name")``, so a dict silently yields "" for every field and every
    assertion built on one compares nothing. That mistake was made twice in one session; this
    helper exists so it cannot be made a third time here.
    """
    return [SimpleNamespace(name=n, role=r) for n, r in pairs]


# ---------------------------------------------------------------------------------------------
# C2 — a non-person must not hold a speaking role
# ---------------------------------------------------------------------------------------------

#: ``(name, feed_title, should_be_published, incident)``
NON_PERSON_CASES = [
    ("Africa Tech Summit", "Africa Tech Summit Podcast", False, "#2064: 19 eps, show as host"),
    ("Machine Learning Street", "Machine Learning Street Talk (MLST)", False, "#2064"),
    ("Conversations with Tyler", "Conversations with Tyler", False, "#2064 exact-title match"),
    ("Latent.Space", "Latent Space: The AI Engineer Podcast", False, "#2064 punctuation variant"),
    ("Trivium China", "Trivium China Podcast", False, "#2064 two-token org"),
    # Positive controls — real people who MUST survive every guard above.
    ("Kevin Roose", "Hard Fork", True, "control: host whose name is not in the title"),
    ("Elad Gil", "No Priors: Artificial Intelligence", True, "control"),
    (
        "Mai-Lan Tomsen Bukovec",
        "The Pragmatic Engineer",
        True,
        "control: the guest m0009 nearly lost",
    ),
]


@pytest.mark.parametrize("name,feed,published,incident", NON_PERSON_CASES)
def test_only_people_reach_the_graph_as_speakers(
    name: str, feed: str, published: bool, incident: str
) -> None:
    hosts, guests = _speaker_lists_for_graph(_roster((name, "host")), [], [], feed_title=feed)
    got = name in (hosts + guests)
    assert got is published, f"{name!r} on {feed!r} — {incident}"


def test_the_eponymous_host_false_positive_is_recorded_not_hidden() -> None:
    """A host whose name LEADS their own show reads as the show (#2064, advisor S1).

    `names_the_show` matches a title PREFIX, and the title genuinely cannot separate
    'Lex Fridman Podcast' from 'Latent Space: The AI Engineer Podcast' — that is world knowledge.
    Zero occurrences across the 55 production feeds today, so this is pinned as a KNOWN failure
    rather than guarded: m0009 reports these as `suspect_demotions` for a human to read.

    If this ever starts passing, the predicate improved and the suspect-demotion reporting can be
    reconsidered. Until then it must not silently change.
    """
    hosts, _g = _speaker_lists_for_graph(
        _roster(("Lex Fridman", "host")), [], [], feed_title="Lex Fridman Podcast"
    )
    assert "Lex Fridman" not in hosts, "known false positive — see suspect_demotions"


# ---------------------------------------------------------------------------------------------
# C1 — the roster beats the hint, and an empty roster falls back
# ---------------------------------------------------------------------------------------------


def test_the_roster_is_the_only_source_when_it_heard_the_episode() -> None:
    """#2065's headline: the graph was handed `detected_hosts`, the PRE-DIARIZATION guess.

    Measured before the fix: Person roles were mentioned 89.5% / host 9.9% / guest 0.6%, while the
    roster had named a guest on 66.9% of episodes and 93.2% of those never reached kg.json.
    """
    hosts, guests = _speaker_lists_for_graph(
        _roster(("Real Host", "host"), ("Real Guest", "guest")),
        ["Hint Host"],
        ["Hint Guest"],
    )
    assert hosts == ["Real Host"] and guests == ["Real Guest"]
    assert "Hint Host" not in hosts, "the hint must not SUPPLEMENT a roster that heard the episode"


def test_the_hint_is_used_only_when_there_is_no_roster_at_all() -> None:
    hosts, guests = _speaker_lists_for_graph([], ["Hint Host"], ["Hint Guest"])
    assert hosts == ["Hint Host"] and guests == ["Hint Guest"]


# ---------------------------------------------------------------------------------------------
# C3 — identity: one human must not split, two humans must not collapse
# ---------------------------------------------------------------------------------------------

#: ``(a, b, same_human, incident)`` — every pair observed in production artifacts.
IDENTITY_CASES = [
    ("Bernard Leong", "Bernard Leung", True, "#2062: m0009 stripped the real host's role"),
    ("Andrej Karpathy", "Andrei Karpathy", True, "#2056 intra-episode"),
    ("Stewart Brand", "Stuart Brand", True, "#2056"),
    ("Teresa Bejan", "Theresa Bejan", True, "#2056"),
    ("Joe Weisenthal", "Joe Wiesenthal", True, "#2056 same-show variant"),
    ("Łukasz Kaiser", "Lukasz Kaiser", True, "#2056 diacritic"),
    ("Björk", "Bjork", True, "#2056 single-token diacritic — the acronym guard's blind spot"),
    # Must NOT merge. Each of these the matcher accepted at some point.
    ("Albert Einstein", "Robert Jensen", False, "#2067: held apart only by same_show_required"),
    ("Alex Bregman", "Lex Friedman", False, "#2067"),
    ("Charles I", "Charles II", False, "#2067: different monarchs"),
    ("Albert Einstein", "Bert Vogelstein", False, "#2067: shared a show, merged before the fix"),
    ("Jensen Huang", "Jesse Zhang", False, "#2056 one-token rule"),
    ("Kaiser Guo", "Kaiser Wilhelm II", False, "#2056: a podcaster and an emperor"),
]


@pytest.mark.parametrize("a,b,same,incident", IDENTITY_CASES)
def test_the_variant_matcher_agrees_with_reality(a: str, b: str, same: bool, incident: str) -> None:
    assert _are_xep_variants(a, b, "person") is same, f"{a!r} vs {b!r} — {incident}"


def test_the_one_token_rule_costs_a_real_pair_and_that_is_deliberate() -> None:
    """`Alexander Carpi` / `Alexandra Karppi` is one human whose name drifted in BOTH tokens.

    The rule that stops `Jensen Huang` becoming `Jesse Zhang` also stops this. Accepted knowingly:
    a false split is visible clutter, a false merge reassigns one person's statements to another.
    Pinned so the trade stays a decision rather than a surprise.
    """
    assert _are_xep_variants("Alexander Carpi", "Alexandra Karppi", "person") is False


def test_same_person_matches_a_name_to_itself_in_any_script() -> None:
    """ASCII-only folding made a CJK speaker unable to match THEMSELVES (advisor S2).

    The fold ran `[^a-z0-9]+` after accent-stripping, which deletes every character of an entirely
    non-Latin name — so the speaker was "never spoke" on every coherence check and unmatchable by
    the migration. Production carries Round Table China, China Plus, ChinaTalk, The Naked Pravda.
    """
    for name in ("张川红", "Владимир Путин", "محمد صلاح", "김정은"):
        assert same_person(name, name) is True, name
    assert same_person("张川红", "李明") is False


# ---------------------------------------------------------------------------------------------
# C4 — attribution must not drift across an unrecognised marker
# ---------------------------------------------------------------------------------------------


def test_an_unrecognised_marker_ends_the_turn_rather_than_extending_the_last_speaker() -> None:
    """#2062: a raw `SPEAKER_NN:` beside real names let the previous speaker absorb it.

    63.1% of production episodes carry mixed markers — a real `Name:` next to an unresolved
    `SPEAKER_NN:`. The old builders emitted boundaries only for RECOGNISED names, so
    `speaker_for_char` returned the last surviving marker and every unrecognised turn was
    attributed to whoever spoke last. That is literally the operator's report: "there's always the
    same name listed on all insights".

    The fix is that both builders now emit a `None`-named boundary for a marker they cannot
    resolve, which ENDS the turn instead of handing it over.
    """
    from podcast_scraper.gi.speakers import build_unverified_named_turns, speaker_for_char

    transcript = "Kevin Roose: first thing.\nSPEAKER_02: second thing.\nKevin Roose: third thing."
    turns = build_unverified_named_turns(transcript)

    assert speaker_for_char(transcript.index("first thing"), turns) == "Kevin Roose"
    assert (
        speaker_for_char(transcript.index("second thing"), turns) is None
    ), "the unrecognised turn must be UNATTRIBUTED, not inherited from the previous speaker"
    assert speaker_for_char(transcript.index("third thing"), turns) == "Kevin Roose"

    assert (26, None) in turns, "the None boundary is the mechanism; pin it so it cannot be dropped"


def test_prose_that_looks_like_a_marker_does_not_become_a_speaker() -> None:
    """The cost of the fix, recorded: any line-start `Word:` now ends a turn.

    `Note:` and `Q:` in prose are treated as markers. That is under-attribution by design — an
    unattributed quote is recoverable, a MIS-attributed one is not — but it is a real cost and is
    pinned here so it stays a decision rather than a surprise.
    """
    from podcast_scraper.gi.speakers import build_unverified_named_turns

    turns = build_unverified_named_turns("Kevin Roose: a point.\nNote: an aside.")
    named = [name for _pos, name in turns if name]
    assert "Note" not in named, "prose labels must not be published as speakers"
