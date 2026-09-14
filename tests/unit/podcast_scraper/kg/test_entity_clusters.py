"""Unit tests for cross-episode entity canonicalization (kg/entity_clusters.py, #852)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.kg.entity_clusters import (
    _are_xep_variants,
    build_entity_canonical_map,
    build_entity_id_map,
    collect_entity_candidates,
    EntityCandidate,
    id_map_from_clusters_payload,
)

pytestmark = pytest.mark.unit


def _cand(cid, kind, name, eps, shows):
    return EntityCandidate(id=cid, kind=kind, name=name, episodes=set(eps), shows=set(shows))


# --- variant rule: MERGE real drift ---------------------------------------------


@pytest.mark.parametrize(
    "a,b,kind",
    [
        ("Cargil", "Cargill", "org"),
        ("Data Bricks", "Databricks", "org"),
        ("Chat GPT", "ChatGPT", "org"),
        ("Byrne Hobart", "Burne Hobart", "person"),
        ("Tracy Alloway", "Tracey Alloway", "person"),
        ("Donald Mackenzie", "Donald McKenzie", "person"),
        ("David Shor", "David Shore", "person"),
    ],
)
def test_xep_variants_merge_real_drift(a, b, kind):
    assert _are_xep_variants(a, b, kind) is True


# --- variant rule: REJECT the landmines -----------------------------------------


@pytest.mark.parametrize(
    "a,b,kind",
    [
        ("UPS", "USPS", "org"),  # acronyms
        ("Claude", "Claude 3", "org"),  # version token
        ("GPT-4", "GPT-4o", "org"),  # version-ish
        ("Bloomberg Audio Studios", "Bloomberg Media Studios", "org"),  # distinct content word
        ("Sam Altman", "Tim Cook", "person"),  # different people
        ("John Smith", "Jane Smith", "person"),  # different first names
    ],
)
def test_xep_variants_reject_landmines(a, b, kind):
    assert _are_xep_variants(a, b, kind) is False


# --- #904 predicate redesign: nickname + token-count tolerance ------------------


@pytest.mark.parametrize(
    "a,b",
    [
        # Nickname class (same surname, nickname/full-name first)
        ("Mike Selig", "Michael Selig"),
        ("Nicholas Snyder", "Nick Snyder"),
        ("Elizabeth Reid", "Liz Reid"),
        ("Emmanuel Roman", "Manny Roman"),
        ("Rich Clarida", "Richard Clarida"),
        ("Rob Goldstein", "Robert Goldstein"),
        # Initial-vs-full first name (J. → Jerome)
        ("J. Powell", "Jerome Powell"),
        # Title prefix on one side only (Dr / Ayatollah / President)
        ("Dr. Elena Fischer", "Elena Fischer"),
        ("Ayatollah Ali Khamenei", "Ali Khamenei"),
        ("President Trump", "Donald Trump"),
        # Family-only reference (last-token match)
        ("Mark Carney", "Carney"),
        ("Donald Trump", "Trump"),
    ],
)
def test_xep_variants_904_nickname_and_token_count_merge(a, b):
    """#904 — predicate redesign covers nickname class + token-count mismatches."""
    assert _are_xep_variants(a, b, "person") is True


@pytest.mark.parametrize(
    "a,b,kind",
    [
        # Two distinct people sharing first name — predicate must NOT merge.
        # `Marco` (alone) vs `Marco Bianchi` is the v2 two-Marcos test:
        # bare `Marco` (p03 wreck diver) vs the surname-disambiguated
        # `Marco Bianchi` (p05 tax-loss researcher). The predicate
        # deliberately does NOT first-name-merge because it can't tell ASR
        # aliases (`Liam` ↔ `Liam Verbeek`, same person) from organic
        # same-first-name pairs. Differentiating needs external signal.
        ("Marco", "Marco Bianchi", "person"),
        ("Jacob Goldstein", "Rob Goldstein", "person"),
        # Family-only reference must NOT cross-merge people sharing a last name
        ("Mark Carney", "John Carney", "person"),
        # Org-side token-count tolerance is intentionally disabled — guards
        # against `Adobe` ↔ `Adobe Creative Cloud` (sub-product, not alias).
        ("Adobe", "Adobe Creative Cloud", "org"),
    ],
)
def test_xep_variants_904_predicate_does_not_overmerge(a, b, kind):
    assert _are_xep_variants(a, b, kind) is False


@pytest.mark.parametrize(
    "a,b",
    [
        # First-name-only alias — currently does NOT merge by design (see the
        # NOTE in `_token_count_tolerant_match`). When a future LLM-tier
        # escalation or same-show evidence-based merge ships, this test
        # should flip to expecting True.
        ("Liam Verbeek", "Liam"),
    ],
)
def test_xep_variants_904_first_name_only_alias_deferred(a, b):
    """First-name-only-alias merge deferred — same predicate shape as the
    two-Marcos distinct-people case; can't disambiguate without external
    signal. Tracked for follow-up (#906 / #921)."""
    assert _are_xep_variants(a, b, "person") is False


# --- canonical map: frequency + same-show ---------------------------------------


def test_canonical_is_highest_frequency():
    cands = {
        "org:cargill": _cand("org:cargill", "org", "Cargill", ["e1", "e2"], ["showA"]),
        "org:cargil": _cand("org:cargil", "org", "Cargil", ["e3"], ["showA"]),
    }
    payload, id_map = build_entity_canonical_map(cands)
    # Lower-frequency variant maps to the higher-frequency canonical.
    assert id_map == {"org:cargil": "org:cargill"}
    assert payload["merged_variants"] == 1
    assert payload["clusters"][0]["canonical_id"] == "org:cargill"


def test_same_show_required_blocks_cross_show_merge():
    cands = {
        "org:cargill": _cand("org:cargill", "org", "Cargill", ["e1", "e2"], ["showA"]),
        "org:cargil": _cand("org:cargil", "org", "Cargil", ["e3"], ["showB"]),  # other show
    }
    _, id_map = build_entity_canonical_map(cands, same_show_required=True)
    assert id_map == {}


def test_landmines_not_merged_in_map():
    cands = {
        "org:claude": _cand("org:claude", "org", "Claude", ["e1", "e2"], ["showA"]),
        "org:claude-3": _cand("org:claude-3", "org", "Claude 3", ["e1"], ["showA"]),
        "org:ups": _cand("org:ups", "org", "UPS", ["e1"], ["showA"]),
        "org:usps": _cand("org:usps", "org", "USPS", ["e1"], ["showA"]),
    }
    _, id_map = build_entity_canonical_map(cands)
    assert id_map == {}


def test_kind_aware_no_cross_kind_merge():
    cands = {
        "person:cargill": _cand("person:cargill", "person", "Cargill", ["e1"], ["showA"]),
        "org:cargil": _cand("org:cargil", "org", "Cargil", ["e1"], ["showA"]),
    }
    _, id_map = build_entity_canonical_map(cands)
    assert id_map == {}


# --- collect + end-to-end on a synthetic corpus ---------------------------------


def _write_kg(path: Path, episode_id: str, show: str, entities):
    nodes = [{"id": f"episode:{episode_id}", "type": "Episode", "properties": {"podcast_id": show}}]
    for eid, name in entities:
        nodes.append({"id": eid, "type": "Entity", "properties": {"name": name}})
    path.write_text(json.dumps({"episode_id": episode_id, "nodes": nodes, "edges": []}))


def test_collect_and_build_id_map_from_corpus(tmp_path):
    # Same show, two episodes: Cargill (ep1, ep2) + Cargil (ep3) → collapse.
    _write_kg(tmp_path / "e1.kg.json", "e1", "showA", [("org:cargill", "Cargill")])
    _write_kg(tmp_path / "e2.kg.json", "e2", "showA", [("org:cargill", "Cargill")])
    _write_kg(tmp_path / "e3.kg.json", "e3", "showA", [("org:cargil", "Cargil")])

    cands = collect_entity_candidates(tmp_path)
    assert cands["org:cargill"].freq == 2
    assert cands["org:cargil"].freq == 1
    assert cands["org:cargill"].shows == {"showA"}

    id_map = build_entity_id_map(tmp_path)
    assert id_map == {"org:cargil": "org:cargill"}


def test_id_map_from_payload_roundtrip():
    cands = {
        "org:cargill": _cand("org:cargill", "org", "Cargill", ["e1", "e2"], ["showA"]),
        "org:cargil": _cand("org:cargil", "org", "Cargil", ["e3"], ["showA"]),
    }
    payload, id_map = build_entity_canonical_map(cands)
    assert id_map_from_clusters_payload(payload) == id_map


class TestDiacriticsAreFoldedBeforeComparing:
    """``Björk`` and ``Bjork`` are one person (#2056).

    Transcripts and feed metadata disagree about diacritics constantly — ASR emits unaccented
    ASCII, the show notes carry the real spelling. Multi-token names already survived this by
    accident, because one differing character barely moves the similarity ratio:

        'Jürgen Schmidhuber' == 'Jurgen Schmidhuber'   -> already True (fuzzy)

    A SINGLE-token name does not, because `_is_acronymish` refuses to fuzzy-match short single
    tokens at all — the UPS/USPS guard. So the mononym case fell through the one gap where the
    ratio test cannot rescue it.

    Folding also makes the multi-token cases match EXACTLY rather than by ratio, which is a
    precision gain, not just a recall one: an exact hit short-circuits before any threshold is
    consulted.
    """

    def test_a_mononym_with_a_diacritic_matches_its_ascii_spelling(self) -> None:
        assert _are_xep_variants("Björk", "Bjork", "person") is True

    @pytest.mark.parametrize(
        "accented,plain",
        [
            ("Łukasz Kaiser", "Lukasz Kaiser"),
            ("Zoë Kravitz", "Zoe Kravitz"),
            ("Jürgen Schmidhuber", "Jurgen Schmidhuber"),
            ("François Chollet", "Francois Chollet"),
            ("Søren Kierkegaard", "Soren Kierkegaard"),
            ("Renée DiResta", "Renee DiResta"),
        ],
    )
    def test_accented_and_plain_spellings_are_one_entity(self, accented, plain) -> None:
        assert _are_xep_variants(accented, plain, "person") is True

    def test_folding_is_exact_not_fuzzy(self) -> None:
        from podcast_scraper.kg.filters import _clean_entity_name

        assert _clean_entity_name("François Chollet") == _clean_entity_name("Francois Chollet")

    def test_folding_does_not_collapse_distinct_people(self) -> None:
        # Stripping accents must not make two different humans equal.
        assert _are_xep_variants("Renée DiResta", "Renata DiResta", "person") is False
        assert _are_xep_variants("Kaiser Guo", "Kaiser Wilhelm II", "person") is False


class TestSameShowRequiredIsLoadBearing:
    """``same_show_required=True`` is the main thing preventing FALSE merges (#2056).

    #2056 lists "two spellings that never co-occur in one show are never compared" as a candidate
    explanation for duplicates surviving, which invites relaxing the gate. Measuring first says
    do NOT. Over 287 production artifacts the cross-episode matcher produced 68 variant pairs, 32
    of which are only held apart by this gate — and they include:

        'Albert Einstein'  == 'Robert Jensen'
        'Alex Bregman'     == 'Lex Friedman'
        'Charles I'        == 'Charles II'
        'Dana Schutz'      == 'Dean Schwartz'
        'Kevin Kelly'      == 'Melvin Key'

    Dropping the gate merges every one of those. A false split is clutter; a false merge
    reassigns one person's statements to another. This test exists so the gate cannot be quietly
    relaxed to "fix duplicates" without confronting that list.

    It also pins that the gate is NOT sufficient — ``Albert Einstein``/``Bert Vogelstein`` share
    a show in the sample and merge today. Fixing that needs matcher PRECISION (an authority list
    or an adjudicator), not a looser gate.
    """

    @staticmethod
    def _candidates(rows):
        from podcast_scraper.kg.entity_clusters import EntityCandidate

        out = {}
        for pid, name, shows in rows:
            out[pid] = EntityCandidate(
                id=pid, kind="person", name=name, episodes={f"ep-{pid}"}, shows=set(shows)
            )
        return out

    def test_different_people_on_different_shows_are_not_merged(self) -> None:
        from podcast_scraper.kg.entity_clusters import build_entity_canonical_map

        cands = self._candidates(
            [
                ("person:albert-einstein", "Albert Einstein", ["Show A"]),
                ("person:robert-jensen", "Robert Jensen", ["Show B"]),
            ]
        )
        _payload, id_map = build_entity_canonical_map(cands, same_show_required=True)
        assert id_map == {}, "the gate is the only thing holding these apart"

    def test_relaxing_the_gate_merges_people_who_never_shared_a_show(self) -> None:
        # Documents the COST of relaxing it. Note the one-token rule (see
        # `TestAPersonMayDifferInOnlyONEToken`) now stops the WORST of these on its own —
        # `Albert Einstein`/`Bert Vogelstein` no longer merges at all. What the gate still buys
        # is everything that drifts in a single token but belongs to two different humans on two
        # different shows, which no name comparison can tell apart.
        from podcast_scraper.kg.entity_clusters import build_entity_canonical_map

        cands = self._candidates(
            [
                ("person:richard-mccoll", "Richard McColl", ["Show A"]),
                ("person:richard-mccollough", "Richard McCollough", ["Show B"]),
            ]
        )
        _payload, id_map = build_entity_canonical_map(cands, same_show_required=False)
        assert id_map, "without the gate, two one-token-apart strangers merge across shows"

    def test_a_real_variant_pair_on_one_show_still_merges(self) -> None:
        # The gate must not be so strict that genuine variants stop collapsing.
        from podcast_scraper.kg.entity_clusters import build_entity_canonical_map

        cands = self._candidates(
            [
                ("person:bernard-leong", "Bernard Leong", ["Analyse Asia"]),
                ("person:bernard-leung", "Bernard Leung", ["Analyse Asia"]),
            ]
        )
        _payload, id_map = build_entity_canonical_map(cands, same_show_required=True)
        assert id_map, "a real spelling variant within one show must still collapse"


class TestAPersonMayDifferInOnlyONEToken:
    """Two tokens both drifting means two different humans (#2056 precision).

    Measured over 287 production artifacts. Every clearly-wrong same-show merge the matcher makes
    differs in BOTH name tokens; almost every correct one differs in exactly one:

        FALSE   'Albert Einstein' == 'Bert Vogelstein'    2 tokens differ
        FALSE   'Jensen Huang'    == 'Jesse Zhang'        2 tokens differ
        FALSE   'Li Lun'          == 'Lily Liu'           2 tokens differ

        TRUE    'Bernard Leong'   == 'Bernard Leung'      1
        TRUE    'Stewart Brand'   == 'Stuart Brand'       1
        TRUE    'Joe Weisenthal'  == 'Joe Wiesenthal'     1
        TRUE    'Nikolai Kononov' == 'Nikolay Kononov'    1

    A transcription or typo error lands on ONE token. Two independent drifts in a two-token name
    is not one person spelled badly, it is two people who rhyme.

    THE COST, stated rather than hidden: ``'Alexander Carpi' == 'Alexandra Karppi'`` is a real
    person whose name drifted in both tokens, and this rule stops merging them. That is the trade
    taken deliberately — a false split is visible clutter, a false merge reassigns one person's
    statements to another. 3 false merges prevented for 1 real merge lost.

    Applies to PEOPLE with equal token counts. Orgs legitimately differ in more than one token
    ("Bank of England" / "Bank of Britain" is a different question) and are untouched.
    """

    @pytest.mark.parametrize(
        "a,b",
        [
            ("Albert Einstein", "Bert Vogelstein"),
            ("Jensen Huang", "Jesse Zhang"),
            ("Li Lun", "Lily Liu"),
        ],
    )
    def test_two_drifting_tokens_are_two_people(self, a: str, b: str) -> None:
        assert _are_xep_variants(a, b, "person") is False

    @pytest.mark.parametrize(
        "a,b",
        [
            ("Bernard Leong", "Bernard Leung"),
            ("Stewart Brand", "Stuart Brand"),
            ("Corey Combs", "Cory Combs"),
            ("Joe Weisenthal", "Joe Wiesenthal"),
            ("Adam Reichardt", "Adam Reichert"),
            ("Nikolai Kononov", "Nikolay Kononov"),
            ("Mark Galeotti", "Mark Galliotti"),
            ("Kevin Warsh", "Kevin Worsch"),
            ("Francois Chollet", "Francois Jollet"),
        ],
    )
    def test_one_drifting_token_is_still_one_person(self, a: str, b: str) -> None:
        assert _are_xep_variants(a, b, "person") is True

    def test_the_cost_is_recorded_not_hidden(self) -> None:
        # A real person this rule now refuses to merge. Pinned so the trade stays visible: if a
        # later change makes this True again, it must also keep the three tests above False.
        assert _are_xep_variants("Alexander Carpi", "Alexandra Karppi", "person") is False

    def test_orgs_are_not_subject_to_the_rule(self) -> None:
        # The one-token rule is about human names; org names are compositional.
        assert _are_xep_variants("Data Bricks", "Databricks", "org") is True


class TestRegnalNumeralsDistinguishPeople:
    """`Charles I` and `Charles II` are two monarchs (#2065 guardrail matrix).

    `_VERSION_TOKEN_RE` is `re.compile(r"\\d")` — ARABIC digits only. Roman numerals are the
    standard way regnal names are distinguished, and they passed straight through to the ratio
    test, which merged them. Found by the guardrail matrix on its first run; *The Rest Is History*
    is in the production corpus.

    THE RULE IS DELIBERATELY NARROW: both differing tokens must be well-formed roman numerals AND
    in the LAST position. A bare roman-numeral test is unsafe here — `li` is a valid numeral and a
    very common Chinese surname, and the corpus carries Round Table China, China Plus and
    ChinaTalk. `Li` sits in first position in those names, so the last-token scope keeps it out of
    reach.
    """

    @pytest.mark.parametrize(
        "a,b",
        [
            ("Charles I", "Charles II"),
            ("Elizabeth I", "Elizabeth II"),
            ("Henry VII", "Henry VIII"),
            ("Louis XIV", "Louis XVI"),
            ("Kaiser Wilhelm I", "Kaiser Wilhelm II"),
        ],
    )
    def test_different_regnal_numbers_are_different_people(self, a: str, b: str) -> None:
        assert _are_xep_variants(a, b, "person") is False

    def test_the_same_monarch_still_matches(self) -> None:
        assert _are_xep_variants("Charles II", "Charles II", "person") is True

    @pytest.mark.parametrize(
        "a,b",
        [
            ("Li Luan", "Li Lun"),
            ("Bernard Leong", "Bernard Leung"),
            ("Stewart Brand", "Stuart Brand"),
        ],
    )
    def test_ordinary_names_are_unaffected(self, a: str, b: str) -> None:
        # `li` is a valid roman numeral; the last-token scope is what keeps these safe.
        assert _are_xep_variants(a, b, "person") is True
