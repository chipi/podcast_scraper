"""The one-record sync check: every people surface agrees with the speaker record (#2075).

One test per violation code, each with the smallest episode that produces it, plus the exemptions
that keep the check from reporting benign cases: ASR spelling variants, cue initials, and
artifacts written before the record.
"""

from __future__ import annotations

from typing import Any, Dict, List

from podcast_scraper.kg.speaker_coherence import check_episode_in_sync

BARBARO = {
    "id": "host",
    "name": "Michael Barbaro",
    "role": "host",
    "placed": True,
    "voices": ["SPEAKER_02"],
}
MATINA = {
    "id": "guest",
    "name": "Matina Stevis-Gridneff",
    "role": "guest",
    "placed": True,
    "voices": ["SPEAKER_00"],
}
KITROEFF = {"id": "unplaced_1", "name": "Natalie Kitroeff", "role": "host", "placed": False}


def _meta(*speakers: Dict[str, Any]) -> Dict[str, Any]:
    return {"feed": {"title": "The Daily"}, "content": {"speakers": list(speakers)}}


def _kg(*cast: tuple) -> Dict[str, Any]:
    return {
        "nodes": [
            {"id": f"person:{i}", "type": "Person", "properties": {"name": n, "role": r}}
            for i, (n, r) in enumerate(cast)
        ]
    }


def _gi(credits: Dict[str, str], *, fields: Dict[str, Any] | None = None) -> Dict[str, Any]:
    people = sorted(set(credits.values()))
    pid = {p: f"person:{p.lower().replace(' ', '-')}" for p in people}
    nodes: List[Dict[str, Any]] = [
        {"id": pid[p], "type": "Person", "properties": {"name": p}} for p in people
    ]
    for q in credits:
        props = {"speaker_id": pid[credits[q]]}
        if fields and q in fields:
            props = {"speaker_id": fields[q]}
        nodes.append({"id": q, "type": "Quote", "properties": props})
    edges = [{"type": "SPOKEN_BY", "from": q, "to": pid[p]} for q, p in credits.items()]
    return {"nodes": nodes, "edges": edges}


def _seg(*labels: str) -> List[Dict[str, Any]]:
    return [{"speaker": f"SPEAKER_{i:02d}", "speaker_label": lab} for i, lab in enumerate(labels)]


DIAG_VOICES: List[Dict[str, Any]] = [
    {"voice": "SPEAKER_02", "resolved_name": "Michael Barbaro", "role": "host", "named": True},
    {
        "voice": "SPEAKER_00",
        "resolved_name": "Matina Stevis-Gridneff",
        "role": "guest",
        "named": True,
    },
    {"voice": "SPEAKER_01", "resolved_name": "SPEAKER_01", "role": "unknown", "named": False},
]

IN_SYNC = dict(
    metadata=_meta(BARBARO, MATINA, KITROEFF),
    kg=_kg(
        ("Michael Barbaro", "host"),
        ("Matina Stevis-Gridneff", "guest"),
        ("Natalie Kitroeff", "mentioned"),
    ),
    gi=_gi({"quote:1": "Matina Stevis-Gridneff"}),
    segments=_seg("Michael Barbaro", "Matina Stevis-Gridneff", "SPEAKER_01"),
    adfree_segments=_seg("Michael Barbaro", "Matina Stevis-Gridneff", "SPEAKER_01"),
    diagnostics={
        "voices": [
            {
                "voice": "SPEAKER_02",
                "resolved_name": "Michael Barbaro",
                "role": "host",
                "named": True,
            },
            {
                "voice": "SPEAKER_00",
                "resolved_name": "Matina Stevis-Gridneff",
                "role": "guest",
                "named": True,
            },
            {
                "voice": "SPEAKER_01",
                "resolved_name": "SPEAKER_01",
                "role": "unknown",
                "named": False,
            },
        ]
    },
)


def _run(**overrides: Any) -> List[str]:
    args = {**IN_SYNC, **overrides}
    return check_episode_in_sync(
        args["metadata"],
        args["kg"],
        args["gi"],
        segments=args["segments"],
        adfree_segments=args["adfree_segments"],
        diagnostics=args["diagnostics"],
        context=args.get("context"),
        legacy_as_placed=args.get("legacy_as_placed", False),
    )


def _codes(violations: List[str]) -> List[str]:
    return sorted(v.split(" ", 1)[0] for v in violations)


def test_a_consistent_episode_has_no_violations() -> None:
    assert _run() == []


class TestEachRule:
    def test_quote_not_placed(self) -> None:
        v = _run(gi=_gi({"quote:1": "Natalie Kitroeff"}))
        assert _codes(v) == ["QUOTE_NOT_PLACED"]

    def test_quote_fields_disagree_with_the_edge(self) -> None:
        v = _run(
            gi=_gi({"quote:1": "Matina Stevis-Gridneff"}, fields={"quote:1": "person:someone-else"})
        )
        assert _codes(v) == ["QUOTE_FIELDS_VS_EDGE"]

    def test_cast_not_placed(self) -> None:
        v = _run(
            kg=_kg(
                ("Michael Barbaro", "host"),
                ("Matina Stevis-Gridneff", "guest"),
                ("Garry Tan", "host"),
            )
        )
        assert _codes(v) == ["CAST_NOT_PLACED"]

    def test_an_unplaced_person_given_a_speaking_role(self) -> None:
        v = _run(
            kg=_kg(
                ("Michael Barbaro", "host"),
                ("Matina Stevis-Gridneff", "guest"),
                ("Natalie Kitroeff", "host"),
            )
        )
        assert _codes(v) == ["UNPLACED_CAST"]

    def test_placed_not_cast(self) -> None:
        v = _run(kg=_kg(("Michael Barbaro", "host")))
        assert _codes(v) == ["PLACED_NOT_CAST"]

    def test_label_not_placed(self) -> None:
        segs = _seg("Michael Barbaro", "Matina Stevis-Gridneff", "Rachel Abrams")
        v = _run(segments=segs, adfree_segments=segs)
        assert _codes(v) == ["LABEL_NOT_PLACED"]

    def test_raw_and_adfree_disagree(self) -> None:
        v = _run(adfree_segments=_seg("Michael Barbaro", "SPEAKER_00"))
        assert "RAW_VS_ADFREE" in _codes(v)

    def test_the_record_and_its_diagnostics_disagree_on_a_role(self) -> None:
        diag = {
            "voices": [
                {
                    "voice": "SPEAKER_02",
                    "resolved_name": "Michael Barbaro",
                    "role": "guest",
                    "named": True,
                },
                {
                    "voice": "SPEAKER_00",
                    "resolved_name": "Matina Stevis-Gridneff",
                    "role": "guest",
                    "named": True,
                },
            ]
        }
        assert _codes(_run(diagnostics=diag)) == ["RECORD_VS_DIAGNOSTICS"]

    def test_diagnostics_named_someone_the_record_did_not_place(self) -> None:
        diag = {
            "voices": list(DIAG_VOICES)
            + [
                {
                    "voice": "SPEAKER_05",
                    "resolved_name": "Rachel Abrams",
                    "role": "host",
                    "named": True,
                }
            ]
        }
        assert _codes(_run(diagnostics=diag)) == ["RECORD_VS_DIAGNOSTICS"]


class TestSplitPerson:
    """A split written consistently to every surface passes every comparison rule; this one reads
    the record alone (validation run: `Elad` guest / `Elad Gil` host on No Priors)."""

    def test_one_human_placed_twice_is_reported(self) -> None:
        """The surviving shape: one person, two ASR spellings of the same surname."""
        a = {
            "id": "guest",
            "name": "Elad Gilman",
            "role": "guest",
            "placed": True,
            "voices": ["S1"],
        }
        b = {
            "id": "host_2",
            "name": "Elad Gilmann",
            "role": "host",
            "placed": True,
            "voices": ["S3"],
        }
        v = check_episode_in_sync(_meta(BARBARO, a, b), _kg(("Michael Barbaro", "host")), None)
        assert any(x.startswith("SPLIT_PERSON") for x in v), v

    def test_a_mononym_and_a_full_name_are_no_longer_a_split(self) -> None:
        """A DELIBERATE gap, not an oversight (#2075). `Elad` + `Elad Gil` on No Priors really was
        one person, but the same rule made `Alex` and `Alex Maasi` one person on Made In Africa,
        where they are two humans in the room — and merging two people is the worse error under
        #876. The mononym clause was removed, so this split is no longer reported here.
        """
        elad = {"id": "guest", "name": "Elad", "role": "guest", "placed": True, "voices": ["S1"]}
        gil = {"id": "host_2", "name": "Elad Gil", "role": "host", "placed": True, "voices": ["S3"]}
        v = check_episode_in_sync(_meta(BARBARO, elad, gil), _kg(("Michael Barbaro", "host")), None)
        assert not any(x.startswith("SPLIT_PERSON") for x in v), v

    def test_two_people_sharing_a_surname_are_not_a_split(self) -> None:
        a = {"id": "guest_1", "name": "Robert Pape", "role": "guest", "placed": True}
        b = {"id": "guest_2", "name": "Karen Pape", "role": "guest", "placed": True}
        v = check_episode_in_sync(
            _meta(a, b), _kg(("Robert Pape", "guest"), ("Karen Pape", "guest")), None
        )
        assert not any(x.startswith("SPLIT_PERSON") for x in v), v


class TestMissingGraph:
    """A graph that does not exist is a different repair from a graph that omits the speakers."""

    def test_no_graph_at_all_is_not_reported_as_the_graph_omitting_them(self) -> None:
        v = _run(kg={})
        assert _codes(v) == ["NO_GRAPH"], v

    def test_a_graph_that_exists_but_omits_a_placed_speaker_is_still_placed_not_cast(self) -> None:
        v = _run(kg=_kg(("Michael Barbaro", "host")))
        assert "PLACED_NOT_CAST" in _codes(v), v
        assert "NO_GRAPH" not in _codes(v), v

    def test_no_graph_and_nobody_placed_reports_nothing(self) -> None:
        """Nothing was placed, so there is no claim for a missing graph to contradict."""
        v = _run(
            kg={},
            metadata=_meta(KITROEFF),
            gi=None,
            segments=None,
            adfree_segments=None,
            diagnostics=None,
        )
        assert v == [], v


class TestContextDigest:
    def test_context_hosts_that_are_not_the_records_are_reported(self) -> None:
        ctx = {"basic": {"hosts": ["Michael Barbaro", "Natalie Kitroeff"], "guests": []}}
        v = _run(context=ctx)
        assert _codes(v) == ["CONTEXT_VS_RECORD", "CONTEXT_VS_RECORD"], v

    def test_a_context_written_from_the_record_is_in_sync(self) -> None:
        ctx = {"basic": {"hosts": ["Michael Barbaro"], "guests": ["Matina Stevis-Gridneff"]}}
        assert _run(context=ctx) == []


class TestWhatIsNotAViolation:
    def test_an_asr_spelling_variant_of_a_placed_person(self) -> None:
        v = _run(kg=_kg(("Michael Barbaro", "host"), ("Matina Stevis Gridneff", "guest")))
        assert v == []

    def test_publisher_cue_initials_on_the_transcript(self) -> None:
        segs = _seg("Michael Barbaro", "Matina Stevis-Gridneff", "MG")
        assert _run(segments=segs, adfree_segments=segs) == []

    def test_a_pre_record_artifact_is_skipped(self) -> None:
        legacy = {"content": {"speakers": [{"id": "host", "name": "Garry Tan", "role": "host"}]}}
        assert _run(metadata=legacy) == []

    def test_but_can_be_audited_as_placed(self) -> None:
        legacy = {
            "content": {"speakers": [{"id": "host", "name": "Michael Barbaro", "role": "host"}]}
        }
        v = _run(metadata=legacy, legacy_as_placed=True)
        assert "QUOTE_NOT_PLACED" in _codes(v)  # the guest was never in this old roster


class TestQuoteFieldsOnlyDisagreeWhenTheySaySomething:
    def test_an_empty_speaker_id_beside_an_edge_is_not_a_violation(self) -> None:
        """The insights view falls back to the edge when the quote's own field is empty."""
        v = _run(gi=_gi({"quote:1": "Matina Stevis-Gridneff"}, fields={"quote:1": None}))
        assert v == []

    def test_a_speaker_id_with_no_edge_is_a_violation(self) -> None:
        gi = {
            "nodes": [
                {
                    "id": "quote:1",
                    "type": "Quote",
                    "properties": {"speaker_id": "person:michael-barbaro"},
                }
            ],
            "edges": [],
        }
        assert _codes(_run(gi=gi)) == ["QUOTE_FIELDS_VS_EDGE"]


def test_an_anonymous_ad_voice_cut_from_the_adfree_file_is_not_a_violation() -> None:
    raw = _seg("Michael Barbaro", "Matina Stevis-Gridneff", "SPEAKER_01")
    adfree = _seg("Michael Barbaro", "Matina Stevis-Gridneff")
    assert _run(segments=raw, adfree_segments=adfree) == []


def test_a_display_label_on_an_unattributed_quote_is_not_a_person() -> None:
    """ "Unidentified speaker" is what a surface shows for an unnamed voice; it names nobody."""
    gi = {
        "nodes": [
            {
                "id": "quote:9",
                "type": "Quote",
                "properties": {"speaker_name": "Unidentified speaker"},
            }
        ],
        "edges": [],
    }
    assert _run(gi=gi) == []
