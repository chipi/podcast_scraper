"""Translation provenance on every claim (S2.11).

WHY THIS CANNOT BE ADDED LATER. In v1 there is no user-visible "translated from X" marker
(D-36), so a listener cannot tell a translated quote from a native-English one. The compensation
is that provenance is written at the moment the claim is made — which is what lets v2 add the
label, a score or a verification pass WITHOUT reprocessing. A claim that shipped without it can
never acquire it, because the units behind it are not recoverable afterwards.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from podcast_scraper.translation.artifacts import TranslationDocument, UnitRecord
from podcast_scraper.translation.provenance import (
    attach_translation_provenance,
    provenance_coverage,
    PROVENANCE_KEY,
    target_units_sha256,
)

pytestmark = pytest.mark.unit


def _doc(**kw: Any) -> TranslationDocument:
    units = kw.pop(
        "units",
        [
            UnitRecord(
                unit_id="t0000.u01",
                turn_id="t0000",
                content_key="k0",
                status="ok",
                sentences=[{"sent_id": "t0000.s01", "en_text": "One."}],
            ),
            UnitRecord(
                unit_id="t0001.u01",
                turn_id="t0001",
                content_key="k1",
                status="ok",
                sentences=[{"sent_id": "t0001.s01", "en_text": "Two."}],
            ),
        ],
    )
    base: Dict[str, Any] = {
        "source_language": "es",
        "model": "google/translategemma-12b-it",
        "prompt": {"name": "shared/translation/translategemma_v1", "sha256": "a" * 64},
        "units": units,
    }
    base.update(kw)
    return TranslationDocument(**base)


_EN_SEGMENTS: List[Dict[str, Any]] = [
    {"unit_id": "t0000.u01", "sent_id": "t0000.s01", "char_start": 6, "char_end": 10},
    {"unit_id": "t0001.u01", "sent_id": "t0001.s01", "char_start": 17, "char_end": 21},
]


def _payload(*spans: tuple) -> Dict[str, Any]:
    return {
        "nodes": [
            {
                "id": f"quote:q{i}",
                "type": "Quote",
                "properties": {"char_start": s, "char_end": e, "text": "x"},
            }
            for i, (s, e) in enumerate(spans)
        ]
        + [{"id": "insight:i0", "type": "Insight", "properties": {"text": "no span"}}]
    }


class TestTheBlock:
    def test_a_quote_gets_the_units_that_produced_it(self) -> None:
        payload = _payload((6, 10))
        assert attach_translation_provenance(payload, _doc(), _EN_SEGMENTS) == 1
        block = payload["nodes"][0]["properties"][PROVENANCE_KEY]
        assert block["translated"] is True
        assert block["source_language"] == "es"
        assert block["unit_ids"] == ["t0000.u01"]
        assert block["model"] == "google/translategemma-12b-it"
        assert len(block["en_sha256"]) == 64

    def test_the_prompt_hash_rides_along(self) -> None:
        """A changed prompt changes the translation without changing the model id — measured
        today, when a paraphrased prompt turned clean translations into commentary."""
        payload = _payload((6, 10))
        attach_translation_provenance(payload, _doc(), _EN_SEGMENTS)
        assert payload["nodes"][0]["properties"][PROVENANCE_KEY]["prompt_sha256"] == "a" * 64

    def test_a_span_crossing_two_units_records_both(self) -> None:
        payload = _payload((6, 21))
        attach_translation_provenance(payload, _doc(), _EN_SEGMENTS)
        assert payload["nodes"][0]["properties"][PROVENANCE_KEY]["unit_ids"] == [
            "t0000.u01",
            "t0001.u01",
        ]

    def test_a_span_resolving_to_nothing_gets_NO_block(self) -> None:
        """Not an empty one. `unit_ids: []` reads as "translated, from nothing", which asserts
        provenance while carrying none — worse than absence, which is at least countable."""
        payload = _payload((11, 16))
        assert attach_translation_provenance(payload, _doc(), _EN_SEGMENTS) == 0
        assert PROVENANCE_KEY not in payload["nodes"][0]["properties"]

    def test_nodes_without_spans_are_left_alone(self) -> None:
        payload = _payload((6, 10))
        attach_translation_provenance(payload, _doc(), _EN_SEGMENTS)
        assert PROVENANCE_KEY not in payload["nodes"][-1]["properties"]


class TestIdempotence:
    def test_running_twice_leaves_one_block(self) -> None:
        """Three writers produce gi.json — the artifact builder,
        `add_spoken_by_edges(replace=True)` and `gi/repair.py` — so a node passing through two
        must not end up with two blocks or a half-updated one."""
        payload = _payload((6, 10))
        attach_translation_provenance(payload, _doc(), _EN_SEGMENTS)
        first = dict(payload["nodes"][0]["properties"][PROVENANCE_KEY])
        attach_translation_provenance(payload, _doc(), _EN_SEGMENTS)
        assert payload["nodes"][0]["properties"][PROVENANCE_KEY] == first

    def test_a_stale_block_is_REMOVED_when_the_span_no_longer_resolves(self) -> None:
        """After a re-render a span can stop touching any unit. Leaving the old block would
        attribute the claim to units that no longer produced it."""
        payload = _payload((6, 10))
        attach_translation_provenance(payload, _doc(), _EN_SEGMENTS)
        assert PROVENANCE_KEY in payload["nodes"][0]["properties"]
        attach_translation_provenance(payload, _doc(), [])
        assert PROVENANCE_KEY not in payload["nodes"][0]["properties"]


class TestTheHashIsLabelFree:
    def test_it_ignores_speaker_labels_and_offsets_entirely(self) -> None:
        """D-40. A rename changes every `Label:` prefix and therefore the whole `.en.txt`.
        Hashing the render would invalidate provenance on every claim whenever naming is re-run,
        for a reason with nothing to do with the translation."""
        a = _doc()
        b = _doc(
            units=[
                UnitRecord(
                    unit_id="t0000.u01",
                    turn_id="t0000",
                    content_key="DIFFERENT",
                    status="ok",
                    sentences=[{"sent_id": "RENUMBERED", "en_text": "One."}],
                ),
                UnitRecord(
                    unit_id="t0001.u01",
                    turn_id="t0001",
                    content_key="ALSO-DIFFERENT",
                    status="ok",
                    sentences=[{"sent_id": "ALSO", "en_text": "Two."}],
                ),
            ]
        )
        assert target_units_sha256(a) == target_units_sha256(b)

    def test_it_changes_when_the_english_text_changes(self) -> None:
        a = _doc()
        b = _doc(
            units=[
                UnitRecord(
                    unit_id="t0000.u01",
                    turn_id="t0000",
                    content_key="k0",
                    status="ok",
                    sentences=[{"sent_id": "t0000.s01", "en_text": "A DIFFERENT TRANSLATION."}],
                )
            ]
        )
        assert target_units_sha256(a) != target_units_sha256(b)

    def test_it_changes_with_the_model(self) -> None:
        """Two runs of the same source through different models are different translations, and
        the corpus has to be able to tell — ADR-143/144's reasoning, at claim level."""
        assert target_units_sha256(_doc()) != target_units_sha256(
            _doc(model="google/translategemma-27b-it")
        )

    def test_failed_units_do_not_enter_the_hash(self) -> None:
        """They contributed no English text, so including their ids would make the hash depend
        on failures rather than on the translation."""
        with_failure = _doc(
            units=_doc().units
            + [UnitRecord(unit_id="t0002.u01", turn_id="t0002", content_key="k2", status="failed")]
        )
        # Not complete, so the hash still describes only the units that produced text.
        assert target_units_sha256(with_failure) == target_units_sha256(_doc())


class TestAnIncompleteTranslationIsNeverDecorated:
    def test_a_failed_translation_decorates_nothing(self) -> None:
        """Provenance describes a translation that was actually published. An incomplete one has
        no `.en.*` on disk (RFC-124 §5.3), so no claim should exist to decorate — and if one
        does, attaching provenance would legitimise it."""
        doc = _doc(
            units=_doc().units
            + [UnitRecord(unit_id="t0002.u01", turn_id="t0002", content_key="k2", status="failed")]
        )
        payload = _payload((6, 10))
        assert attach_translation_provenance(payload, doc, _EN_SEGMENTS) == 0
        assert PROVENANCE_KEY not in payload["nodes"][0]["properties"]


class TestCoverage:
    def test_the_gap_is_what_gets_counted(self) -> None:
        """A span-bearing node without provenance on a translated episode is a claim v2 can
        never label, and it is invisible unless counted."""
        payload = _payload((6, 10), (11, 16))
        attach_translation_provenance(payload, _doc(), _EN_SEGMENTS)
        assert provenance_coverage(payload) == {"span_nodes": 2, "with_provenance": 1}

    def test_an_empty_artifact_reports_zeroes_rather_than_dividing_by_them(self) -> None:
        assert provenance_coverage({"nodes": []}) == {"span_nodes": 0, "with_provenance": 0}


class TestItReachesEveryWriter:
    """S2.11 requires provenance from EVERY gi.json writer, and there are more than ten call
    sites — the artifact builder, `add_spoken_by_edges(replace=True)`, `gi/repair.py`, the
    bridge post-pass, topic clustering. A decoration step added at each is one that will be
    missed at the eleventh, so it hangs off the single choke point they all pass through.
    """

    def test_write_artifact_decorates_from_the_ambient_episode(self, tmp_path: Any) -> None:
        from podcast_scraper.gi.io import write_artifact
        from podcast_scraper.translation.provenance import publish_episode_translation

        publish_episode_translation(_doc(), _EN_SEGMENTS)
        try:
            payload = _payload((6, 10))
            out = tmp_path / "ep.gi.json"
            write_artifact(out, payload, validate=False)

            import json

            written = json.loads(out.read_text(encoding="utf-8"))
            block = written["nodes"][0]["properties"][PROVENANCE_KEY]
            assert block["unit_ids"] == ["t0000.u01"]
        finally:
            publish_episode_translation(None, None)

    def test_an_english_episode_writes_no_provenance(self, tmp_path: Any) -> None:
        """The no-op path, which every one of the 678 English episodes takes."""
        import json

        from podcast_scraper.gi.io import write_artifact
        from podcast_scraper.translation.provenance import publish_episode_translation

        publish_episode_translation(None, None)
        payload = _payload((6, 10))
        out = tmp_path / "ep.gi.json"
        write_artifact(out, payload, validate=False)
        written = json.loads(out.read_text(encoding="utf-8"))
        assert PROVENANCE_KEY not in written["nodes"][0]["properties"]

    def test_publishing_a_second_episode_replaces_the_first(self, tmp_path: Any) -> None:
        """The leak this design prevents. Workers are reused across episodes, so a value set and
        never replaced would decorate episode B with episode A's units — provenance pointing at
        text from a different show."""
        import json

        from podcast_scraper.gi.io import write_artifact
        from podcast_scraper.translation.provenance import publish_episode_translation

        publish_episode_translation(_doc(), _EN_SEGMENTS)
        publish_episode_translation(None, None)  # the next episode is English
        payload = _payload((6, 10))
        out = tmp_path / "ep2.gi.json"
        write_artifact(out, payload, validate=False)
        written = json.loads(out.read_text(encoding="utf-8"))
        assert PROVENANCE_KEY not in written["nodes"][0]["properties"]

    def test_a_broken_decoration_never_blocks_the_write(self, tmp_path: Any) -> None:
        """An artifact that fails to write is an episode lost. Provenance is not worth that."""
        import json

        from podcast_scraper.gi.io import write_artifact
        from podcast_scraper.translation.provenance import publish_episode_translation

        class _Exploding(dict):
            def get(self, *_a: Any, **_k: Any) -> Any:
                raise RuntimeError("boom")

        publish_episode_translation(_doc(), [_Exploding()])
        try:
            payload = _payload((6, 10))
            out = tmp_path / "ep3.gi.json"
            write_artifact(out, payload, validate=False)
            assert json.loads(out.read_text(encoding="utf-8"))["nodes"], "the artifact landed"
        finally:
            publish_episode_translation(None, None)
