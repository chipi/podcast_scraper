"""RFC-124 §6.2: a translated episode is indexed TWICE — English analysis body and source body.

WHY. Goal 6 is "an episode findable in the language it was spoken in", and the English-first
resolution order (D-38) cannot deliver it by construction: `TranscriptPurpose.ANALYSIS` resolves
AWAY from the source language the moment a translation exists. So for a translated episode the
only text that ever reached the index was the English render, and a query in the episode's own
language matched nothing. `segments_nonen` — the vector-less tier built for exactly this — only
ever received episodes whose translation was pending or had failed.

The RFC named four things that are easy to get wrong here. Three of them are correctness traps
rather than features, and each has its own class below:

* **Chunk ids must carry language**, because LanceDB merges on id — so the second layer would
  silently OVERWRITE the first rather than adding to it.
* **The quote-offset verifier must skip the source layer.** GI's Quote offsets index the English
  analysis body; both layers key on the same `episode_id`; char offsets are just integers. An
  English quote span "overlapping" a Spanish chunk span is a spurious pass in the one
  measurement #528 exists to produce.
* **Insight→segment linking and the transcript lift must skip it too.** Both layers share the
  audio timeline, so a time-based match cannot tell them apart, and a char-overlap match cannot
  either.

The discriminator is an explicit `index_layer` marker, NOT "is the language English". That
distinction is the subtle part: an UNTRANSLATED Spanish episode's chunks are Spanish AND are the
analysis space, because the Spanish body is the only body. What consumers need to know is "are
these offsets comparable with GI's", and only the marker says that. Absent means analysis, which
is right for every row in every index built before the field existed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from podcast_scraper.search.indexer import (
    _collect_docs_for_episode,
    _source_layer_body,
    _transcript_path,
    INDEX_LAYER_KEY,
    INDEX_LAYER_SOURCE,
    is_source_layer_chunk,
)

pytestmark = pytest.mark.unit

_ES = "Hola y bienvenidos al programa de hoy. Hablamos de la inflación. "
_EN = "Hello and welcome to today's show. We talk about inflation. "

# D-44: a TRANSLATED episode is a SWAPPED one. The canonical body holds English (ASR wrote the
# source there, the atomic swap replaced it) and the source is kept at its language-tagged name.
# A PENDING one never swapped, so its canonical body is still the source and there is no tagged
# sibling — which is exactly the signal the completeness gate reads.
_TRANSLATED = ["ep1.txt", "ep1.adfree.txt", "ep1.es.txt"]
_PENDING = ["ep1.txt", "ep1.adfree.txt"]


def _episode(tmp_path: Path, files: List[str], language: Optional[str]) -> Tuple[Path, Dict, Path]:
    root = Path(str(tmp_path)).resolve()
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    # The canonical names hold ENGLISH for a swapped episode. `_PENDING` asks for the same two
    # names and must get the SOURCE text, so the caller's list decides which body is which: a
    # pending episode is built from `_PENDING_BODIES`.
    bodies = {
        "ep1.txt": _EN * 30,
        "ep1.adfree.txt": _EN * 30,
        "ep1.es.txt": _ES * 30,
        "ep1.es.adfree.txt": _ES * 30,
    }
    if files is _PENDING:
        bodies = {"ep1.txt": _ES * 30, "ep1.adfree.txt": _ES * 30}
    for name in files:
        (root / "transcripts" / name).write_text(bodies[name], encoding="utf-8")
    doc = {
        "episode": {"episode_id": "ep1", "title": "Ep 1", "language": language},
        "feed": {"feed_id": "f1", "title": "Feed", "language": language},
        "content": {"transcript_file_path": "transcripts/ep1.txt"},
    }
    meta_path = root / "m.metadata.json"
    meta_path.write_text(json.dumps(doc), encoding="utf-8")
    return root, doc, meta_path


def _transcript_rows(tmp_path: Path, files: List[str], language: Optional[str]) -> List[Tuple]:
    root, doc, meta_path = _episode(tmp_path, files, language)
    rows = _collect_docs_for_episode(
        root,
        meta_path,
        doc,
        target_tokens=60,
        overlap_tokens=10,
        metadata_relative_path="m.metadata.json",
    )
    return [(i, txt, m) for i, txt, m in rows if m.get("doc_type") == "transcript"]


class TestATranslatedEpisodeGetsBOTHLayers:
    def test_chunks_exist_in_the_source_language_AND_english(self, tmp_path: Path) -> None:
        rows = _transcript_rows(tmp_path, _TRANSLATED, "es")
        languages = {m.get("language") for _i, _t, m in rows}
        assert languages == {"en", "es"}

    def test_the_english_chunks_hold_english_text(self, tmp_path: Path) -> None:
        rows = _transcript_rows(tmp_path, _TRANSLATED, "es")
        english = [txt for _i, txt, m in rows if m.get("language") == "en"]
        assert english and all("welcome" in t.lower() for t in english)

    def test_the_source_chunks_hold_the_SOURCE_text(self, tmp_path: Path) -> None:
        """The point of the whole layer: a query in the episode's own language has something to
        match. Before this, `inflación` appeared nowhere in the index for this episode."""
        rows = _transcript_rows(tmp_path, _TRANSLATED, "es")
        source = [txt for _i, txt, m in rows if m.get("language") == "es"]
        assert source and any("inflación" in t for t in source)

    def test_only_the_source_rows_carry_the_layer_marker(self, tmp_path: Path) -> None:
        rows = _transcript_rows(tmp_path, _TRANSLATED, "es")
        for _i, _t, m in rows:
            expected = m.get("language") == "es"
            assert is_source_layer_chunk(m) is expected


class TestTheIdsDoNotCollide:
    """LanceDB merges on id: identical ids mean the second layer overwrites the first, and the
    index would end up with ONE layer's text under both languages' worth of rows."""

    def test_every_id_is_unique(self, tmp_path: Path) -> None:
        rows = _transcript_rows(tmp_path, _TRANSLATED, "es")
        ids = [i for i, _t, _m in rows]
        assert len(ids) == len(set(ids))

    def test_the_source_ids_are_suffixed_and_the_english_ones_are_NOT(self, tmp_path: Path) -> None:
        """English ids stay byte-identical to what they were before this layer existed, so
        stored `source_segment_id` references keep resolving — the RFC's stated reason for
        suffixing only the new side."""
        rows = _transcript_rows(tmp_path, _TRANSLATED, "es")
        for i, _t, m in rows:
            if m.get("language") == "es":
                assert i.endswith(":src")
            else:
                assert not i.endswith(":src")

    def test_an_english_episodes_ids_are_unchanged_by_this_feature(self, tmp_path: Path) -> None:
        rows = _transcript_rows(tmp_path, _PENDING, "en")
        assert [i for i, _t, _m in rows] == [f"chunk:f1__ep1:{n}" for n in range(len(rows))]


class TestWhenTHERE_IS_NoSecondLayer:
    """One layer in every case where a second would be a duplicate or a lie."""

    def test_a_pending_translation_gets_one_layer_in_its_own_language(self, tmp_path: Path) -> None:
        """With no English render the primary pass ALREADY indexes the source language, so a
        second pass would chunk the same body twice under two ids."""
        rows = _transcript_rows(tmp_path, _PENDING, "es")
        assert {m.get("language") for _i, _t, m in rows} == {"es"}
        assert not any(is_source_layer_chunk(m) for _i, _t, m in rows)

    def test_an_english_episode_gets_one_layer(self, tmp_path: Path) -> None:
        rows = _transcript_rows(tmp_path, _PENDING, "en")
        assert {m.get("language") for _i, _t, m in rows} == {"en"}
        assert not any(is_source_layer_chunk(m) for _i, _t, m in rows)

    def test_an_episode_with_no_recorded_language_gets_one_layer(self, tmp_path: Path) -> None:
        """Most of the corpus. A second layer here would double the index for 678 episodes."""
        rows = _transcript_rows(tmp_path, _PENDING, None)
        assert {m.get("language") for _i, _t, m in rows} == {None}

    def test_the_helper_refuses_when_both_resolvers_agree(self, tmp_path: Path) -> None:
        """Defensive: if the source resolver ever returned the body the primary pass read, the
        second pass would double every chunk instead of adding a layer."""
        root, doc, _mp = _episode(tmp_path, _PENDING, "es")
        # `indexed_language` is "es" here, so the guard short-circuits before resolution.
        assert _source_layer_body(root, doc, "es") is None

    def test_the_helper_names_the_source_body_for_a_translated_episode(
        self, tmp_path: Path
    ) -> None:
        root, doc, _mp = _episode(tmp_path, _TRANSLATED, "es")
        result = _source_layer_body(root, doc, "en")
        assert result is not None
        path, language = result
        assert language == "es"
        # D-44: the source body carries the language tag. `ep1.adfree.txt` is now the ENGLISH
        # analysis base — the body the primary pass reads — so naming it here would index the same
        # text twice under two ids instead of adding a layer.
        assert str(path.relative_to(root)) == "transcripts/ep1.es.txt"
        assert path != _transcript_path(root, doc), "must not be the body the primary pass read"


class TestTheQuoteOffsetVerifierSkipsTheSourceLayer:
    """#528 measures whether GI Quote offsets land inside indexed chunks. Quote offsets index
    the ENGLISH body; a source chunk's offsets index the Spanish one. Both are integers keyed to
    the same episode, so an "overlap" between them is arithmetic, not evidence."""

    @staticmethod
    def _metadata() -> Dict[str, Dict[str, Any]]:
        return {
            "chunk:f1__ep1:0": {
                "doc_type": "transcript",
                "episode_id": "ep1",
                "char_start": 0,
                "char_end": 500,
                "language": "en",
            },
            "chunk:f1__ep1:0:src": {
                "doc_type": "transcript",
                "episode_id": "ep1",
                "char_start": 0,
                "char_end": 480,
                "language": "es",
                INDEX_LAYER_KEY: INDEX_LAYER_SOURCE,
            },
        }

    def test_only_the_analysis_layer_is_counted(self) -> None:
        from podcast_scraper.search.gil_chunk_offset_verify import (
            transcript_chunk_spans_by_episode,
        )

        spans = transcript_chunk_spans_by_episode(self._metadata())
        assert spans == {"ep1": [(0, 500)]}, "the source layer must not inflate the chunk count"

    def test_a_legacy_row_with_no_marker_is_still_counted(self) -> None:
        """Every index built before the marker existed has no such field, and all of its rows
        are analysis-space. Treating absent as "source" would empty the verifier."""
        from podcast_scraper.search.gil_chunk_offset_verify import (
            transcript_chunk_spans_by_episode,
        )

        legacy = {
            "chunk:f1__ep1:0": {
                "doc_type": "transcript",
                "episode_id": "ep1",
                "char_start": 10,
                "char_end": 20,
            }
        }
        assert transcript_chunk_spans_by_episode(legacy) == {"ep1": [(10, 20)]}


class TestTheLiftPathSkipsTheSourceLayer:
    def test_a_source_layer_row_is_never_lifted(self, tmp_path: Path) -> None:
        """A char-range overlap against the English-derived GI artifact would attach Spanish
        text as the evidence under an English insight, and the overlap would look valid."""
        from podcast_scraper.search.transcript_chunk_lift import (
            lift_row_if_transcript,
            TranscriptLiftGiCache,
        )

        row: Dict[str, Any] = {
            "metadata": {
                "doc_type": "transcript",
                "episode_id": "ep1",
                "char_start": 0,
                "char_end": 100,
                "language": "es",
                INDEX_LAYER_KEY: INDEX_LAYER_SOURCE,
            }
        }
        lift_row_if_transcript(
            row, Path(str(tmp_path)), Path(str(tmp_path)) / "gi.json", TranscriptLiftGiCache()
        )
        assert "lifted" not in row


class TestInsightLinkingSkipsTheSourceLayer:
    """Both layers share the audio timeline, so a time-based link cannot tell them apart."""

    @staticmethod
    def _segments() -> List[Any]:
        from podcast_scraper.search.backend import SegmentDocument

        # Spelled out rather than splatted from a dict: mypy cannot narrow `**dict[str, object]`
        # against the dataclass signature, and a `# type: ignore` here would hide a real
        # signature change in the thing under test.
        return [
            SegmentDocument(
                id="ep1_chunk_0_src",
                text="español",
                show_id="f1",
                episode_id="ep1",
                start_time=0.0,
                end_time=30.0,
                language="es",
            ),
            SegmentDocument(
                id="ep1_chunk_0",
                text="english",
                show_id="f1",
                episode_id="ep1",
                start_time=0.0,
                end_time=30.0,
                language="en",
            ),
        ]

    def test_an_insight_links_the_ENGLISH_segment_even_when_listed_second(self) -> None:
        """Order matters: the linker takes the FIRST time match, and the source segment is put
        first here on purpose — without the filter it would win."""
        from podcast_scraper.search.segments import link_insights_to_segments

        segments = self._segments()
        mapping = link_insights_to_segments(segments, [("ins1", 5.0, 10.0)])
        assert mapping == {"ins1": "ep1_chunk_0"}

    def test_the_source_segment_gets_no_linked_insight(self) -> None:
        from podcast_scraper.search.segments import link_insights_to_segments

        segments = self._segments()
        link_insights_to_segments(segments, [("ins1", 5.0, 10.0)])
        by_id = {s.id: s for s in segments}
        assert by_id["ep1_chunk_0_src"].linked_insight_ids == []
        assert by_id["ep1_chunk_0"].linked_insight_ids == ["ins1"]

    def test_a_segment_with_no_language_still_links(self) -> None:
        """The pre-language corpus, which is most of it. Excluding those would stop insight
        linking for 678 episodes."""
        from podcast_scraper.search.backend import SegmentDocument
        from podcast_scraper.search.segments import link_insights_to_segments

        seg = SegmentDocument(
            id="ep1_chunk_0",
            text="t",
            show_id="f1",
            episode_id="ep1",
            start_time=0.0,
            end_time=30.0,
        )
        assert link_insights_to_segments([seg], [("ins1", 5.0, 10.0)]) == {"ins1": "ep1_chunk_0"}


class TestTheSourceResolverOnlyReturnsTaggedPaths:
    """D-44: the source body carries the language tag, so the resolver cannot return English.

    This class used to assert "no candidate is an English render", checked with
    `is_english_render_relpath` — a predicate that inspected the suffix stack to work out whether a
    path was a translation. Both are gone: English is the UNSUFFIXED canonical file, so a candidate
    for the SOURCE layer is tagged by construction and there is nothing to detect.
    """

    def test_every_candidate_carries_the_language_tag(self) -> None:
        from podcast_scraper.workflow.transcript_resolution import (
            source_language_relpath_candidates,
        )

        candidates = source_language_relpath_candidates("transcripts/ep1.txt", "es")
        assert candidates
        for cand in candidates:
            assert ".es." in cand, cand

    def test_english_has_no_source_layer_at_all(self) -> None:
        """The canonical body already holds English, so there is no second body to index."""
        from podcast_scraper.workflow.transcript_resolution import (
            source_language_relpath_candidates,
        )

        assert source_language_relpath_candidates("transcripts/ep1.txt", "en") == []

    def test_ad_free_comes_first(self) -> None:
        """Matching ANALYSIS. For a translated episode it usually will not exist — ad excision
        runs on the English text (D-19) — so this normally lands on the tagged source body."""
        from podcast_scraper.workflow.transcript_resolution import (
            source_language_relpath_candidates,
        )

        assert source_language_relpath_candidates("transcripts/ep1.txt", "es") == [
            "transcripts/ep1.es.adfree.txt",
            "transcripts/ep1.es.txt",
        ]
