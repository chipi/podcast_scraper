"""Tests for GIL load layer (build_inspect_output, evidence span, find artifact)."""

import json
from pathlib import Path

import pytest

from podcast_scraper.gi import (
    build_artifact,
    build_inspect_output,
    find_artifact_by_episode_id,
    find_artifact_by_insight_id,
    load_artifact_and_transcript,
)
from podcast_scraper.gi.load import (
    _transcript_path_from_artifact_path,
    get_evidence_span,
    load_transcript_for_evidence,
)


@pytest.mark.unit
class TestGILLoad:
    """Load layer and build_inspect_output."""

    def test_transcript_path_from_artifact_path(self):
        """Transcript path is output_dir/transcripts/<base>.txt."""
        p = Path("/out/metadata/1 - episode.gi.json")
        t = _transcript_path_from_artifact_path(p)
        assert t == Path("/out/transcripts/1 - episode.txt")

    def test_get_evidence_span_excerpt(self):
        """Evidence span excerpt is transcript slice."""
        text = "Hello world here is the quote."
        span = get_evidence_span(text, 6, 11, transcript_ref="ep.txt")
        assert span.char_start == 6
        assert span.char_end == 11
        assert span.excerpt == "world"

    def test_get_evidence_span_out_of_range_excerpt_none(self):
        """When char_start/char_end are out of range, excerpt is None."""
        text = "hello"
        span = get_evidence_span(text, 0, 100, transcript_ref="ep.txt")
        assert span.char_start == 0
        assert span.char_end == 100
        assert span.excerpt is None

    def test_load_transcript_for_evidence_missing_returns_none(self, tmp_path):
        """load_transcript_for_evidence returns None when file is missing."""
        missing = tmp_path / "missing.txt"
        assert not missing.exists()
        assert load_transcript_for_evidence(missing) is None

    def test_load_transcript_for_evidence_existing_returns_content(self, tmp_path):
        """load_transcript_for_evidence returns file content when file exists."""
        path = tmp_path / "transcript.txt"
        path.write_text("Evidence here.", encoding="utf-8")
        assert load_transcript_for_evidence(path) == "Evidence here."

    def test_build_inspect_output_from_artifact(self):
        """build_inspect_output produces InspectOutput with insights and stats."""
        artifact = build_artifact(
            "ep:1",
            "Some transcript.",
            prompt_version="v1",
            insight_texts=["A real insight extracted from the transcript."],
        )
        out = build_inspect_output(artifact, "Some transcript.")
        assert out.episode_id == "ep:1"
        assert len(out.insights) == 1
        # Ungrounded because no evidence provider is wired in this test — grounding is the
        # subject of the evidence-stack tests, not of build_inspect_output's shape. There is no
        # Quote either: quotes come from grounding, and the placeholder that used to manufacture
        # one from a transcript slice is gone (#1657).
        assert out.insights[0].grounded is False
        assert out.stats["insight_count"] == 1
        assert out.stats["quote_count"] == 0

    def test_build_inspect_output_episode_title_and_publish_date(self):
        """Episode node title/publish_date propagate to each InsightSummary."""
        artifact = build_artifact(
            "ep:1",
            "Some transcript.",
            prompt_version="v1",
            episode_title="My Episode",
            publish_date="2024-06-01T12:00:00Z",
            insight_texts=["A real insight extracted from the transcript."],
        )
        out = build_inspect_output(artifact, "Some transcript.")
        assert out.insights[0].episode_title == "My Episode"
        assert out.insights[0].publish_date == "2024-06-01T12:00:00Z"

    def test_load_artifact_and_transcript_roundtrip(self, tmp_path):
        """Load artifact from path; transcript optional."""
        payload = build_artifact(
            "ep:1",
            "Evidence here.",
            prompt_version="v1",
            insight_texts=["A real insight extracted from the transcript."],
        )
        gi_path = tmp_path / "metadata" / "1_ep.gi.json"
        gi_path.parent.mkdir(parents=True)
        with open(gi_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        (tmp_path / "transcripts").mkdir()
        (tmp_path / "transcripts" / "1_ep.txt").write_text("Evidence here.", encoding="utf-8")
        artifact, transcript, tpath = load_artifact_and_transcript(
            gi_path, validate=True, load_transcript=True
        )
        assert artifact["episode_id"] == "ep:1"
        assert transcript == "Evidence here."
        assert tpath == tmp_path / "transcripts" / "1_ep.txt"

    def test_find_artifact_by_insight_id(self, tmp_path):
        """find_artifact_by_insight_id returns path to artifact containing insight."""
        payload = build_artifact(
            "ep:1",
            "Evidence here.",
            prompt_version="v1",
            insight_texts=["A real insight extracted from the transcript."],
        )
        insight_id = next(n["id"] for n in payload["nodes"] if n.get("type") == "Insight")
        gi_path = tmp_path / "metadata" / "ep1.gi.json"
        gi_path.parent.mkdir(parents=True)
        with open(gi_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        found = find_artifact_by_insight_id(tmp_path, insight_id)
        assert found == gi_path
        assert find_artifact_by_insight_id(tmp_path, "nonexistent") is None

    def test_find_artifact_by_episode_id(self, tmp_path):
        """find_artifact_by_episode_id returns path to artifact with given episode_id."""
        payload = build_artifact(
            "ep:1",
            "Evidence here.",
            prompt_version="v1",
            insight_texts=["A real insight extracted from the transcript."],
        )
        metadata = tmp_path / "metadata"
        metadata.mkdir(parents=True)
        gi_path = metadata / "ep1.gi.json"
        with open(gi_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        found = find_artifact_by_episode_id(tmp_path, "ep:1")
        assert found == gi_path
        assert find_artifact_by_episode_id(tmp_path, "ep:nonexistent") is None

    def test_find_artifact_by_episode_id_no_metadata_dir_returns_none(self, tmp_path):
        """find_artifact_by_episode_id returns None when metadata dir does not exist."""
        assert find_artifact_by_episode_id(tmp_path, "ep:1") is None

    def test_find_artifact_by_episode_id_multi_feed_requires_feed_id(self, tmp_path):
        """Same episode_id under two feeds is ambiguous without feed_id."""
        import json as json_mod

        corpus = tmp_path / "corpus"
        for fid, slug in (("feed_a", "rss_a"), ("feed_b", "rss_b")):
            mdir = corpus / "feeds" / slug / "run" / "metadata"
            mdir.mkdir(parents=True, exist_ok=True)
            meta_doc = {"feed": {"feed_id": fid}, "episode": {"episode_id": "dup"}}
            (mdir / "ep.metadata.json").write_text(
                json_mod.dumps(meta_doc),
                encoding="utf-8",
            )
            payload = build_artifact(
                "dup",
                "t",
                prompt_version="v1",
                insight_texts=["A real insight extracted from the transcript."],
            )
            with open(mdir / "ep.gi.json", "w", encoding="utf-8") as handle:
                json_mod.dump(payload, handle)

        assert find_artifact_by_episode_id(corpus, "dup") is None
        one = find_artifact_by_episode_id(corpus, "dup", feed_id="feed_a")
        assert one is not None
        assert "rss_a" in str(one)


@pytest.mark.unit
class TestItReadsTheBodyTheArtifactDECLARES:
    """#2253: which transcript an evidence span is sliced from is DECLARED, not guessed.

    GI records `transcript_ref` on every offset-bearing node — the body those `char_start` /
    `char_end` were measured against. Slicing any other body returns the right NUMBER of
    characters from the wrong place, which prints as plausible text and raises nothing. That was
    the bug: the reader derived `<base>.txt` while GI had measured against `<base>.adfree.txt`,
    minutes shorter, so every span on an ad-excised episode was displaced by the length of the ads
    before it.

    Two guessing fixes were tried first and both were wrong the same way — one routed through the
    translation module, one checked `.adfree.txt` inline and left a translated episode reading its
    source body at English offsets. These tests pin the declaration being obeyed instead.
    """

    @staticmethod
    def _artifact(ref: object, *, offsets: bool = True) -> dict:
        props: dict = {"char_start": 0, "char_end": 5} if offsets else {}
        if ref is not None:
            props["transcript_ref"] = ref
        return {"episode_id": "ep1", "nodes": [{"type": "Quote", "properties": props}]}

    def _artifact_path(self, tmp_path: Path) -> Path:
        meta = tmp_path / "metadata"
        meta.mkdir(parents=True, exist_ok=True)
        (tmp_path / "transcripts").mkdir(parents=True, exist_ok=True)
        return meta / "01 - ep.gi.json"

    def test_it_returns_the_declared_body_even_when_it_is_the_adfree_one(
        self, tmp_path: Path
    ) -> None:
        """The bug, in one assertion: the declared body wins over the derived `.txt`."""
        ap = self._artifact_path(tmp_path)
        got = _transcript_path_from_artifact_path(
            ap, self._artifact("transcripts/01 - ep.adfree.txt")
        )
        assert got == (tmp_path / "transcripts" / "01 - ep.adfree.txt").resolve()

    def test_it_returns_the_declared_body_for_a_TRANSLATED_episode(self, tmp_path: Path) -> None:
        """The case the second attempt got wrong. An English render is just another declared ref —
        no `.en.` knowledge is needed here, which is why `gi` stays free of the language code."""
        ap = self._artifact_path(tmp_path)
        got = _transcript_path_from_artifact_path(
            ap, self._artifact("transcripts/01 - ep.en.adfree.txt")
        )
        assert got == (tmp_path / "transcripts" / "01 - ep.en.adfree.txt").resolve()

    def test_an_artifact_that_declares_nothing_falls_back_to_the_derived_path(
        self, tmp_path: Path
    ) -> None:
        """Artifacts written before the field existed. One predictable answer, not a guess."""
        ap = self._artifact_path(tmp_path)
        got = _transcript_path_from_artifact_path(ap, self._artifact(None))
        assert got == tmp_path / "transcripts" / "01 - ep.txt"

    def test_no_artifact_in_hand_falls_back_too(self, tmp_path: Path) -> None:
        ap = self._artifact_path(tmp_path)
        assert _transcript_path_from_artifact_path(ap) == (tmp_path / "transcripts" / "01 - ep.txt")

    def test_DISAGREEING_declarations_fall_back_rather_than_pick_one(self, tmp_path: Path) -> None:
        """One artifact comes from one body, so this should be impossible — and if it happens,
        picking either would show the wrong evidence for half the insights. One predictable
        wrongness beats per-node wrongness."""
        ap = self._artifact_path(tmp_path)
        artifact = {
            "episode_id": "ep1",
            "nodes": [
                {
                    "type": "Quote",
                    "properties": {
                        "char_start": 0,
                        "char_end": 5,
                        "transcript_ref": "transcripts/a.txt",
                    },
                },
                {
                    "type": "Quote",
                    "properties": {
                        "char_start": 6,
                        "char_end": 9,
                        "transcript_ref": "transcripts/b.txt",
                    },
                },
            ],
        }
        assert _transcript_path_from_artifact_path(ap, artifact) == (
            tmp_path / "transcripts" / "01 - ep.txt"
        )

    def test_a_ref_that_escapes_the_run_directory_is_refused(self, tmp_path: Path) -> None:
        """`transcript_ref` is produced by this pipeline, but it is still a path read out of a
        file, so it is confined under the run dir before being opened."""
        ap = self._artifact_path(tmp_path)
        got = _transcript_path_from_artifact_path(ap, self._artifact("../../../../etc/passwd"))
        assert got == tmp_path / "transcripts" / "01 - ep.txt"

    def test_a_node_with_NO_offsets_declares_nothing(self, tmp_path: Path) -> None:
        """Only offset-bearing nodes answer the question — a node with a ref but no `char_start`
        is describing something else and must not decide which body gets sliced."""
        ap = self._artifact_path(tmp_path)
        got = _transcript_path_from_artifact_path(
            ap, self._artifact("transcripts/01 - ep.adfree.txt", offsets=False)
        )
        assert got == tmp_path / "transcripts" / "01 - ep.txt"
