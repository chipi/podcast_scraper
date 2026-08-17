"""Tests for GIL corpus export helpers."""

import json

import pytest
from conftest import artifact_with_grounded_insights

from podcast_scraper.gi import build_artifact, write_artifact
from podcast_scraper.gi.contracts import build_gi_corpus_bundle_output
from podcast_scraper.gi.corpus import export_merged_json, export_ndjson, load_gi_artifacts


@pytest.mark.unit
class TestGiCorpusExport:
    """NDJSON and merged bundle export."""

    def test_export_merged_json_counts(self, tmp_path):
        """Merged bundle lists artifacts and insight/quote totals."""
        (tmp_path / "metadata").mkdir()
        for i, eid in enumerate(("ep:a", "ep:b"), start=1):
            art = artifact_with_grounded_insights(eid, f"Transcript {i} body here.")
            write_artifact(tmp_path / "metadata" / f"{i}.gi.json", art, validate=True)
        paths = sorted((tmp_path / "metadata").glob("*.gi.json"))
        loaded = load_gi_artifacts(paths, validate=True, strict=False)
        bundle = export_merged_json(loaded, output_dir=tmp_path)
        validated = build_gi_corpus_bundle_output(bundle)
        assert validated.artifact_count == 2
        assert validated.insight_count_total >= 2
        assert validated.quote_count_total >= 2

    def test_export_ndjson_lines(self, tmp_path):
        """NDJSON emits one object per line with _artifact_path."""
        (tmp_path / "metadata").mkdir()
        write_artifact(
            tmp_path / "metadata" / "one.gi.json",
            build_artifact(
                "ep:1",
                "Text.",
                prompt_version="v1",
                insight_texts=["A real insight extracted from the transcript."],
            ),
            validate=True,
        )
        paths = list((tmp_path / "metadata").glob("*.gi.json"))
        loaded = load_gi_artifacts(paths, validate=True, strict=False)
        lines: list[str] = []

        def _w(s: str) -> None:
            lines.append(s)

        export_ndjson(loaded, output_dir=tmp_path, stream_write=_w)
        assert len(lines) == 1
        row = json.loads(lines[0].strip())
        assert "_artifact_path" in row
        assert row.get("episode_id") == "ep:1"
