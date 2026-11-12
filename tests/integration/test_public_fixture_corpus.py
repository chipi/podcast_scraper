"""The public fixture corpus (E04 step 8): current with ``v3``, holding nothing the private
extension writes, internally consistent, and read by public's readers and API.

``scripts/tools/build_public_fixture_corpus.py`` derives it; its docstring states the rules.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration]

_REPO = Path(__file__).resolve().parents[2]
_PUBLIC = _REPO / "tests" / "fixtures" / "app-validation-corpus" / "public"
_V3 = _REPO / "tests" / "fixtures" / "app-validation-corpus" / "v3"


def test_the_committed_corpus_is_current_with_v3() -> None:
    script = _REPO / "scripts" / "tools" / "build_public_fixture_corpus.py"
    run = subprocess.run(
        [sys.executable, str(script), "--check"], capture_output=True, text=True, check=False
    )
    assert run.returncode == 0, run.stdout + run.stderr + "\nrun the script without --check"


def test_nothing_the_private_extension_writes() -> None:
    files = [p.relative_to(_PUBLIC) for p in _PUBLIC.rglob("*") if p.is_file()]
    assert files
    private = [
        p
        for p in files
        if p.name.endswith((".gi.json", ".bridge.json", ".speakers.diagnostics.json"))
        or ".adfree." in p.name
        or p.parts[0] in ("search", ".viewer")
        or ("enrichments" in p.parts[:-1] and p.parts[0] != "enrichments")
    ]
    assert not private, private[:5]
    corpus_enrichers = {p.stem for p in (_PUBLIC / "enrichments").glob("*.json")} - {"run_summary"}
    assert corpus_enrichers == {"topic_cooccurrence_corpus"}
    for meta in _PUBLIC.rglob("*.metadata.json"):
        content = json.loads(meta.read_text(encoding="utf-8")).get("content") or {}
        assert "speakers" not in content and "diarization_num_speakers" not in content, meta


def test_transcripts_are_plain_and_segments_still_slice_their_text() -> None:
    names = set()
    for meta in _V3.rglob("*.metadata.json"):
        for s in (json.loads(meta.read_text()).get("content") or {}).get("speakers") or []:
            names.add(s["name"])
    prefix = re.compile(r"^(%s): " % "|".join(re.escape(n) for n in sorted(names)), re.M)
    ranged = 0
    for seg in _PUBLIC.rglob("*.segments.json"):
        text = Path(str(seg)[: -len(".segments.json")] + ".txt").read_text(encoding="utf-8")
        assert not prefix.search(text), seg
        rows = json.loads(seg.read_text(encoding="utf-8"))
        rows = rows if isinstance(rows, list) else rows.get("segments", [])
        for row in rows:
            assert "speaker_label" not in row, seg
            if "char_start" in row:
                ranged += 1
                assert text[row["char_start"] : row["char_end"]] == row["text"], (seg, row)
    # Every ranged segment v3 has outside its ad-free files (which the public corpus drops).
    in_v3 = sum(
        "char_start" in row
        for seg in _V3.rglob("*.segments.json")
        if ".adfree." not in seg.name
        for row in json.loads(seg.read_text(encoding="utf-8"))
    )
    assert ranged == in_v3 and ranged > 1000, (ranged, in_v3)


def test_the_kg_has_no_seated_people_and_validates() -> None:
    from podcast_scraper.kg.schema import validate_artifact

    for kg_path in _PUBLIC.rglob("*.kg.json"):
        kg = json.loads(kg_path.read_text(encoding="utf-8"))
        # Minimal validation: the strict 2.x schema refuses v3's own fixture KGs (``topic_source``).
        validate_artifact(kg, strict=False)
        roles = {
            (n.get("properties") or {}).get("role")
            for n in kg.get("nodes") or []
            if n.get("type") == "Person"
        }
        assert not roles & {"host", "guest", "speaker"}, kg_path


def test_public_readers_see_every_episode() -> None:
    from podcast_scraper.kg.corpus import newest_run_artifact_paths, scan_kg_artifact_paths
    from podcast_scraper.search.corpus_scope import discover_metadata_files
    from podcast_scraper.server.corpus_catalog import build_catalog_rows

    assert len(discover_metadata_files(_PUBLIC)) == 55
    assert len(build_catalog_rows(_PUBLIC)) == 55
    kg = newest_run_artifact_paths(_PUBLIC, scan_kg_artifact_paths(_PUBLIC), ".kg.json")
    assert len(kg) == 55
