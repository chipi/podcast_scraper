"""One guest must arrive in the app as ONE person, not one per ASR garble.

The v3 fixtures deliberately mis-hear guest names ("Skanda Amarnath" also appears as
"Skanda Amarnauth" and "Skanda Eminas"). Without folding those onto the canonical name at corpus
build time, the knowledge graph carries a separate ``person:`` node per spelling and the show page
lists the same guest three times.

The fold is authored ground truth from the manifest — deliberately NOT a similarity match.
``speaker_detectors/name_canonicalization.py`` documents why the heuristic version is unsafe
(#876: a wrong name is worse than a garble left alone), so the safety property asserted here is
that two DIFFERENT guests who share a first name are never merged.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "build_app_validation_corpus.py"
_spec = importlib.util.spec_from_file_location("_bavc", _SCRIPT)
assert _spec and _spec.loader
_bavc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bavc)

_MANIFEST = (
    Path(__file__).resolve().parents[2] / "fixtures" / "ground-truth" / "v3" / "manifest.json"
)


def _write_manifest(tmp_path: Path, guests: list[dict]) -> Path:
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"podcasts": [{"pod_id": "p01", "guests": guests}]}), encoding="utf-8")
    return p


class TestLoadNameVariants:
    def test_folds_garbles_severe_garbles_and_nicknames(self, tmp_path: Path) -> None:
        m = _write_manifest(
            tmp_path,
            [
                {
                    "canonical_name": "Skanda Amarnath",
                    "garble_variants": ["Skanda Amarnauth"],
                    "severe_garble": "Skanda Eminas",
                    "nickname_variants": [],
                },
                {
                    "canonical_name": "Richard Clarida",
                    "garble_variants": [],
                    "severe_garble": None,
                    "nickname_variants": ["Rich Clarida"],
                },
            ],
        )
        out = _bavc._load_name_variants(m)
        assert out["skanda amarnauth"] == "Skanda Amarnath"
        assert out["skanda eminas"] == "Skanda Amarnath"
        assert out["rich clarida"] == "Richard Clarida"
        # The canonical spelling is not a key — it needs no rewrite.
        assert "skanda amarnath" not in out

    def test_excludes_alias_invention(self, tmp_path: Path) -> None:
        """An invented-but-plausible wrong name is a DIFFERENT authored failure mode.

        Folding it away would erase the very case it exists to represent.
        """
        m = _write_manifest(
            tmp_path,
            [
                {
                    "canonical_name": "Liam Verbeek",
                    "alias_invention": "Liam Vandermeer",
                    "garble_variants": ["Liam Verbeck"],
                    "severe_garble": None,
                    "nickname_variants": [],
                }
            ],
        )
        out = _bavc._load_name_variants(m)
        assert out["liam verbeck"] == "Liam Verbeek"
        assert "liam vandermeer" not in out

    def test_raises_rather_than_guessing_on_an_ambiguous_variant(self, tmp_path: Path) -> None:
        m = _write_manifest(
            tmp_path,
            [
                {"canonical_name": "Ann Smith", "garble_variants": ["A Smith"]},
                {"canonical_name": "Anna Smith", "garble_variants": ["A Smith"]},
            ],
        )
        with pytest.raises(ValueError, match="ambiguous"):
            _bavc._load_name_variants(m)

    def test_missing_manifest_is_empty_not_an_error(self, tmp_path: Path) -> None:
        assert _bavc._load_name_variants(tmp_path / "nope.json") == {}


class TestCanonicalizePersonsIn:
    def test_collapses_variant_nodes_onto_one_canonical_id(self) -> None:
        doc: dict[str, Any] = {
            "nodes": [
                {
                    "id": "person:speaker-01",
                    "type": "Person",
                    "properties": {"name": "Skanda Amarnath"},
                },
                {
                    "id": "person:speaker-02",
                    "type": "Person",
                    "properties": {"name": "Skanda Amarnauth"},
                },
                {
                    "id": "person:speaker-03",
                    "type": "Person",
                    "properties": {"name": "Skanda Eminas"},
                },
            ],
            "edges": [{"from": "person:speaker-02", "to": "topic:x"}],
        }
        _bavc._canonicalize_persons_in(
            doc,
            {"skanda amarnauth": "Skanda Amarnath", "skanda eminas": "Skanda Amarnath"},
        )
        ids = [n["id"] for n in doc["nodes"] if n["type"] == "Person"]
        assert ids == ["person:skanda-amarnath"]  # three nodes deduped to one
        assert doc["nodes"][0]["properties"]["name"] == "Skanda Amarnath"
        # The edge follows the surviving node rather than dangling.
        assert doc["edges"][0]["from"] == "person:skanda-amarnath"

    def test_never_merges_two_different_guests_sharing_a_first_name(self) -> None:
        """The safety property. #876: a wrong name is worse than a garble left alone."""
        doc: dict[str, Any] = {
            "nodes": [
                {"id": "person:speaker-01", "type": "Person", "properties": {"name": "Daniel Cho"}},
                {
                    "id": "person:speaker-02",
                    "type": "Person",
                    "properties": {"name": "Daniel Olufemi"},
                },
            ],
            "edges": [],
        }
        _bavc._canonicalize_persons_in(doc, {"daniel choh": "Daniel Cho"})
        ids = sorted(n["id"] for n in doc["nodes"] if n["type"] == "Person")
        assert ids == ["person:daniel-cho", "person:daniel-olufemi"]

    def test_no_variant_map_behaves_as_before(self) -> None:
        doc: dict[str, Any] = {
            "nodes": [
                {"id": "person:speaker-01", "type": "Person", "properties": {"name": "Jordan Park"}}
            ],
            "edges": [],
        }
        _bavc._canonicalize_persons_in(doc)
        assert doc["nodes"][0]["id"] == "person:jordan-park"


class TestShippedCorpusIsCanonical:
    """The committed corpus must not carry a garbled person id — that is what the app renders."""

    def test_no_authored_garble_survives_as_a_person_id(self) -> None:
        corpus = Path(__file__).resolve().parents[2] / "fixtures" / "app-validation-corpus" / "v3"
        if not corpus.is_dir() or not _MANIFEST.is_file():
            pytest.skip("validation corpus or manifest not present")
        variants = _bavc._load_name_variants(_MANIFEST)
        garbled_ids = {f"person:{_bavc.slug(v)}" for v in variants}
        canonical_ids = {f"person:{_bavc.slug(c)}" for c in variants.values()}
        # A garble whose slug collides with its own canonical form is already folded.
        garbled_ids -= canonical_ids
        assert garbled_ids, "expected the manifest to author at least one distinct garble"

        offenders: list[str] = []
        for p in list(corpus.rglob("*.json")) + list(corpus.rglob("*.jsonl")):
            if "lance_index" in p.parts:
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
            for gid in garbled_ids:
                if f'"{gid}"' in text:
                    offenders.append(f"{p.relative_to(corpus)}: {gid}")
        assert not offenders, "garbled person ids in the shipped corpus:\n" + "\n".join(
            sorted(offenders)[:20]
        )
