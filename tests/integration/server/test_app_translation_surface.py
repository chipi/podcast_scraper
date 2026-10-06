"""The translated-episode API surface (S2.8): `?lang=`, the translation fields, the status.

WHAT THIS IS FOR. There is no user-visible "translated from X" marker in v1 (D-36), so the API
has to at least MAKE THE FACTS AVAILABLE: which language is being served, whether it came from
a model, which model, and — on detail — why a non-English episode might have no insights. A
client that has to infer any of that from the text or from absences will infer it wrongly.

Every field is ADDITIVE, so an existing client sees exactly today's response.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.integration.server.test_app_episodes import _client, _only_slug, _write_corpus

pytestmark = [pytest.mark.integration]

STEM = "0001-hello"


def _make_translated(root: Path, *, status: str = "translated", complete: bool = True) -> None:
    """Swap the episode over and write the ledger, as the translation stage would (D-44).

    A COMPLETE translation means the canonical body holds English and the SOURCE is kept at its
    language-tagged name — that tagged file is what the completeness gate reads, because only the
    atomic swap creates it. An incomplete one never swapped: the canonical body is still the source
    and there is no tagged sibling, which is why `complete=False` writes neither.
    """
    tr = root / "transcripts"
    if complete:
        # The source moves aside. Written UNCONDITIONALLY: this is simulating a COMPLETED swap, so
        # the tagged source must exist whatever the fixture laid down before — and `_write_corpus`
        # writes only the segments sidecar, not the transcript body, so a copy-if-present guard
        # silently produced an episode the completeness gate read as never-translated.
        canon = tr / f"{STEM}.txt"
        (tr / f"{STEM}.es.txt").write_text(
            canon.read_text(encoding="utf-8") if canon.is_file() else "Alice: Hola mundo.\n",
            encoding="utf-8",
        )
        seg = tr / f"{STEM}.segments.json"
        if seg.is_file():
            (tr / f"{STEM}.es.segments.json").write_text(
                seg.read_text(encoding="utf-8"), encoding="utf-8"
            )
        # ...and the translation takes the canonical names
        (tr / f"{STEM}.txt").write_text("Alice: Hello world.\nBob: Second.\n", encoding="utf-8")
        (tr / f"{STEM}.segments.json").write_text(
            json.dumps(
                [
                    {
                        "id": 0,
                        "start": 0.0,
                        "end": 2.5,
                        "text": "Hello world.",
                        "speaker_label": "Alice",
                        "unit_id": "t0000.u01",
                        "sent_id": "t0000.s01",
                    },
                    {
                        "id": 1,
                        "start": 2.5,
                        "end": 5.0,
                        "text": "Second.",
                        "speaker_label": "Bob",
                        "unit_id": "t0001.u01",
                        "sent_id": "t0001.s01",
                    },
                ]
            ),
            encoding="utf-8",
        )
    (tr / f"{STEM}.translation.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "status": status,
                "source_language": "es",
                "model": "google/translategemma-12b-it",
                "units": [],
            }
        ),
        encoding="utf-8",
    )


class TestTheDefaultIsEnglish:
    def test_a_translated_episode_serves_english_by_default(self, tmp_path: Path) -> None:
        """D-38, at the API. The default is the precedence's, not this endpoint's — `?lang=`
        selects the alternative rather than deciding the default."""
        _write_corpus(tmp_path)
        _make_translated(tmp_path)
        slug = _only_slug(tmp_path)
        body = _client(tmp_path).get(f"/api/app/episodes/{slug}/segments").json()

        assert body["language"] == "en"
        assert body["machine_translated"] is True
        assert body["translation_model"] == "google/translategemma-12b-it"
        assert body["source_language"] == "es"
        assert [s["text"] for s in body["segments"]] == ["Hello world.", "Second."]

    def test_lang_selects_the_source_language(self, tmp_path: Path) -> None:
        _write_corpus(tmp_path)
        _make_translated(tmp_path)
        slug = _only_slug(tmp_path)
        body = _client(tmp_path).get(f"/api/app/episodes/{slug}/segments?lang=es").json()

        assert body["language"] == "es"
        assert body["machine_translated"] is False, "the source text is what was actually spoken"
        assert body["translation_model"] is None
        assert body["source_language"] == "es"

    def test_lang_en_is_explicit_and_matches_the_default(self, tmp_path: Path) -> None:
        _write_corpus(tmp_path)
        _make_translated(tmp_path)
        slug = _only_slug(tmp_path)
        body = _client(tmp_path).get(f"/api/app/episodes/{slug}/segments?lang=en").json()
        assert body["language"] == "en" and body["machine_translated"] is True


class TestHonestyAboutWhatWasServed:
    def test_an_english_episode_reports_no_translation(self, tmp_path: Path) -> None:
        """The path every one of the 678 existing episodes takes. The fields are present and
        false/null rather than absent, so a client needs no special case."""
        _write_corpus(tmp_path)
        slug = _only_slug(tmp_path)
        body = _client(tmp_path).get(f"/api/app/episodes/{slug}/segments").json()

        assert body["machine_translated"] is False
        assert body["translation_model"] is None
        assert body["source_language"] is None

    def test_asking_for_english_that_does_not_exist_reports_the_SOURCE(
        self, tmp_path: Path
    ) -> None:
        """The field says what was ACTUALLY served, not what was requested. Claiming `en` for
        source-language text is exactly the confusion that makes English NLP read Spanish."""
        _write_corpus(tmp_path)
        _make_translated(tmp_path, status="failed", complete=False)
        slug = _only_slug(tmp_path)
        body = _client(tmp_path).get(f"/api/app/episodes/{slug}/segments?lang=en").json()

        assert body["language"] == "es"
        assert body["machine_translated"] is False

    def test_a_withheld_translation_still_serves_the_source(self, tmp_path: Path) -> None:
        """RFC-124 §5.3 withholds the swap when units failed. The episode must still play."""
        _write_corpus(tmp_path)
        _make_translated(tmp_path, status="failed", complete=False)
        slug = _only_slug(tmp_path)
        resp = _client(tmp_path).get(f"/api/app/episodes/{slug}/segments")
        assert resp.status_code == 200
        assert [s["text"] for s in resp.json()["segments"]] == ["Hello world.", "Second."]


class TestTranslationStatusOnDetail:
    def test_a_failed_translation_is_reported_so_absent_insights_are_explicable(
        self, tmp_path: Path
    ) -> None:
        """Without this a client sees an episode with no insights and cannot tell whether the
        show is uninteresting, GI failed, or a translation did — three very different things."""
        _write_corpus(tmp_path)
        _make_translated(tmp_path, status="failed", complete=False)
        slug = _only_slug(tmp_path)
        body = _client(tmp_path).get(f"/api/app/episodes/{slug}").json()
        assert body["translation_status"] == "failed"

    def test_a_translated_episode_reports_translated(self, tmp_path: Path) -> None:
        _write_corpus(tmp_path)
        _make_translated(tmp_path, status="translated")
        slug = _only_slug(tmp_path)
        assert (
            _client(tmp_path).get(f"/api/app/episodes/{slug}").json()["translation_status"]
            == "translated"
        )

    def test_an_english_episode_reports_null(self, tmp_path: Path) -> None:
        """Null, not `skipped`: there is no translation record for an English episode, and
        inventing one would make the field unable to distinguish "never translated" from
        "translation decided there was nothing to do"."""
        _write_corpus(tmp_path)
        slug = _only_slug(tmp_path)
        assert (
            _client(tmp_path).get(f"/api/app/episodes/{slug}").json()["translation_status"] is None
        )

    def test_a_corrupt_ledger_does_not_break_the_endpoint(self, tmp_path: Path) -> None:
        _write_corpus(tmp_path)
        (tmp_path / "transcripts" / f"{STEM}.translation.json").write_text("{bad", "utf-8")
        slug = _only_slug(tmp_path)
        resp = _client(tmp_path).get(f"/api/app/episodes/{slug}")
        assert resp.status_code == 200
        assert resp.json()["translation_status"] is None
