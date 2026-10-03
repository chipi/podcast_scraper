"""m0016: a name published on a voice the corpus's ad signatures call an ad leaves every surface.

Shape from prod (2026-10-02): "Daniel Atkinson" (Wirecutter) seated as a guest on a voice that reads
the same ad in many shows, labelled in the segments, a guest Person in KG, the speaker of quotes in
GI, an identity in the bridge. The signatures are the BUILT ``search/ad_signatures.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from podcast_scraper.providers.ml.diarization import ad_signatures as ads
from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations.m0016_ad_reader_names_removed import (
    AdReaderNamesRemovedMigration,
    dry_run_report,
    NO_SIGNATURES_MESSAGE,
    undo,
)
from podcast_scraper.upgrade.registry import get_migrations

pytestmark = [pytest.mark.unit]

AD = "Daniel Atkinson"
AD_ID = "person:daniel-atkinson"
HOST = "Brian Winter"
AD_TEXT = (
    "this episode is brought to you by wirecutter the new york times product recommendation "
    "service that tests everything so you do not have to and you can trust what they say about "
    "the best vacuum the best mattress and the best headphones for the money this year"
)
HOST_TEXT = (
    "welcome to the show and thank you for joining me today because we are going to talk about "
    "what happened in the election and why the numbers that you saw this week are not what they "
    "seem to be at all and that is what this conversation is about "
) * 4
GERMAN_TEXT = (
    "das ist ein sehr guter tag und wir sind auch noch hier und ich habe nicht gesagt dass der "
    "mann die frau mit dem hund sieht und sich dabei nicht freut für eine lange zeit auf dem berg"
)


def _w(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _r(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _signatures(root: Path, recurring_text: str = "", ad_languages: Optional[List[str]] = None):
    doc = {
        "schema_version": ads.SCHEMA_VERSION,
        "built_at": "2026-10-02T09:00:00Z",
        "ad_languages": ad_languages or [],
        "recurring": sorted(ads.shingles(ads.words(recurring_text))),
    }
    _w(root / "search" / ads.FILENAME, doc)


def _episode(
    root: Path,
    ad_text: str = AD_TEXT,
    ad_name_also_real: bool = False,
    stated: str = "",
    ad_label: str = AD,
) -> Dict[str, Path]:
    run = root / "feeds" / "show" / "run_1"
    meta = run / "metadata"
    stem = "ep1"
    paths = {
        "meta": meta / f"{stem}.metadata.json",
        "kg": meta / f"{stem}.kg.json",
        "gi": meta / f"{stem}.gi.json",
        "bridge": meta / f"{stem}.bridge.json",
        "seg": run / "transcripts" / f"{stem}.segments.json",
        "adfree": run / "transcripts" / f"{stem}.adfree.segments.json",
    }
    speakers: List[Dict[str, Any]] = [
        {"id": "host", "name": HOST, "role": "host", "placed": True, "voices": ["SPEAKER_00"]},
        {"id": "guest", "name": AD, "role": "guest", "placed": True, "voices": ["SPEAKER_01"]},
    ]
    segs: List[Dict[str, Any]] = [
        {"start": 0.0, "speaker": "SPEAKER_00", "speaker_label": HOST, "text": HOST_TEXT},
        {"start": 9.0, "speaker": "SPEAKER_01", "speaker_label": AD, "text": ad_text},
    ]
    if ad_name_also_real:
        # the name also sits on a real voice of the episode, as a host's does on a German ad
        speakers[1]["voices"] = ["SPEAKER_01", "SPEAKER_02"]
        segs.append(
            {"start": 20.0, "speaker": "SPEAKER_02", "speaker_label": AD, "text": HOST_TEXT}
        )
    segs[1]["speaker_label"] = ad_label
    if ad_label != AD:
        speakers.pop(1)
    if stated == "roster_source":
        speakers[1]["source"] = "known_hosts"
    _w(
        paths["meta"],
        {
            "episode": {"episode_id": "ep1"},
            "feed": {"title": "Show"},
            "content": {
                "transcript_file_path": f"transcripts/{stem}.txt",
                "speakers": speakers,
                "detected_hosts": [HOST] + ([AD] if stated == "detected_hosts" else []),
                "detected_guests": [],
            },
        },
    )
    if stated == "diagnostics":
        _w(
            run / "transcripts" / f"{stem}.speakers.diagnostics.json",
            {"tried": {"known_hosts": [AD]}},
        )
    _w(paths["seg"], segs)
    _w(paths["adfree"], segs)
    ad_name, ad_pid = AD, AD_ID
    _w(
        paths["kg"],
        {
            "nodes": [
                {"id": ad_pid, "type": "Person", "properties": {"name": ad_name, "role": "guest"}},
                {"id": "person:other", "type": "Person", "properties": {"name": "Other One"}},
            ],
            "edges": [{"type": "GUESTS_ON", "from": ad_pid, "to": "podcast:show"}],
        },
    )
    _w(
        paths["gi"],
        {
            "nodes": [
                {"id": ad_pid, "type": "Person", "properties": {"name": ad_name}},
                {"id": "quote:1", "type": "Quote", "properties": {"speaker_id": ad_pid}},
                {
                    "id": "insight:1",
                    "type": "Insight",
                    "properties": {
                        "speaker": ad_name,
                        "tier": 3,
                        "grounded": True,
                        "surfaceable": True,
                        "routing_tag": "surface",
                    },
                },
            ],
            "edges": [
                {"type": "SPOKEN_BY", "from": "quote:1", "to": ad_pid},
                {"type": "SUPPORTED_BY", "from": "insight:1", "to": "quote:1"},
            ],
        },
    )
    _w(
        paths["bridge"],
        {"identities": [{"id": ad_pid, "type": "person", "display_name": ad_name}]},
    )
    return paths


@pytest.fixture
def corpus(tmp_path: Path) -> Dict[str, Path]:
    _signatures(tmp_path, AD_TEXT)
    paths = _episode(tmp_path)
    paths["root"] = tmp_path
    return paths


def _snapshot(corpus: Dict[str, Path]) -> Dict[str, bytes]:
    return {k: p.read_bytes() for k, p in corpus.items() if k != "root" and p.is_file()}


def _run(root: Path, dry: bool = False, **options: Any):
    ctx = MigrationContext(corpus_root=root, dry_run=dry, options=options)
    return AdReaderNamesRemovedMigration().apply(ctx)


def test_plan_lists_the_hit(corpus: Dict[str, Path]) -> None:
    msg = AdReaderNamesRemovedMigration().plan(MigrationContext(corpus_root=corpus["root"]))
    assert "1 hit(s) ['Daniel Atkinson']" in msg and "1 episode(s)" in msg


def test_every_surface_stops_naming_the_ad_reader(corpus: Dict[str, Path]) -> None:
    result = _run(corpus["root"])
    assert result.details["episodes"] == 1
    assert result.details["hits"] == [
        ["feeds/show/run_1/metadata/ep1.metadata.json", "SPEAKER_01", AD]
    ]
    content = _r(corpus["meta"])["content"]
    assert [s["name"] for s in content["speakers"]] == [HOST]
    assert content["detected_guests"] == []
    for key in ("seg", "adfree"):
        rows = _r(corpus[key])
        assert "speaker_label" not in rows[1] and rows[1]["voice_type"] == "commercial"
        assert rows[1]["speaker"] == "SPEAKER_01"
        assert rows[0]["speaker_label"] == HOST and "voice_type" not in rows[0]
    kg = _r(corpus["kg"])
    assert AD_ID not in {n["id"] for n in kg["nodes"]} and kg["edges"] == []
    nodes = {n["id"]: n for n in _r(corpus["gi"])["nodes"]}
    assert AD_ID not in nodes
    assert nodes["quote:1"]["properties"]["speaker_id"] is None
    assert nodes["quote:1"]["properties"]["speaker_voice_type"] == "commercial"
    ins = nodes["insight:1"]["properties"]
    assert "speaker" not in ins and ins["surfaceable"] is False and ins["routing_tag"] == "connect"
    assert ins["speaker_voice_type"] == "commercial"
    assert _r(corpus["bridge"])["identities"] == []


def test_receipt_header_freezes_the_set_and_the_signature_identity(corpus: Dict[str, Path]) -> None:
    _run(corpus["root"])
    header = json.loads(
        (corpus["root"] / "ad_reader_names_removed.jsonl").read_text().splitlines()[0]
    )
    assert header["kind"] == "header"
    assert header["hits"] == [["feeds/show/run_1/metadata/ep1.metadata.json", "SPEAKER_01", AD]]
    assert header["signatures"]["built_at"] == "2026-10-02T09:00:00Z"
    assert len(header["signatures"]["sha256"]) == 64


def test_a_name_that_survives_on_a_real_voice_loses_only_the_ad_voice(tmp_path: Path) -> None:
    """The name is also on a real voice of the episode: the person stays, the ad voice loses it."""
    _signatures(tmp_path, AD_TEXT)
    paths = _episode(tmp_path, ad_name_also_real=True)
    _run(tmp_path)
    rows = _r(paths["seg"])
    assert "speaker_label" not in rows[1] and rows[1]["voice_type"] == "commercial"
    assert rows[2]["speaker_label"] == AD
    entry = [s for s in _r(paths["meta"])["content"]["speakers"] if s["name"] == AD][0]
    assert entry["voices"] == ["SPEAKER_02"]
    assert AD_ID in {n["id"] for n in _r(paths["kg"])["nodes"]}


@pytest.mark.parametrize("stated", ["detected_hosts", "diagnostics", "roster_source"])
def test_a_stated_name_on_a_recurring_voice_is_not_a_hit(tmp_path: Path, stated: str) -> None:
    """A host reads the same promo every week, so their own voice recurs. Nobody stated a reader."""
    _signatures(tmp_path, AD_TEXT)
    paths = _episode(tmp_path, stated=stated)
    before = _snapshot(paths)
    result = _run(tmp_path)
    assert result.details["hits"] == [] and result.details["episodes"] == 0
    assert _snapshot(paths) == before
    sigs = ads.AdSignatures(
        recurring=frozenset(ads.shingles(ads.words(AD_TEXT))), ad_languages=frozenset()
    )
    report = dry_run_report(tmp_path, sigs)
    assert report["hits_by_name"] == {} and report["excluded_stated_by_name"] == {AD: 1}


def test_an_unstated_name_on_a_recurring_voice_is_a_hit(tmp_path: Path) -> None:
    _signatures(tmp_path, AD_TEXT)
    _episode(tmp_path)
    sigs = ads.AdSignatures(
        recurring=frozenset(ads.shingles(ads.words(AD_TEXT))), ad_languages=frozenset()
    )
    report = dry_run_report(tmp_path, sigs)
    assert report["hits_by_name"] == {AD: 1} and report["episodes"] == 1
    assert report["excluded_stated_by_name"] == {} and report["language_only_by_name"] == {}


@pytest.mark.parametrize("label", ["SPEAKER_01", "SPEAKER_00", "Host"])
def test_a_raw_voice_id_label_is_not_a_hit(tmp_path: Path, label: str) -> None:
    _signatures(tmp_path, AD_TEXT)
    paths = _episode(tmp_path, ad_label=label)
    before = _snapshot(paths)
    assert _run(tmp_path).details["hits"] == []
    assert _snapshot(paths) == before


def test_dry_run_report_lists_language_only_by_name(tmp_path: Path) -> None:
    _episode(tmp_path, ad_text=GERMAN_TEXT)
    sigs = ads.AdSignatures(recurring=frozenset(), ad_languages=frozenset({"de"}))
    report = dry_run_report(tmp_path, sigs)
    assert report["hits_by_name"] == {} and report["language_only_by_name"] == {AD: 1}


def test_second_run_is_a_no_op(corpus: Dict[str, Path]) -> None:
    _run(corpus["root"])
    after = _snapshot(corpus)
    assert _run(corpus["root"]).details["episodes"] == 0
    assert _snapshot(corpus) == after


def test_dry_run_writes_nothing(corpus: Dict[str, Path]) -> None:
    before = _snapshot(corpus)
    result = _run(corpus["root"], dry=True)
    assert result.details["totals"]["roster_entries"] == 1 and result.dry_run
    assert _snapshot(corpus) == before
    assert not (corpus["root"] / "ad_reader_names_removed.jsonl").exists()
    assert not (corpus["root"] / ".podcast_scraper").exists()


def test_no_signatures_file_is_a_clean_no_op(tmp_path: Path) -> None:
    paths = _episode(tmp_path)
    before = _snapshot(paths)
    ctx = MigrationContext(corpus_root=tmp_path)
    migration = AdReaderNamesRemovedMigration()
    result = migration.apply(ctx)
    assert result.applied and result.message == NO_SIGNATURES_MESSAGE
    assert migration.plan(ctx) == NO_SIGNATURES_MESSAGE
    assert migration.verify(ctx) == (True, NO_SIGNATURES_MESSAGE)
    assert _snapshot(paths) == before
    assert not (tmp_path / "ad_reader_names_removed.jsonl").exists()


def test_language_only_hit_is_excluded_by_default_and_listed(tmp_path: Path) -> None:
    _signatures(tmp_path, "", ["de"])
    paths = _episode(tmp_path, ad_text=GERMAN_TEXT)
    before = _snapshot(paths)
    migration = AdReaderNamesRemovedMigration()
    result = _run(tmp_path)
    assert result.details["hits"] == []
    assert result.details["language_only"] == [
        ["feeds/show/run_1/metadata/ep1.metadata.json", "SPEAKER_01", AD]
    ]
    assert "EXCLUDED" in result.message
    assert _snapshot(paths) == before
    assert "1 language_only hit(s) held for hand review" in migration.plan(
        MigrationContext(corpus_root=tmp_path)
    )
    included = _run(tmp_path, include_language_only=True)
    assert included.details["episodes"] == 1
    assert _r(paths["seg"])[1]["voice_type"] == "commercial"


def test_language_only_env_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _signatures(tmp_path, "", ["de"])
    paths = _episode(tmp_path, ad_text=GERMAN_TEXT)
    monkeypatch.setenv("M0016_INCLUDE_LANGUAGE_ONLY", "1")
    assert _run(tmp_path).details["episodes"] == 1
    assert "speaker_label" not in _r(paths["seg"])[1]


def test_verify_fails_before_apply_and_passes_after(corpus: Dict[str, Path]) -> None:
    ctx = MigrationContext(corpus_root=corpus["root"])
    migration = AdReaderNamesRemovedMigration()
    ok, msg = migration.verify(ctx)
    assert not ok and "not applied yet" in msg
    _run(corpus["root"])
    ok, msg = migration.verify(ctx)
    assert ok, msg


def test_verify_judges_the_frozen_set_not_a_rebuilt_file(corpus: Dict[str, Path]) -> None:
    """A later rebuild that no longer flags the voice must not turn verify green or red."""
    _run(corpus["root"])
    _signatures(corpus["root"], "", [])
    ok, msg = AdReaderNamesRemovedMigration().verify(MigrationContext(corpus_root=corpus["root"]))
    assert ok, msg


def test_undo_restores_byte_for_byte_and_verify_then_fails(corpus: Dict[str, Path]) -> None:
    before = _snapshot(corpus)
    _run(corpus["root"])
    restored, refused = undo(corpus["root"])
    assert refused == [] and restored == 6
    assert _snapshot(corpus) == before
    ctx = MigrationContext(corpus_root=corpus["root"])
    ok, _msg = AdReaderNamesRemovedMigration().verify(ctx)
    assert not ok


def test_registered_after_0015() -> None:
    ids = [m.id for m in get_migrations()]
    assert ids.index("0016_ad_reader_names_removed") == (
        ids.index("0015_unpublishable_speaker_names_removed") + 1
    )
