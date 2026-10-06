"""#2290: an episode the corpus already holds costs a guid lookup and nothing else.

Nightly 4d62d1ab (2026-10-06) made 562 speaker-detection LLM calls for 12 processed episodes:
``prepare_episode_download_args`` detected speakers for every selected episode, and only the
download step afterwards found 538 of them already present and threw the names away.

Two things are pinned here. The up-front check must stop all per-episode work for a present
episode (and only for it). And it must never claim "skip" where the download step would NOT
skip -- a wrong up-front skip silently drops an episode the run was meant to (re)process -- so
each early-skip answer is checked against the download step's own decision on the same state.
"""

from __future__ import annotations

import json
import queue
import typing
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper import Config, models
from podcast_scraper.workflow import episode_processor, run_index
from podcast_scraper.workflow.helpers import get_episode_id_from_episode
from podcast_scraper.workflow.stages import processing

pytestmark = pytest.mark.unit

FEED_URL = "https://example.com/podcast.xml"
VTT = [("https://example.com/t.vtt", "text/vtt")]


@pytest.fixture(autouse=True)
def _reset_index_cache():
    run_index.reset_corpus_metadata_index_cache_for_tests()
    yield
    run_index.reset_corpus_metadata_index_cache_for_tests()


def _episode(guid: str, idx: int, transcript_urls=None) -> models.Episode:
    item = ET.Element("item")
    ET.SubElement(item, "title").text = f"Ep {guid}"
    ET.SubElement(item, "guid").text = guid
    ep = models.Episode(
        idx=idx,
        title=f"Ep {guid}",
        title_safe=f"Ep {guid}",
        item=item,
        transcript_urls=list(transcript_urls or []),
    )
    ep.media_url = f"https://example.com/{guid}.mp3"
    return ep


def _cfg(tmp_path: Path, **update) -> Config:
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    cfg = Config(
        rss=FEED_URL,
        output_dir=str(corpus),
        skip_existing=True,
        single_feed_uses_corpus_layout=True,
        transcribe_missing=True,
    )
    return cfg.model_copy(update=update) if update else cfg


def _seed(feed_dir: Path, guid: str, idx: int = 1, *, segments: bool = True) -> None:
    run_dir = feed_dir / "run_20260101-000000_priorAA"
    name = f"{idx:04d} - Ep {guid}"
    (run_dir / "transcripts").mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (run_dir / "transcripts" / f"{name}.txt").write_text("hello", encoding="utf-8")
    if segments:
        (run_dir / "transcripts" / f"{name}.segments.json").write_text("[]", encoding="utf-8")
    eid, _ = get_episode_id_from_episode(_episode(guid, idx), FEED_URL)
    (run_dir / "metadata" / f"{name}.metadata.json").write_text(
        json.dumps(
            {
                "episode": {"guid": guid, "episode_id": eid},
                "content": {"transcript_file_path": f"transcripts/{name}.txt"},
            }
        ),
        encoding="utf-8",
    )


def _fresh_run(cfg: Config) -> Path:
    fresh = Path(str(cfg.output_dir)) / "run_20260102-000000_freshBB"
    fresh.mkdir(parents=True, exist_ok=True)
    return fresh


def _evidence(cfg: Config, ep, fresh: Path, temp_dir: str | None = "tmp") -> str | None:
    return episode_processor.presence_skip_evidence(ep, cfg, str(fresh), None, temp_dir)


# --- the up-front answer agrees with the download step, everywhere ---------------------------


def _download_step_skipped(cfg, ep, fresh: Path, temp_dir) -> bool:
    """Run the REAL download step on *ep*; True when it recorded a skip-existing skip.

    Network fetches are stubbed to fail, so an episode the step would process shows up as an
    attempted fetch rather than a skip.
    """
    metrics = MagicMock()
    metrics.stage_did_run.return_value = True
    with (
        patch.object(episode_processor, "_download_or_reuse_media", return_value=(False, 0, 0)),
        patch.object(episode_processor, "_fetch_transcript_content", return_value=None),
    ):
        episode_processor.process_episode_download(
            ep, cfg, temp_dir, str(fresh), None, queue.Queue(), None, pipeline_metrics=metrics
        )
    return any(
        c.kwargs.get("status") == "skipped" and c.kwargs.get("error_type") == "SkipExisting"
        for c in metrics.update_episode_status.call_args_list
    )


# Every stage the config accepts, read from the model: a stage added later is covered without
# anyone remembering to list it here (#2260 adds translate_only).
_STAGES = list(typing.get_args(Config.model_fields["pipeline_stage"].annotation))
_VARIANTS = {
    "plain": {},
    "reprocess-ids": {"reprocess_episode_ids": ["gA"]},
    "no-skip-existing": {"skip_existing": False},
    "backfill": {"backfill_transcript_segments": True, "generate_gi": True},
    "summaries": {"generate_summaries": True},
    "no-transcription": {"transcribe_missing": False},
}


@pytest.mark.parametrize("present", [True, False], ids=["present", "new"])
@pytest.mark.parametrize("publisher", [False, True], ids=["audio", "publisher"])
@pytest.mark.parametrize("variant", sorted(_VARIANTS))
@pytest.mark.parametrize("stage", _STAGES)
def test_the_early_answer_is_the_download_steps_answer(
    tmp_path, stage, variant, publisher, present
):
    cfg = _cfg(tmp_path, pipeline_stage=stage, **_VARIANTS[variant])
    _seed(Path(str(cfg.output_dir)), "gA", segments=variant != "backfill")
    fresh = _fresh_run(cfg)
    if variant == "summaries":
        # A transcript in THIS run dir: the reuse branch hands it to summarization.
        (fresh / "transcripts").mkdir()
        (fresh / "transcripts" / "0001 - Ep gA.vtt").write_text("WEBVTT", encoding="utf-8")
    temp_dir = str(tmp_path / "tmp")
    guid = "gA" if present else "gNEW"
    ep = _episode(guid, 1, transcript_urls=VTT if publisher else None)

    early = _evidence(cfg, ep, fresh, temp_dir=temp_dir) is not None
    run_index.reset_corpus_metadata_index_cache_for_tests()
    later = _download_step_skipped(
        cfg, _episode(guid, 1, VTT if publisher else None), fresh, temp_dir
    )

    assert early == later


def test_the_nightly_case_is_skipped_early_on_both_routes(tmp_path):
    """The case #2290 exists for: corpus layout, skip-existing, plain full run, present."""
    cfg = _cfg(tmp_path)
    _seed(Path(str(cfg.output_dir)), "gA")
    fresh = _fresh_run(cfg)
    assert _evidence(cfg, _episode("gA", 1), fresh) is not None
    assert _evidence(cfg, _episode("gA", 1, transcript_urls=VTT), fresh) is not None
    assert _evidence(cfg, _episode("gNEW", 2), fresh) is None


# --- the ordering: nothing per-episode runs before identification ----------------------------


def test_prepare_runs_no_per_episode_work_for_a_present_episode(tmp_path):
    cfg = _cfg(tmp_path)
    _seed(Path(str(cfg.output_dir)), "gA")
    fresh = _fresh_run(cfg)
    present, new = _episode("gA", 1), _episode("gNEW", 2)
    metrics = MagicMock()
    resources = SimpleNamespace(
        temp_dir=str(tmp_path / "tmp"), transcription_jobs=None, transcription_jobs_lock=None
    )

    with (
        patch.object(
            processing, "_check_episode_size_skip", return_value=processing._NO_SIZE_SKIP
        ) as size_probe,
        patch.object(processing, "_detect_speakers_for_episode", return_value=None) as detect,
    ):
        args = processing.prepare_episode_download_args(
            [present, new],
            cfg,
            str(fresh),
            None,
            resources,
            SimpleNamespace(cached_hosts=set()),
            metrics,
        )

    assert [a[0] for a in args] == [new]
    assert [c.args[0] for c in detect.call_args_list] == [new]
    assert [c.args[1] for c in size_probe.call_args_list] == [new]
    skipped = [
        c.kwargs
        for c in metrics.update_episode_status.call_args_list
        if c.kwargs["status"] == "skipped"
    ]
    assert len(skipped) == 1 and skipped[0]["error_type"] == "SkipExisting"
