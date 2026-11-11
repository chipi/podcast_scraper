"""The item context a processing job carries (ADR-177 decision 7).

A hook that needs the item's type reads ``content_types.current_item()``; hooks gain no type
argument, so a podcast job, which carries no context, runs exactly as before.
"""

from __future__ import annotations

import inspect
import threading
from typing import Any, cast, List

import pytest

from podcast_scraper import content_types as ct
from podcast_scraper.workflow.stages import processing
from podcast_scraper.workflow.types import ProcessingJob

pytestmark = pytest.mark.unit

TEXT = ct.ContentType(
    name="test_text_item",
    id_scheme="item",
    stage_profile=ct.StageProfile(name="text", stages=frozenset({"summary", "kg"})),
    default_transport="fixture_feed",
)


def test_with_no_job_the_current_item_is_a_podcast() -> None:
    assert ct.current_item() is ct.PODCAST_ITEM
    assert ct.current_item().content_type is ct.PODCAST


def test_a_job_with_no_item_is_processed_as_a_podcast() -> None:
    with ct.processing_item(None) as item:
        assert item is ct.PODCAST_ITEM
        assert ct.current_item() is ct.PODCAST_ITEM


def test_an_item_is_current_for_the_block_and_reset_after() -> None:
    item = ct.ItemContext(content_type=TEXT, item_id="item:test_text_item:1", body_path="/b.txt")
    with ct.processing_item(item):
        assert ct.current_item() is item
        assert not ct.current_item().content_type.stage_profile.runs("diarization")
    assert ct.current_item() is ct.PODCAST_ITEM


def test_the_context_is_reset_when_the_block_raises() -> None:
    item = ct.ItemContext(content_type=TEXT)
    with pytest.raises(RuntimeError):
        with ct.processing_item(item):
            raise RuntimeError("stage failed")
    assert ct.current_item() is ct.PODCAST_ITEM


def test_another_thread_does_not_see_this_threads_item() -> None:
    seen: List[ct.ItemContext] = []
    with ct.processing_item(ct.ItemContext(content_type=TEXT)):
        t = threading.Thread(target=lambda: seen.append(ct.current_item()))
        t.start()
        t.join()
    assert seen == [ct.PODCAST_ITEM]


def test_a_processing_job_defaults_to_no_item() -> None:
    job = ProcessingJob(
        episode=cast(Any, object()),  # only the item's default is checked here
        transcript_path="t.txt",
        transcript_source="direct_download",
        detected_names=None,
        whisper_model=None,
    )
    assert job.item is None


def test_the_worker_sets_the_jobs_item_around_metadata_generation() -> None:
    # Structural: the per-job worker is nested in process_processing_jobs_concurrent. The text-in
    # entry point's run (a later commit) exercises it with a real item.
    src = inspect.getsource(processing.process_processing_jobs_concurrent)
    with_block = src.split("content_types.processing_item(", 1)
    assert len(with_block) == 2, "the worker no longer sets the item context"
    assert 'getattr(job, "item", None)' in with_block[1].split("\n", 1)[0]
    assert "metadata_stage.call_generate_metadata(" in with_block[1].split("return True", 1)[0]
