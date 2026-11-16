"""Gate b of the core release: podcast output is unchanged with a second content type installed.

ADR-177 adds content types, a text-in entry point, stage profiles and a KG 3.0 writer for new
types; ADR-178 keeps podcasts on KG 2.x until R3b. Installing another type must therefore change
nothing a podcast run writes. This runs the same podcast feed twice through ``run_pipeline`` (the
real HTTP client against the local test server, metadata and KG on, no model), once with no
extension and once with the test-only text type and its transport registered, and compares every
file the two runs wrote. Only what differs between ANY two runs is normalised away: the output
directory, the run's id and suffix, and wall-clock times and durations.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from podcast_scraper import content_types as ct
from podcast_scraper.extensions import Extension, use_extensions
from tests.conftest import create_test_config

pytestmark = [pytest.mark.e2e]

TEXT = ct.ContentType(
    name="test_text_item",
    id_scheme="item",
    stage_profile=ct.StageProfile(name="test_text_item", stages=frozenset({ct.SUMMARY, ct.KG})),
    default_transport="fixture_feed",
    source_kind="test",
)


def _list(source: object, cursor: Optional[str]) -> Tuple[List[ct.SourceItem], Optional[str]]:
    raise AssertionError("a podcast run must never list through the text type's transport")


TEXT_TYPE = Extension(
    name="test-text-type",
    content_types=lambda: [TEXT],
    transports=lambda: [ct.Transport(name="fixture_feed", list_items=_list)],
)


def _run(feed_url: str, out: Path, extensions: List[Extension]) -> Dict[str, bytes]:
    cfg = create_test_config(
        rss_url=feed_url,
        output_dir=str(out),
        max_episodes=2,
        generate_metadata=True,
        metadata_format="json",
        generate_summaries=False,
        generate_kg=True,
        kg_extraction_source="metadata_only",
        auto_speakers=False,
        transcribe_missing=False,
    )
    with use_extensions(extensions):
        count, _ = ct.run_source(cfg, None)
    assert count > 0
    files = {str(p.relative_to(out)): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}
    return _normalised(files, out)


def _normalised(files: Dict[str, bytes], out: Path) -> Dict[str, bytes]:
    """Strip what differs between any two runs: the output dir and the run's stamp."""
    runs = {k.split("/")[0] for k in files if k.startswith("run_")}
    assert len(runs) == 1, runs
    stamp = next(iter(runs))[len("run_") :].rsplit("_", 1)[0]
    norm: Dict[str, bytes] = {}
    for key, body in files.items():
        text = body.decode("utf-8", "replace")
        for old, new in ((str(out), "<OUT>"), (stamp, "<STAMP>")):
            key, text = key.replace(old, new), text.replace(old, new)
        if key.endswith(".json"):
            name = key.rsplit("/", 1)[-1]
            timing = name in _RUN_TIMING_FILES or name.endswith(".manifest.json")
            text = json.dumps(_drop_volatile(json.loads(text), timing), indent=1, sort_keys=True)
        norm[key] = text.encode("utf-8")
    return norm


#: Keys whose value is a wall-clock time or a run's own id or config hash (its output dir is
#: part of the config), never content.
_VOLATILE_KEYS = {
    "run_id",
    "created_at",
    "updated_at",
    "generated_at",
    "extracted_at",
    "emitted_at",
    "retrieved_at",
    "processing_timestamp",
    "config_sha256",
}
#: Timing figures, measured on this machine for this run; dropped only in the run-level files
#: that hold them (and the per-episode processing manifests' stage ``duration_s``, RFC-109), so
#: an episode's own ``duration_seconds`` is still compared.
_RUN_TIMING_FILES = {"metrics.json", "run.json"}
_TIMING = re.compile(r"(^time_|seconds|duration|_sec$|_ms$|latency|p50|p95|p99|elapsed)")


def _drop_volatile(value: Any, timing: bool) -> Any:
    if isinstance(value, dict):
        return {
            k: _drop_volatile(v, timing)
            for k, v in value.items()
            if k not in _VOLATILE_KEYS and not (timing and _TIMING.search(k))
        }
    if isinstance(value, list):
        return [_drop_volatile(v, timing) for v in value]
    return value


def _differences(a: Dict[str, bytes], b: Dict[str, bytes]) -> List[str]:
    out = [f"only in the first run: {k}" for k in sorted(set(a) - set(b))]
    out += [f"only in the second run: {k}" for k in sorted(set(b) - set(a))]
    for key in sorted(set(a) & set(b)):
        if a[key] != b[key]:
            la, lb = a[key].decode().splitlines(), b[key].decode().splitlines()
            first = next(((x, y) for x, y in zip(la, lb) if x != y), (len(la), len(lb)))
            out.append(f"{key}: {first}")
    return out


def test_two_plain_runs_are_identical_once_normalised(e2e_server, tmp_path) -> None:
    """The control: the normaliser strips only what differs between ANY two runs. Without it, the
    gate below could pass because the normaliser hid a real difference."""
    feed = e2e_server.urls.feed("podcast1_multi_episode")
    a = _run(feed, tmp_path / "a", [])
    b = _run(feed, tmp_path / "b", [])
    assert any(k.endswith(".kg.json") for k in a) and any(k.endswith(".metadata.json") for k in a)
    assert _differences(a, b) == []


def test_a_podcast_run_is_unchanged_with_the_text_type_installed(e2e_server, tmp_path) -> None:
    """Gate b: the same feed, with and without the test-only text type registered."""
    feed = e2e_server.urls.feed("podcast1_multi_episode")
    plain = _run(feed, tmp_path / "plain", [])
    with use_extensions([TEXT_TYPE]):
        assert ct.content_type("test_text_item") is TEXT  # the type really is installed
    with_type = _run(feed, tmp_path / "with_type", [TEXT_TYPE])
    assert _differences(plain, with_type) == []
    for key, body in with_type.items():
        if key.endswith((".metadata.json", ".kg.json")):
            assert b"content_type" not in body, key  # podcasts write no type until R3b
