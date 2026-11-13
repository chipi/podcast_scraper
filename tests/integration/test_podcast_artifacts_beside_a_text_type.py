"""Deploy gate 2 (E04): podcast artifacts are unchanged when a test-only type runs beside them.

The mixed-corpus guards (``test_mixed_corpus_guards``) cover the readers. This covers the writers:
the real batch entry (``cli.main`` with a feeds spec, the nightly's path) runs two podcast feeds
with only HTTP mocked, once alone and once with a source of the test-only text type in the same
spec. Every podcast artifact under ``feeds/`` must be the same set, and must differ between the
two runs only where two runs of the podcasts alone already differ (timestamps and paths that carry
them), so the third run, podcasts alone again, measures that noise instead of a hand-kept list.

The run's own records, the files directly in a run directory (``run.json``, ``metrics.json``,
``index.json``, the run manifest), are left out, as the gate leaves out run manifests: their
timings are coarse, so two runs alone could agree on one by chance and the run beside the text type
was then flagged for a timing (4 of 10 runs, 2026-11-13).
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Set, Tuple
from unittest.mock import patch

import pytest

from podcast_scraper import cli
from podcast_scraper.extensions import use_extensions
from podcast_scraper.rss import downloader
from tests.integration.test_text_in_fixture_type import _ext, _transport

pytestmark = [pytest.mark.integration]

_tests_dir = Path(__file__).parent.parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))
_spec = importlib.util.spec_from_file_location("gate2_parent_conftest", _tests_dir / "conftest.py")
assert _spec is not None and _spec.loader is not None
_parent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_parent)

_FEEDS = {
    "https://alpha.example/feed.xml": ("Feed Alpha", "guid-alpha-1", "https://alpha.example/1.txt"),
    "https://beta.example/feed.xml": ("Feed Beta", "guid-beta-1", "https://beta.example/1.txt"),
}
_TEXT_SOURCE = "https://example.com/fixture-source"
_RUN_DIR = re.compile(r"run_[^/]+")
# Artifact names carry the run's time: "0001 - Title_20261113-181508_0ca7efcd.bridge.json".
_RUN_TIME = re.compile(r"_\d{8}-\d{6}_")


def _rss(title: str, guid: str, transcript_url: str) -> str:
    return f"""<?xml version='1.0'?>
<rss xmlns:podcast="https://podcastindex.org/namespace/1.0">
  <channel>
    <language>en</language>
    <title>{title}</title>
    <item>
      <title>{title} episode one</title>
      <guid>{guid}</guid>
      <pubDate>Mon, 02 Nov 2026 09:00:00 GMT</pubDate>
      <podcast:transcript url="{transcript_url}" type="text/plain" />
    </item>
  </channel>
</rss>""".strip()


def _run(corpus: Path, with_text_type: bool) -> None:
    from podcast_scraper.workflow import run_index

    responses: Dict[str, Callable[[], Any]] = {}
    for rss_url, (title, guid, tr_url) in _FEEDS.items():
        responses[downloader.normalize_url(rss_url)] = partial(
            _parent.create_rss_response, _rss(title, guid, tr_url), rss_url
        )
        responses[downloader.normalize_url(tr_url)] = partial(
            _parent.create_transcript_response,
            f"{title}: a short transcript about trail building and rivers.",
            tr_url,
        )

    def fetch(url: str, user_agent: Any, timeout: Any, stream: bool = False) -> Any:
        factory = responses.get(downloader.normalize_url(url))
        if factory is None:
            raise AssertionError(f"unexpected HTTP request: {url}")
        return factory()

    lines = ["feeds:"] + [f"  - url: {u}" for u in _FEEDS]
    if with_text_type:
        lines.append(f"  - url: {_TEXT_SOURCE}\n    content_type: test_text_item")
    spec = corpus.parent / f"{corpus.name}.feeds.spec.yaml"
    spec.write_text("\n".join(lines) + "\n", encoding="utf-8")
    argv = [
        "--feeds-spec",
        str(spec),
        "--output-dir",
        str(corpus),
        "--no-transcript-cache",
        "--no-auto-speakers",
        "--generate-metadata",
        "--generate-kg",
        "--kg-extraction-source",
        "metadata_only",
    ]
    # Prod runs each batch in its own process; the corpus index cache must not span runs here.
    run_index.reset_corpus_metadata_index_cache_for_tests()
    with (
        use_extensions([_ext(_transport([]))]),
        patch("podcast_scraper.downloader.fetch_url", side_effect=fetch),
        patch("podcast_scraper.downloader.fetch_rss_feed_url", side_effect=fetch),
    ):
        assert cli.main(argv) == 0


def _podcast_files(corpus: Path) -> Dict[str, Any]:
    """Every podcast artifact under ``feeds/``, keyed by its path with run names normalised."""
    out: Dict[str, Any] = {}
    for path in sorted((corpus / "feeds").rglob("*")):
        if not path.is_file() or _RUN_DIR.fullmatch(path.parent.name):
            continue  # directly in a run dir: the run's records, not an episode's artifact
        rel = _RUN_TIME.sub("_*_", _RUN_DIR.sub("run_*", path.relative_to(corpus).as_posix()))
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            out[rel] = json.loads(text)
        except ValueError:
            out[rel] = text
    return out


def _leaves(value: Any, at: str = "") -> Iterator[Tuple[str, Any]]:
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _leaves(v, f"{at}/{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from _leaves(v, f"{at}[{i}]")
    else:
        yield at, value


def _differences(a: Dict[str, Any], b: Dict[str, Any]) -> Set[str]:
    diffs: Set[str] = set()
    for rel in a.keys() & b.keys():
        left, right = dict(_leaves(a[rel])), dict(_leaves(b[rel]))
        for key in left.keys() | right.keys():
            if left.get(key, object()) != right.get(key, object()):
                diffs.add(f"{rel}:{key}")
    return diffs


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> Tuple[Dict[str, Any], ...]:
    base = tmp_path_factory.mktemp("gate2")
    out = []
    for name, with_text in (("alone", False), ("again", False), ("beside", True)):
        # Each run in a new second: two runs within one second share their second-resolution
        # stamps, and the noise run would then miss them.
        time.sleep(1.1)
        corpus = base / name
        _run(corpus, with_text)
        out.append(_podcast_files(corpus))
    beside = base / "beside"
    assert sorted((beside / "sources").rglob("*.metadata.json")), "the text source did not run"
    return tuple(out)


def test_the_text_type_ran_and_the_podcasts_wrote_artifacts(
    runs: Tuple[Dict[str, Any], ...],
) -> None:
    alone, _, _ = runs
    kinds = {Path(rel).name.split(".", 1)[-1] for rel in alone}
    assert {"metadata.json", "kg.json", "txt"} <= kinds, sorted(kinds)


def test_the_same_podcast_files_are_written(runs: Tuple[Dict[str, Any], ...]) -> None:
    alone, again, beside = runs
    assert sorted(alone) == sorted(again), "two runs of the podcasts alone wrote different files"
    assert sorted(beside) == sorted(alone)


def test_the_text_type_changes_nothing_beyond_run_to_run_noise(
    runs: Tuple[Dict[str, Any], ...],
) -> None:
    alone, again, beside = runs
    noise = _differences(alone, again)
    introduced = _differences(alone, beside) - noise
    assert not introduced, sorted(introduced)
