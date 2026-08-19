"""The cost gate, exercised as a real process against a real mini corpus.

WHY A SUBPROCESS AND NOT A FUNCTION CALL. Everything else on this branch tests that the run stops
SPENDING. None of it tests that the process EXITS — and on 2026-08-19 a container was found "Up 7
days" whose process had raised ``CostCapExceeded`` and then hung forever, because the exception
unwound past the line releasing a non-daemon worker. A cost abort that leaves the process alive is
not a fix: it converts a money leak into a silent zombie holding the corpus.

An in-process test cannot see that. `pytest.raises(CostCapExceeded)` passes whether or not stray
threads remain. Only a real process, with a real timeout on it, answers "did it terminate".

Nothing here contacts a provider. The corpus is a handful of on-disk metadata files and the cap is
set so the gate refuses at SELECTION — before a single byte moves. That is also the honest limit
of this file: it proves the pre-flight path exits, not the mid-run abort path, which needs a
provider to be reached. That case is covered structurally by
``test_transcription_event_always_set.py`` and ``test_transcription_supervision.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from podcast_scraper.rss.feed_cache import cache_path_for_url, ENV_RSS_CACHE_DIR

pytestmark = [pytest.mark.e2e]

#: Generous relative to the work (selection over ~40 tiny JSON files), tight relative to "hung".
#: The 2026-08-12 zombie would blow through this and the test would report a timeout, which is
#: exactly the signal wanted.
EXIT_TIMEOUT_SECONDS = 120


def _write_corpus(feed_dir: Path, episodes: list[tuple[str, str, int]]) -> None:
    run = feed_dir / "run_20260815-120000"
    (run / "metadata").mkdir(parents=True, exist_ok=True)
    for i, (guid, episode_id, duration) in enumerate(episodes, start=1):
        (run / "metadata" / f"{i:04d} - Episode {i}.metadata.json").write_text(
            json.dumps(
                {
                    "episode": {
                        "episode_id": episode_id,
                        "guid": guid,
                        "title": f"Episode {i}",
                        "duration_seconds": duration,
                        "published_date": "Mon, 01 Jan 2026 00:00:00 +0000",
                    },
                    "content": {"transcript_source": "whisper_transcription"},
                }
            ),
            encoding="utf-8",
        )


RSS_URL = "https://example.com/feed.xml"


def _seed_rss_cache(cache_dir: Path, episodes: int, seconds: int) -> None:
    """Put the feed on disk so the run needs no network.

    fetch_and_parse_feed consults feed_cache BEFORE the downloader, so a seeded cache makes the
    whole run offline. Reprocess mode still fetches the feed — it reads the episode SET from disk
    but the feed itself is fetched first — which is why this is required rather than optional.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    itunes = "http://www.itunes.com/dtds/podcast-1.0.dtd"
    items = "".join(
        f"<item><guid>g{i}</guid><title>Episode {i}</title>"
        f'<itunes:duration xmlns:itunes="{itunes}">{seconds}</itunes:duration>'
        f'<enclosure url="https://example.com/g{i}.mp3" type="audio/mpeg" length="1000"/>'
        "<pubDate>Mon, 01 Jan 2026 00:00:00 +0000</pubDate></item>"
        for i in range(1, episodes + 1)
    )
    xml = (
        '<?xml version="1.0"?><rss version="2.0"><channel>'
        f"<title>Mini Show</title><description>local mini corpus</description>{items}"
        "</channel></rss>"
    )
    cache_path_for_url(cache_dir, RSS_URL).write_text(xml, encoding="utf-8")


def _run_cli(args: list[str], cwd: Path, cache_dir: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    # Placeholder credentials: config validates that a key EXISTS for the chosen provider, and
    # the gate refuses before anything is ever sent anywhere.
    env["DEEPGRAM_API_KEY"] = "not-a-real-key-nothing-is-called"
    env[ENV_RSS_CACHE_DIR] = str(cache_dir)
    # No profile is loaded here (cloud_balanced requires a reachable LiteLLM gateway), so the
    # action would default to "observe" and nothing would ever be refused.
    env["COST_SOFT_CAP_ACTION"] = "abort"
    env.pop("COST_SOFT_CAP_USD_PER_RUN", None)
    return subprocess.run(
        [sys.executable, "-m", "podcast_scraper.cli", *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=EXIT_TIMEOUT_SECONDS,
        check=False,
    )


def _args(feed_dir: Path, cap: str) -> list[str]:
    return [
        RSS_URL,  # POSITIONAL: --rss is the repeatable multi-feed flag, not this
        "--output-dir",
        str(feed_dir),
        "--reprocess-existing-only",
        "--transcription-provider",
        "deepgram",
        "--deepgram-model",
        "nova-3",
        "--cost-soft-cap-usd-per-run",
        cap,
        "--no-transcript-cache",
    ]


@pytest.fixture
def mini_corpus(tmp_path: Path) -> Path:
    """40 one-hour episodes on disk — ~$10.75 of ASR, comfortably over a $1 cap."""
    root = tmp_path / "corpus"
    feed = root / "feeds" / "rss_example.com_mini"
    _write_corpus(feed, [(f"g{i}", f"ep{i}", 3600) for i in range(40)])
    return root


def test_the_process_EXITS_when_the_gate_refuses(mini_corpus: Path, tmp_path: Path) -> None:
    """The property no in-process test can establish.

    If this times out, the run stopped spending and did not stop RUNNING — which is the 7-day
    container, reproduced. The assertion message says so, because a bare timeout is easy to
    dismiss as flake.
    """
    feed_dir = mini_corpus / "feeds" / "rss_example.com_mini"
    cache = mini_corpus / "rsscache"
    _seed_rss_cache(cache, episodes=40, seconds=3600)
    try:
        result = _run_cli(_args(feed_dir, "1.0"), cwd=Path.cwd(), cache_dir=cache)
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"the process did NOT exit within {EXIT_TIMEOUT_SECONDS}s. A cost abort that hangs is "
            "the 2026-08-19 zombie: it stops spending and never terminates, so the container "
            "never dies and nobody notices."
        )

    combined = result.stdout + result.stderr
    assert result.returncode != 0, f"a refused run must not report success\n{combined[-3000:]}"
    assert "REFUSING TO START" in combined, combined[-3000:]


def test_the_refusal_names_the_numbers_an_operator_needs(mini_corpus: Path, tmp_path: Path) -> None:
    """A stop the operator cannot act on is half a fix."""
    feed_dir = mini_corpus / "feeds" / "rss_example.com_mini"
    cache = mini_corpus / "rsscache"
    _seed_rss_cache(cache, episodes=40, seconds=3600)
    result = _run_cli(_args(feed_dir, "1.0"), cwd=Path.cwd(), cache_dir=cache)
    combined = result.stdout + result.stderr

    # the selection manifest: the line whose absence hid a 678-episode selection for six hours
    assert "selection: 40 of 40 episodes" in combined, combined[-3000:]
    assert "audio-hours" in combined
    # and the actionable half
    assert "would fit" in combined
    assert "split the work-list" in combined


def test_a_run_INSIDE_the_budget_is_not_refused(mini_corpus: Path) -> None:
    """The gate must not simply refuse everything — that would pass the test above trivially."""
    feed_dir = mini_corpus / "feeds" / "rss_example.com_mini"
    cache = mini_corpus / "rsscache"
    _seed_rss_cache(cache, episodes=40, seconds=3600)
    result = _run_cli(_args(feed_dir, "500.0"), cwd=Path.cwd(), cache_dir=cache)
    combined = result.stdout + result.stderr
    assert "REFUSING TO START" not in combined, combined[-3000:]
    assert "selection: 40 of 40 episodes" in combined
