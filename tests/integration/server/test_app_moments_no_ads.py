"""A short listen never plays an ad (operator 2026-10-12).

Over the COMMITTED fixture corpus, through ``GET /episodes/{slug}/moments``: 4 English clips used
to run on into the break ("We'll be right back after a quick word from our sponsors." and the
sponsor read after it), because the clip grows by whole segments to its minimum length. Each still
plays, and now ends where its point does. In the five translated shows, the English render ran on
another timeline than the quotes (a clip played "Start at strava.com/podcast"), and the break's
lead-in was itself an insight; both are fixed in the fixture, and this covers them too. The corpus
is copied, never served in place.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server import app_sessions
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_slugs import _slug_index
from podcast_scraper.server.app_user_store import get_or_create_user

pytestmark = [pytest.mark.integration]

CORPUS = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "app-validation-corpus" / "v3"
#: What an ad read or break says in this corpus, in each of its languages — the measurement, not
#: the product's rule. The generator's own ad-copy list (`is_ad_copy`: sponsor reads, host-read
#: native ads, credits) plus the breaks and spoken URLs; the first version of this check knew only
#: the sponsor wording and missed a native ad and a "thanks to X for supporting the show".
AD = re.compile(
    r"sponsor|brought to you|right back after|\.com/podcast|free trial|our partners"
    r"|publicitari|pubblicitari|publicitaire|werbepause|show notes|supporting the show",
    re.I,
)


def _is_ad(text: str) -> bool:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    from build_app_validation_corpus import is_ad_copy

    return bool(AD.search(text)) or is_ad_copy(text)


#: The four clips that leaked, by episode, with where their point ends — the line before "We'll be
#: right back after a quick word from our sponsors.", on the audio's real timeline since
#: 2026-10-15 (it was 108 / 84 / 54 / 66 s on the fixture's old 6 s grid).
WERE_LEAKING = {"p02_e02": 149_587, "p04_e01": 112_278, "p08_e02": 72_541, "p09_e02": 87_101}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("noads") / "v3"
    shutil.copytree(CORPUS, root, ignore=shutil.ignore_patterns("lance_index"))
    return root


def _client(root: Path) -> TestClient:
    data_dir = root.parent / "_appdata"
    app = create_app(root, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    user = get_or_create_user(data_dir, provider="stub", subject="s", email="u@x.com", name="U")
    client = TestClient(app)
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)
    return client


def _moments(corpus: Path) -> dict[str, tuple[list[dict], list[dict]]]:
    client = _client(corpus)
    out = {}
    for slug, row in sorted(_slug_index(corpus).items()):
        label = Path(row.metadata_relative_path).name[: -len(".metadata.json")]
        seg_path = corpus / row.metadata_relative_path.replace("/metadata/", "/transcripts/")
        raw = json.loads(seg_path.with_name(f"{label}.segments.json").read_text("utf-8"))
        segs = raw.get("segments", raw) if isinstance(raw, dict) else raw
        out[label] = (client.get(f"/api/app/episodes/{slug}/moments").json()["moments"], segs)
    return out


def test_no_clip_plays_an_ad_read(corpus: Path) -> None:
    leaks = []
    for label, (moments, segs) in _moments(corpus).items():
        reads = [(s["start"] * 1000, s["end"] * 1000) for s in segs if AD.search(s["text"])]
        for m in moments:
            said = m["clip_text"] + " " + m["text"]
            if any(a < m["end_ms"] and b > m["start_ms"] for a, b in reads) or _is_ad(said):
                leaks.append((label, m["start_ms"], m["clip_text"][:80]))
    assert leaks == []


def test_the_clips_that_ran_into_a_break_still_play_and_end_before_it(corpus: Path) -> None:
    """Every clip that could play for these points — the episode's moment CANDIDATES, not only
    the ones picked: on the real timeline (2026-10-15) p04_e01 picks two other moments, and the
    clip for this point, still built and still ending at the break, is no longer among them."""
    from podcast_scraper.server.app_corpus_access import load_json_artifact
    from podcast_scraper.server.app_episode_search import (
        episode_ad_spans,
        episode_segments,
        episode_turn_starts,
    )
    from podcast_scraper.server.app_moments import MomentsConfig, score_candidates

    rows = {
        Path(r.metadata_relative_path).name[: -len(".metadata.json")]: r
        for r in _slug_index(corpus).values()
    }
    for label, end_ms in WERE_LEAKING.items():
        row = rows[label]
        candidates = score_candidates(
            load_json_artifact(corpus, row.gi_relative_path),
            MomentsConfig(),
            episode_segments(corpus, row),
            episode_ad_spans(corpus, row),
            episode_turn_starts(corpus, row),
        )
        clip = next(m for m in candidates if m.start_ms < end_ms <= m.start_ms + 30_000)
        assert clip.end_ms == end_ms, label
        assert clip.clip_text and not AD.search(clip.clip_text), label


def test_no_reel_clip_plays_an_ad(corpus: Path) -> None:
    """Every subject's reel: a clip ran on into the host's native ad ("Linear runs my whole
    engineering ops … use the link in show notes") and another into a sponsor credit, both the
    next speaker's line. A clip now ends where its speaker does."""
    from podcast_scraper.server.app_goal_reel import goal_reel

    subjects = {"topic:risk-management", "topic:systems-thinking", "topic:reliability"}
    for name in ("search/topic_clusters.json", "enrichments/topic_theme_clusters.json"):
        doc = json.loads((corpus / name).read_text("utf-8"))
        clusters = doc["clusters"] if "clusters" in doc else doc["data"]["clusters"]
        subjects |= {str(c["graph_compound_parent_id"]) for c in clusters}
    leaks = [
        (s, c.episode_slug, c.moment.clip_text[:80])
        for s in sorted(subjects)
        for c in goal_reel(corpus, s, minutes=15)
        if _is_ad(c.moment.clip_text + " " + c.moment.text)
    ]
    assert leaks == []
