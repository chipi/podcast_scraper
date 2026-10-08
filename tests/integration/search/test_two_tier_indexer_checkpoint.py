"""An incremental index build that is killed keeps the work it finished (root cause of #2299).

Prod 2026-10-07: the post-run update had 367 changed episodes, hit its time limit, and — because
fingerprints were written only at the end — recorded none of its work, so every later run started
the whole backlog again and search stayed stale. The build now checkpoints every N re-embedded
episodes: flush, prune the finished episodes' superseded rows, save their fingerprints.

Model-free: ``_embed`` is a counting stub (as in test_two_tier_indexer_incremental.py), and a
"kill" is the embedder raising mid-build, which leaves exactly what a killed process leaves.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]

pytest.importorskip("lancedb")

from podcast_scraper.search import two_tier_indexer as tti  # noqa: E402

_EMBED_DIM = 8
EPISODES = ["ep1", "ep2", "ep3", "ep4", "ep5", "ep6"]


class Killed(Exception):
    """Stands in for the subprocess timeout killing the build."""


class _Embedder:
    def __init__(self, kill_on: str | None = None) -> None:
        self.calls = 0
        self.kill_on = kill_on

    def __call__(self, text: str, model_id: str, *, allow_download: bool):
        if self.kill_on and self.kill_on in text:
            raise Killed(text)
        self.calls += 1
        h = abs(hash(text))
        return [float((h >> (i * 3)) & 0x7) / 7.0 for i in range(_EMBED_DIM)]


def _rows(ep_id: str, variant: str):
    # The insight id is CONTENT-keyed, like prod's derived rows: a changed episode's new insight
    # lands beside the old one, which only the prune removes (#1969).
    return [
        (
            f"insight:{ep_id}:{variant}",
            f"insight text for {ep_id} {variant}",
            {"doc_type": "insight", "episode_id": ep_id, "feed_id": "show1", "grounded": True},
        ),
        (
            f"chunk:{ep_id}",
            f"transcript chunk for {ep_id} {variant}",
            {
                "doc_type": "transcript",
                "episode_id": ep_id,
                "feed_id": "show1",
                "timestamp_start_ms": 0,
                "timestamp_end_ms": 1000,
            },
        ),
    ]


def _corpus(monkeypatch, tmp_path, variants: dict, embedder: _Embedder):
    corpus = tmp_path / "corpus"
    (corpus / "metadata").mkdir(parents=True, exist_ok=True)
    paths = [corpus / "metadata" / f"{ep}.metadata.json" for ep in EPISODES]
    monkeypatch.setattr(tti, "discover_metadata_files", lambda root: list(paths))
    monkeypatch.setattr(
        tti, "_load_metadata_file", lambda p: {"episode": {"episode_id": p.name.split(".")[0]}}
    )
    monkeypatch.setattr(tti, "episode_root_from_metadata_path", lambda p: corpus)
    monkeypatch.setattr(
        tti,
        "_collect_docs_for_episode",
        lambda root, meta_path, *a, **k: _rows(
            meta_path.name.split(".")[0], variants[meta_path.name.split(".")[0]]
        ),
    )
    monkeypatch.setattr(tti, "_embed", embedder)
    return corpus, corpus / "search" / "lance_index"


def _insight_ids(lance) -> set:
    import lancedb

    return set(lancedb.connect(str(lance)).open_table("insights").to_pandas()["id"])


def test_a_killed_update_keeps_its_finished_episodes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PODCAST_INDEX_CHECKPOINT_EPISODES", "2")
    old = {ep: "a" for ep in EPISODES}
    corpus, lance = _corpus(monkeypatch, tmp_path, old, _Embedder())
    tti.build_two_tier_index(corpus, lance, drop_existing=True)

    # Every episode changes (the prod backlog), and the update is killed while on ep5.
    new = {ep: "b" for ep in EPISODES}
    _corpus(monkeypatch, tmp_path, new, _Embedder(kill_on="ep5"))
    with pytest.raises(Killed):
        tti.build_two_tier_index(corpus, lance)

    # The next update re-embeds only what was not finished: ep5 and ep6, 2 docs each — not all 6.
    retry = _Embedder()
    _corpus(monkeypatch, tmp_path, new, retry)
    stats = tti.build_two_tier_index(corpus, lance)
    assert retry.calls == 4
    assert stats.episodes_skipped_unchanged == 4

    # ...and every superseded row is gone, including those of the episodes pruned at a checkpoint.
    assert _insight_ids(lance) == {f"insight:{ep}:b" for ep in EPISODES}


def test_a_full_rebuild_never_checkpoints(tmp_path, monkeypatch) -> None:
    # A full rebuild clears tables at the end and cannot be resumed: a killed one must leave the
    # previous fingerprints alone, so the next build redoes everything.
    monkeypatch.setenv("PODCAST_INDEX_CHECKPOINT_EPISODES", "2")
    variants = {ep: "a" for ep in EPISODES}
    corpus, lance = _corpus(monkeypatch, tmp_path, variants, _Embedder())
    tti.build_two_tier_index(corpus, lance, drop_existing=True)
    before = tti._fingerprints_path(lance).read_text(encoding="utf-8")

    changed = {ep: "c" for ep in EPISODES}
    _corpus(monkeypatch, tmp_path, changed, _Embedder(kill_on="ep5"))
    with pytest.raises(Killed):
        tti.build_two_tier_index(corpus, lance, drop_existing=True)
    assert tti._fingerprints_path(lance).read_text(encoding="utf-8") == before


def test_an_uninterrupted_update_is_unchanged_by_checkpoints(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PODCAST_INDEX_CHECKPOINT_EPISODES", "2")
    corpus, lance = _corpus(monkeypatch, tmp_path, {ep: "a" for ep in EPISODES}, _Embedder())
    tti.build_two_tier_index(corpus, lance, drop_existing=True)

    update = _Embedder()
    _corpus(monkeypatch, tmp_path, {ep: "b" for ep in EPISODES}, update)
    stats = tti.build_two_tier_index(corpus, lance)
    assert update.calls == 12
    assert stats.stale_rows_pruned == 6  # one superseded insight per episode, across checkpoints
    assert _insight_ids(lance) == {f"insight:{ep}:b" for ep in EPISODES}
    # Nothing left to do afterwards.
    again = _Embedder()
    _corpus(monkeypatch, tmp_path, {ep: "b" for ep in EPISODES}, again)
    tti.build_two_tier_index(corpus, lance)
    assert again.calls == 0


@pytest.mark.parametrize(("env", "expected"), [(None, 25), ("5", 5), ("0", 25), ("x", 25)])
def test_checkpoint_interval_from_env(env, expected, monkeypatch) -> None:
    if env is None:
        monkeypatch.delenv("PODCAST_INDEX_CHECKPOINT_EPISODES", raising=False)
    else:
        monkeypatch.setenv("PODCAST_INDEX_CHECKPOINT_EPISODES", env)
    assert tti._checkpoint_every() == expected
