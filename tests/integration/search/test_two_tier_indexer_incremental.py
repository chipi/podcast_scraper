"""Guardrail: the two-tier indexer must be INCREMENTAL — re-embedding only new/changed episodes.

This is the D8 regression suite (see docs/wip/DESIGN-D8-incremental-indexer-skip.md). It reproduces
the prod defect where every incremental corpus add re-embedded the ENTIRE corpus (O(N)): adding one
episode to 107 took ~10.6 min because ``build_two_tier_index`` had no unchanged-episode skip and
``episodes_skipped_unchanged`` was a dead stat.

The guardrail asserts EMBED CALL COUNT (compute), not just stored rows — upsert is idempotent for
storage but was not for compute. ``_embed`` is stubbed as a counter so the assertions are exact and
model-free.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]

pytest.importorskip("lancedb")

from podcast_scraper.search import two_tier_indexer as tti  # noqa: E402

_EMBED_DIM = 8


class _EmbedCounter:
    """Deterministic fake embedder that counts calls. Distinct text -> distinct vector so the
    LanceDB upsert path behaves; the test only asserts the CALL COUNT."""

    def __init__(self) -> None:
        self.calls = 0
        self.texts: list[str] = []

    def __call__(self, text: str, model_id: str, *, allow_download: bool):  # matches tti._embed
        self.calls += 1
        self.texts.append(text)
        h = abs(hash(text))
        return [float((h >> (i * 3)) & 0x7) / 7.0 for i in range(_EMBED_DIM)]


def _episode_rows(ep_id: str, *, variant: str = "a"):
    """Two embeddable docs per episode (one insight, one transcript chunk)."""
    return [
        (
            f"insight:{ep_id}",
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


def _install_corpus(monkeypatch, tmp_path, episodes: dict[str, str], counter: _EmbedCounter):
    """Wire the indexer to a synthetic corpus of ``{episode_id: variant}`` with a counting embedder.

    ``episodes`` maps episode_id -> content variant; changing a variant changes the collected rows
    (so a "changed episode" re-embeds). Re-call to mutate the corpus between builds.
    """
    corpus = tmp_path / "corpus"
    (corpus / "metadata").mkdir(parents=True, exist_ok=True)
    meta_paths = {ep: corpus / "metadata" / f"{ep}.metadata.json" for ep in episodes}

    monkeypatch.setattr(tti, "discover_metadata_files", lambda root: list(meta_paths.values()))
    monkeypatch.setattr(
        tti, "_load_metadata_file", lambda p: {"episode": {"episode_id": p.name.split(".")[0]}}
    )
    monkeypatch.setattr(tti, "episode_root_from_metadata_path", lambda p: corpus)
    monkeypatch.setattr(
        tti,
        "_collect_docs_for_episode",
        lambda episode_root, meta_path, *a, **k: _episode_rows(
            meta_path.name.split(".")[0], variant=episodes[meta_path.name.split(".")[0]]
        ),
    )
    monkeypatch.setattr(tti, "_embed", counter)
    return corpus


def test_full_build_embeds_all_episodes(tmp_path, monkeypatch):
    counter = _EmbedCounter()
    corpus = _install_corpus(monkeypatch, tmp_path, {"ep1": "a", "ep2": "a"}, counter)
    lance = corpus / "search" / "lance_index"

    stats = tti.build_two_tier_index(corpus, lance, drop_existing=True)

    assert stats.episodes == 2
    assert counter.calls == 4  # 2 episodes x 2 docs
    assert getattr(stats, "episodes_skipped_unchanged", 0) == 0


def test_reindex_unchanged_corpus_embeds_nothing(tmp_path, monkeypatch):
    """THE core D8 guardrail: rebuilding an unchanged corpus must re-embed ZERO episodes."""
    counter = _EmbedCounter()
    corpus = _install_corpus(monkeypatch, tmp_path, {"ep1": "a", "ep2": "a"}, counter)
    lance = corpus / "search" / "lance_index"

    tti.build_two_tier_index(corpus, lance, drop_existing=True)
    first = counter.calls
    assert first == 4

    stats2 = tti.build_two_tier_index(corpus, lance, drop_existing=False)

    assert counter.calls - first == 0, "unchanged corpus must not re-embed any episode (D8)"
    assert stats2.episodes_skipped_unchanged == 2


def test_incremental_add_embeds_only_new_episode(tmp_path, monkeypatch):
    counter = _EmbedCounter()
    episodes = {"ep1": "a", "ep2": "a"}
    corpus = _install_corpus(monkeypatch, tmp_path, episodes, counter)
    lance = corpus / "search" / "lance_index"
    tti.build_two_tier_index(corpus, lance, drop_existing=True)
    baseline = counter.calls

    episodes["ep3"] = "a"  # add one episode
    _install_corpus(monkeypatch, tmp_path, episodes, counter)
    stats = tti.build_two_tier_index(corpus, lance, drop_existing=False)

    assert counter.calls - baseline == 2, "only the new episode's 2 docs should embed (D8)"
    assert stats.episodes_skipped_unchanged == 2
    assert stats.episodes == 3


def test_changed_episode_is_reembedded(tmp_path, monkeypatch):
    counter = _EmbedCounter()
    episodes = {"ep1": "a", "ep2": "a"}
    corpus = _install_corpus(monkeypatch, tmp_path, episodes, counter)
    lance = corpus / "search" / "lance_index"
    tti.build_two_tier_index(corpus, lance, drop_existing=True)
    baseline = counter.calls

    episodes["ep1"] = "b"  # content change -> fingerprint differs
    _install_corpus(monkeypatch, tmp_path, episodes, counter)
    stats = tti.build_two_tier_index(corpus, lance, drop_existing=False)

    assert counter.calls - baseline == 2, "only the changed episode's docs re-embed (D8)"
    assert stats.episodes_skipped_unchanged == 1


def test_full_rebuild_ignores_fingerprints_and_reembeds_all(tmp_path, monkeypatch):
    counter = _EmbedCounter()
    corpus = _install_corpus(monkeypatch, tmp_path, {"ep1": "a", "ep2": "a"}, counter)
    lance = corpus / "search" / "lance_index"
    tti.build_two_tier_index(corpus, lance, drop_existing=True)
    baseline = counter.calls

    stats = tti.build_two_tier_index(corpus, lance, drop_existing=True)

    assert counter.calls - baseline == 4, "drop_existing=True must re-embed everything"
    assert stats.episodes_skipped_unchanged == 0
