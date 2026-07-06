"""#1148 producer mechanism — the six enricher-use-case structures + gold emission.

Asserts that ``scripts/build_v3_fixtures.py`` can render the authored structures
into the transcript and emit ``expected_enrichment`` gold (per-episode +
corpus-level), and — crucially — that an episode authoring *none* of them
produces byte-identical output (so the existing committed fixtures are
untouched until content is authored).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_generator():
    path = PROJECT_ROOT / "scripts" / "build_v3_fixtures.py"
    spec = importlib.util.spec_from_file_location("build_v3_fixtures", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def v3():
    return _load_generator()


def _podcast_with(v3, ep):
    """A minimal 3-guest podcast wrapping one authored episode."""
    guest = v3.GuestV3(name="Ada Lovelace", role="researcher", expertise="algorithms")
    co = v3.GuestV3(name="Alan Turing", role="mathematician", expertise="computation")
    opp = v3.GuestV3(name="Grace Hopper", role="engineer", expertise="compilers")
    return v3.PodcastV3(
        pod_id="pZ",
        title="Test Feed",
        domain="Computing",
        host="Rae Chen",
        guests={"ada": guest, "alan": co, "grace": opp},
        recurring_orgs=["Acme"],
        episodes=[ep],
    )


def test_unauthored_episode_emits_no_new_ground_truth_keys(v3):
    ep = v3.EpisodeV3(
        ep_id="e01",
        title="Baseline",
        primary_guest="ada",
        primary_topic="topic:algorithms",
        secondary_topics=["topic:computation"],
        sponsor_brands=["Acme"],
        talking_points=["Algorithms compound when they compose."],
    )
    _text, gt = v3.render_episode(_podcast_with(v3, ep), ep)
    # None of the #1148 keys appear → committed fixtures stay byte-identical.
    for key in (
        v3.EXPECTED_ENRICHMENT_KEY,
        "publish_offset_days",
        "additional_guests",
        "topic_claims",
        "contradiction_claims",
        "insight_density",
        "publish_date",
    ):
        assert key not in gt


def test_authored_structures_render_and_emit_gold(v3):
    ep = v3.EpisodeV3(
        ep_id="e02",
        title="Loaded",
        primary_guest="ada",
        primary_topic="topic:algorithms",
        secondary_topics=["topic:computation"],
        sponsor_brands=["Acme"],
        talking_points=["Composition is the core idea."],
        publish_offset_days=90,
        additional_guests=["alan"],
        insight_density="low",
        topic_claims=[
            {
                "topic_id": "topic:computation",
                "speaker": "grace",
                "claim": "Compilers are leverage.",
                "grounded": True,
            }
        ],
        contradiction_claims=[
            {
                "topic_id": "topic:algorithms",
                "speaker_a": "ada",
                "claim_a": "Elegance beats brute force.",
                "speaker_b": "alan",
                "claim_b": "Brute force wins at scale.",
            }
        ],
        expected_enrichment={
            "guest_coappearance": {"expected_pairs": [["pZ:ada", "pZ:alan"]]},
            "grounding_rate": {"expected_rate": 0.6},
        },
    )
    text, gt = v3.render_episode(_podcast_with(v3, ep), ep)
    # Structures rendered into the transcript.
    assert "Alan Turing" in text  # additional guest
    assert "Grace Hopper" in text and "Compilers are leverage." in text  # topic claim
    assert "genuine disagreement" in text and "Brute force wins at scale." in text  # opposition
    # Gold + structure keys emitted.
    assert gt[v3.EXPECTED_ENRICHMENT_KEY]["guest_coappearance"]["expected_pairs"] == [
        ["pZ:ada", "pZ:alan"]
    ]
    assert gt["publish_offset_days"] == 90
    assert gt["publish_date"] == "2024-03-31"  # CORPUS_EPOCH 2024-01-01 + 90d
    assert gt["insight_density"] == "low"
    assert gt["additional_guests"] == ["alan"]


def test_corpus_meta_summary_and_gold(v3, tmp_path, monkeypatch):
    meta = v3.CorpusV3Meta(
        shared_topics=["topic:algorithms"],
        contradiction_pairs=[{"topic_id": "topic:algorithms"}],
        seeded_users=[{"user_id": "u1", "heard": ["pZ_e02"]}],
        expected_enrichment={
            "topic_similarity": {"expected_neighbours": {"topic:algorithms": ["topic:computation"]}}
        },
    )
    summary = v3.emit_corpus([], dry_run=True, corpus_meta=meta)
    assert summary["shared_topics"] == ["topic:algorithms"]
    assert summary["contradiction_pair_count"] == 1
    assert summary["seeded_user_count"] == 1
    assert summary["corpus_gold_enricher_ids"] == ["topic_similarity"]


def test_build_v3_corpus_meta_is_authored(v3):
    # The corpus meta carries the risk-management ↔ systems-thinking overlap web.
    meta = v3.build_v3_corpus_meta()
    assert "topic:risk-management" in meta.shared_topics
    assert len(meta.seeded_users) == 3
    assert "guest_coappearance" in meta.expected_enrichment
    assert meta.contradiction_pairs  # the diversify-vs-concentrate opposition
