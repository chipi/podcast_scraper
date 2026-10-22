"""The voice identity rung (design stage 3): groups, vote, decisions, and when it abstains."""

from __future__ import annotations

import json
import types

import pytest

from podcast_scraper.providers.ml.diarization import voice_identity as vi
from podcast_scraper.providers.ml.diarization.roster import SpeakerRole, SpeakerRoster

A = (1.0, 0.0, 0.0)  # the host's voice
B = (0.0, 1.0, 0.0)  # a guest's voice
NEAR_A = (0.9, 0.3, 0.0)  # cos to A ~0.95


def hv(ep, name, vec, role="host", talk=300.0):
    return vi.HistoryVoice(episode_id=ep, name=name, role=role, talk=talk, vec=vi._unit(vec))


def test_group_named_by_majority_from_two_episodes():
    groups = vi.build_groups([hv("e1", "Ann Lee", A), hv("e2", "Ann Lee", NEAR_A)], 0.7, 60)
    assert [(g.person, g.display, g.role) for g in groups] == [("ann lee", "Ann Lee", "host")]


def test_one_episode_is_not_enough_to_name_a_group():
    assert vi.build_groups([hv("e1", "Ann Lee", A), hv("e1", "Ann Lee", NEAR_A)], 0.7, 60) == []


def test_a_tied_vote_names_no_group():
    hist = [
        hv("e1", "Ann Lee", A),
        hv("e2", "Ann Lee", A),
        hv("e3", "Bo Kim", A),
        hv("e4", "Bo Kim", A),
    ]
    assert vi.build_groups(hist, 0.7, 60) == []


def test_a_minority_of_swapped_names_does_not_decide_the_group():
    hist = [hv("e1", "Ann Lee", A), hv("e2", "Ann Lee", A), hv("e3", "Bo Kim", A)]
    assert [g.person for g in vi.build_groups(hist, 0.7, 60)] == ["ann lee"]


def test_short_history_voices_do_not_vote():
    hist = [hv("e1", "Ann Lee", A, talk=59.0), hv("e2", "Ann Lee", A, talk=59.0)]
    assert vi.build_groups(hist, 0.7, 60) == []
    assert vi.build_groups(hist, 0.7, 30)


GROUPS = [vi.VoiceGroup(person="ann lee", display="Ann Lee", role="host", centroid=vi._unit(A))]


def test_the_host_name_moves_off_the_guest_onto_the_host_voice():
    voices = {"S0": (None, "unknown", 900.0, A), "S1": ("Ann Lee", "host", 800.0, B)}
    assert vi.decide(voices, GROUPS, 0.7) == {"S0": ("Ann Lee", "host"), "S1": (None, "host")}


def test_a_name_stays_on_a_voice_that_is_not_clearly_someone_else():
    mid = (0.6, 0.8, 0.0)  # cos to A 0.6: below the threshold, above not_same
    voices = {"S0": (None, "unknown", 900.0, A), "S1": ("Ann Lee", "host", 800.0, mid)}
    assert vi.decide(voices, GROUPS, 0.7, not_same=0.5) == {"S0": ("Ann Lee", "host")}
    assert vi.decide(voices, GROUPS, 0.7) == {"S0": ("Ann Lee", "host")}  # the default is 0.5
    assert vi.decide(voices, GROUPS, 0.7, not_same=0.7)["S1"] == (None, "host")


def test_below_the_threshold_nothing_changes():
    voices = {"S0": (None, "unknown", 900.0, (0.65, 0.76, 0.0))}  # cos 0.65
    assert vi.decide(voices, GROUPS, 0.7) == {}
    assert vi.decide(voices, GROUPS, 0.6) == {"S0": ("Ann Lee", "host")}


def test_a_split_host_keeps_the_name_on_both_voices():
    voices = {"S0": ("Ann Lee", "host", 900.0, A), "S1": (None, "guest", 400.0, NEAR_A)}
    assert vi.decide(voices, GROUPS, 0.7) == {"S1": ("Ann Lee", "host")}


def test_a_very_short_voice_is_not_named():
    voices = {"S0": (None, "unknown", 9.0, A)}
    assert vi.decide(voices, GROUPS, 0.7) == {}


def _write_episode(feed, run, eid, voices, model="pyannote/speaker-diarization-community-1"):
    meta = feed / run / "metadata" / f"{eid}.metadata.json"
    meta.parent.mkdir(parents=True)
    (feed / run / "transcripts").mkdir()
    meta.write_text(
        json.dumps(
            {
                "episode": {"episode_id": eid},
                "content": {"transcript_file_path": f"transcripts/{eid}.txt"},
            }
        )
    )
    (feed / run / "transcripts" / f"{eid}.speakers.diagnostics.json").write_text(
        json.dumps(
            {
                "voices": [
                    {
                        "voice": v,
                        "resolved_name": n,
                        "named": bool(n),
                        "role": "host",
                        "talk_time_s": 600,
                    }
                    for v, n, _ in voices
                ],
                "voice_embeddings": {
                    "model_name": model,
                    "dimension": 3,
                    "by_voice": {v: list(vec) for v, _, vec in voices},
                },
            }
        )
    )


def test_feed_history_leaves_out_this_episode_and_other_models(tmp_path, monkeypatch):
    feed = tmp_path / "feeds" / "f1"
    _write_episode(feed, "run_1", "ep1", [("SPEAKER_00", "Ann Lee", A)])
    _write_episode(feed, "run_2", "ep2", [("SPEAKER_00", "Ann Lee", A)])
    _write_episode(feed, "run_3", "ep3", [("SPEAKER_00", "Ann Lee", A)], model="other/model")
    import podcast_scraper.search.corpus_scope as cs

    monkeypatch.setattr(cs, "dedupe_metadata_paths_newest_run_per_episode", lambda root, m: m)
    hist = vi.feed_history(feed, "ep1", "pyannote/speaker-diarization-community-1")
    assert [h.episode_id for h in hist] == ["ep2"]


def _roster(**voices):
    return SpeakerRoster(
        by_voice={
            v: SpeakerRole(name=n or v, role=r, named=bool(n), source="llm_resolution")
            for v, (n, r) in voices.items()
        },
        num_speakers=len(voices),
    )


def _diarization(vectors, talk):
    segs = [types.SimpleNamespace(speaker=v, start=0.0, end=t) for v, t in talk.items()]
    return types.SimpleNamespace(
        speaker_embeddings=vectors, segments=segs, speaker_embeddings_model="", model_name="m"
    )


@pytest.mark.parametrize("missing", ["vectors", "feed", "episode"])
def test_apply_abstains_without_what_it_needs(missing, tmp_path):
    roster = _roster(S0=("Ann Lee", "host"))
    d = _diarization({} if missing == "vectors" else {"S0": list(A)}, {"S0": 600.0})
    out = vi.apply_to_roster(
        roster,
        d,
        None if missing == "feed" else tmp_path,
        None if missing == "episode" else "ep9",
        0.7,
    )
    assert out is roster


def test_apply_changes_the_roster_and_records_the_source(tmp_path, monkeypatch):
    monkeypatch.setattr(
        vi, "feed_history", lambda *a: [hv("e1", "Ann Lee", A), hv("e2", "Ann Lee", A)]
    )
    roster = _roster(S0=(None, "unknown"), S1=("Ann Lee", "host"))
    d = _diarization({"S0": list(A), "S1": list(B)}, {"S0": 900.0, "S1": 800.0})
    steps = []
    trace = types.SimpleNamespace(voice=lambda v, rung, decision, **kw: steps.append((v, decision)))
    out = vi.apply_to_roster(roster, d, tmp_path, "ep9", 0.7, trace=trace)
    assert (out.by_voice["S0"].name, out.by_voice["S0"].named, out.by_voice["S0"].source) == (
        "Ann Lee",
        True,
        "voice_identity",
    )
    assert (out.by_voice["S1"].name, out.by_voice["S1"].named) == ("S1", False)
    assert sorted(steps) == [("S0", "named"), ("S1", "removed")]
