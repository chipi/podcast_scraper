"""A name the model gives to two voices names neither of them (ladder doc v45).

Rule 5, one person one voice, used to keep the voice the model listed first and refuse the
other. The listing order says nothing about which voice is the person. Measured on the stored
answers (2026-10-24): where the kept voice was served the name, it was the wrong voice 11 times
and the right one 4 times, and in each of the 11 the refused voice was the labelled person (the
host's name on the guest's voice: c064, y023; the host's name on a co-host: v114, v253, v428).
A wrong name is worse than a hidden one, so neither voice keeps it; each keeps its role.
"""

from __future__ import annotations

import json

from podcast_scraper.speaker_detectors.resolution import resolve_voices_and_roles

HOST = "Welcome back to the show. Today we look at how small lenders price risk."
GUEST = "When we started the fund, the first thing we saw was how little data there was."


def _resolve(payload: dict, report: dict | None = None):
    return resolve_voices_and_roles(
        stated_names=["Avery Quill", "Morgan Fenwick"],
        voice_texts={
            "SPEAKER_00": GUEST,
            "SPEAKER_01": HOST,
            "SPEAKER_02": HOST + " Back after this.",
        },
        complete=lambda _prompt: json.dumps({"voices": payload}),
        known_hosts=["Avery Quill"],
        episode_title="Pricing risk",
        report=report,
    )


def test_the_name_given_to_two_voices_is_kept_by_neither() -> None:
    out = _resolve(
        {
            "SPEAKER_00": {"name": "Avery Quill", "role": "host"},
            "SPEAKER_01": {"name": "Avery Quill", "role": "host"},
        }
    )
    assert all(v.name is None for v in out.values())


def test_each_voice_keeps_its_role() -> None:
    out = _resolve(
        {
            "SPEAKER_00": {"name": "Morgan Fenwick", "role": "guest"},
            "SPEAKER_01": {"name": "Morgan Fenwick", "role": "host"},
        }
    )
    assert {v: lv.role for v, lv in out.items()} == {"SPEAKER_00": "guest", "SPEAKER_01": "host"}


def test_the_other_names_are_untouched() -> None:
    out = _resolve(
        {
            "SPEAKER_00": {"name": "Morgan Fenwick", "role": "guest"},
            "SPEAKER_01": {"name": "Avery Quill", "role": "host"},
            "SPEAKER_02": {"name": "Avery Quill", "role": "host"},
        }
    )
    assert out["SPEAKER_00"].name == "Morgan Fenwick"


def test_both_verdicts_are_recorded_as_duplicate() -> None:
    report: dict = {}
    _resolve(
        {
            "SPEAKER_00": {"name": "Avery Quill", "role": "host"},
            "SPEAKER_01": {"name": "Avery Quill", "role": "host"},
        },
        report,
    )
    assert [v["outcome"] for v in report["verdicts"]] == ["duplicate", "duplicate"]
