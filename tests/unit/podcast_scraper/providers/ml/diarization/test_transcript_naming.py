"""Transcript naming: the model names every voice from the whole transcript (see the module)."""

from __future__ import annotations

import pytest

from podcast_scraper.providers.ml.diarization import transcript_naming as tn
from podcast_scraper.providers.ml.diarization.naming_trace import NamingTrace
from podcast_scraper.providers.ml.diarization.roster import SpeakerRole, SpeakerRoster

pytestmark = [pytest.mark.unit]

WORDS = " ".join(["word"] * 45)

# The prompt measured on the labelled naming sets (podcast-scraper-eval-data,
# autoresearch/naming_rethink/qwen_verify.py). The prompt-store templates must render it byte
# for byte.
MEASURED_PROMPT = """\
You identify the speakers of one podcast episode. Below are the feed and episode
metadata, then the FULL diarized transcript: each line is "[mm:ss] SPEAKER_xx: text". Voice ids
carry no names.

For EVERY voice listed at the end, decide:
- name: the real person this voice is, or null. Use only this file: self-introductions, being
  introduced or welcomed by another voice (and who speaks next), being addressed by name, the
  metadata naming hosts and guests, sign-offs. Diarization can merge two people into one voice or
  split one person into two voices; judge a voice by most of its speech. Two voices may be the same
  person. A voice that only TALKS ABOUT a person is not that person.
- role: host | guest | ad | promo | clip | unknown (ad = an advertisement read; promo = a trailer or
  promotion for another show or product; clip = an archival or quoted recording; unknown = mixed or
  unclear).
- confidence: high | medium | low.
- evidence: a short verbatim quote with its [mm:ss] that supports the name, or "" when name is null.
A wrong name is worse than a missing one: when unsure, null.

Return JSON only:
{"voices": {"SPEAKER_00": {"name": "Full Name or null", "role": "...", \
"confidence": "...", "evidence": "..."}}}

@TEXT@

VOICES TO JUDGE: @VOICES@
"""


def _aligned():
    return [
        (
            {"start": 0.0, "end": 5.0, "text": "Welcome to the show, I'm Russ Roberts."},
            "SPEAKER_00",
        ),
        ({"start": 5.0, "end": 9.0, "text": WORDS}, "SPEAKER_00"),
        ({"start": 9.0, "end": 60.0, "text": "Thanks for having me, Russ. " + WORDS}, "SPEAKER_01"),
        ({"start": 61.0, "end": 64.0, "text": "This episode is brought to you by"}, "SPEAKER_02"),
        ({"start": 64.0, "end": 65.0, "text": "uh"}, None),
    ]


def _roster():
    return SpeakerRoster(
        by_voice={
            # The ladder swapped the two people.
            "SPEAKER_00": SpeakerRole(
                name="Stephen Witt", role="guest", named=True, source="llm_resolution"
            ),
            "SPEAKER_01": SpeakerRole(
                name="Russ Roberts", role="host", named=True, source="llm_resolution"
            ),
            "SPEAKER_02": SpeakerRole(
                name="Daniel Atkinson", role="guest", named=True, source="self_intro"
            ),
        },
        num_speakers=3,
    )


SWAP_FIX = (
    '{"voices": {"SPEAKER_00": {"name": "Russ Roberts", "role": "host", "confidence": "high",'
    ' "evidence": "[00:00] I\'m Russ Roberts"},'
    ' "SPEAKER_01": {"name": "Stephen Witt", "role": "guest", "confidence": "medium",'
    ' "evidence": ""}}}'
)


def _ask(complete, aligned=None, **kw):
    return tn.ask(
        aligned or _aligned(),
        complete,
        feed_title="EconTalk",
        feed_description=None,
        episode_title="Ep 1",
        episode_description=None,
        **kw,
    )


def _sent(aligned, feed, feed_desc, title, desc) -> str:
    """The transcript text inside the prompt production sends (``fit_prompt``, no clipping)."""
    prompt, voices, _clip = tn.fit_prompt(
        tn.header_lines(feed, feed_desc, title, desc), tn.turn_lines(aligned), None
    )
    assert prompt is not None
    head, _, rest = MEASURED_PROMPT.partition("@TEXT@")
    tail = rest.replace("@VOICES@", ", ".join(voices))
    assert prompt.startswith(head) and prompt.endswith(tail)
    return prompt[len(head) : len(prompt) - len(tail)]


def test_the_transcript_block_merges_turns_and_keeps_the_measured_layout() -> None:
    text = _sent(_aligned(), "EconTalk", "Hosted by Russ Roberts.", "Ep 1", "With Stephen Witt.")
    lines = text.splitlines()
    assert lines[:6] == [
        "FEED: EconTalk",
        "FEED DESCRIPTION: Hosted by Russ Roberts.",
        "EPISODE: Ep 1",
        "EPISODE DESCRIPTION: With Stephen Witt.",
        "",
        "TRANSCRIPT (diarized voices; no names attached):",
    ]
    assert lines[6].startswith("[00:00] SPEAKER_00: Welcome to the show, I'm Russ Roberts. word")
    assert lines[7].startswith("[00:09] SPEAKER_01: Thanks for having me, Russ.")
    assert lines[8] == "[01:01] SPEAKER_02: This episode is brought to you by"
    assert lines[9] == "[01:04] UNATTRIBUTED: uh"
    assert tn.voices_to_judge(text) == ["SPEAKER_00", "SPEAKER_01"]


def test_the_prompt_store_renders_the_measured_prompt_byte_for_byte() -> None:
    text = _sent(_aligned(), "EconTalk", None, "Ep 1", None)
    voices = ["SPEAKER_00", "SPEAKER_01"]
    expected = MEASURED_PROMPT.replace("@TEXT@", text).replace("@VOICES@", ", ".join(voices))
    assert tn.build_prompt(text, voices) == expected


def test_the_answer_reads_null_strings_as_no_name_and_keeps_confidence_and_evidence() -> None:
    raw = (
        '{"voices": {"SPEAKER_00": {"name": "null", "role": "ad"},'
        ' "SPEAKER_01": {"name": "Kim Lee", "role": "Guest", "confidence": "HIGH",'
        ' "evidence": "[01:00] I\'m Kim"}}}'
    )
    full = tn.parse_answer_full("<think>x</think>" + raw)
    assert full is not None
    assert full["SPEAKER_00"]["name"] is None and full["SPEAKER_00"]["role"] == "ad"
    assert full["SPEAKER_01"] == {
        "name": "Kim Lee",
        "role": "guest",
        "confidence": "high",
        "evidence": "[01:00] I'm Kim",
    }
    assert tn.parse_answer_full("not json") is None
    assert tn.parse_answer_full('{"speakers": {}}') is None


def test_a_swap_is_corrected_and_a_voice_it_does_not_call_a_participant_is_unnamed() -> None:
    def complete(prompt: str) -> str:
        assert "VOICES TO JUDGE: SPEAKER_00, SPEAKER_01" in prompt
        return SWAP_FIX

    asked = _ask(complete)
    assert asked.ok and asked.voices_asked == ["SPEAKER_00", "SPEAKER_01"]
    trace = NamingTrace()
    out = tn.apply_answer(_roster(), asked.pairs(), trace, language="en")
    assert (out.by_voice["SPEAKER_00"].name, out.by_voice["SPEAKER_00"].role) == (
        "Russ Roberts",
        "host",
    )
    assert (out.by_voice["SPEAKER_01"].name, out.by_voice["SPEAKER_01"].role) == (
        "Stephen Witt",
        "guest",
    )
    assert out.by_voice["SPEAKER_00"].source == tn.SOURCE
    assert out.by_voice["SPEAKER_02"].named is False
    assert trace.voices["SPEAKER_02"][0]["decision"] == "removed"


def test_a_name_the_model_confirms_keeps_the_ladders_source() -> None:
    # The record keeps a host named like the show only when the voice introduced itself
    # (metadata_generation._build_speaker_record): confirming the name must not erase that.
    roster = SpeakerRoster(
        by_voice={
            "SPEAKER_00": SpeakerRole(
                name="Peter Attia", role="host", named=True, source="self_intro"
            ),
        },
        num_speakers=1,
    )
    out = tn.apply_answer(roster, {"SPEAKER_00": ("peter attia", "host")}, language="en")
    assert out.by_voice["SPEAKER_00"].source == "self_intro"
    assert out.by_voice["SPEAKER_00"].name == "Peter Attia"


def test_a_failed_skipped_or_unreadable_call_is_reported_not_raised() -> None:
    def boom(prompt: str) -> str:
        raise RuntimeError("gateway down")

    failed = _ask(boom)
    assert not failed.ok and failed.report["error"] == "RuntimeError: gateway down"
    unreadable = _ask(lambda p: "sorry")
    assert not unreadable.ok and unreadable.report["error"] == "unreadable answer"
    assert unreadable.report["raw"] == "sorry"
    short = [({"start": 0.0, "end": 1.0, "text": "hi"}, "SPEAKER_00")]
    skipped = _ask(lambda p: pytest.fail("must not be called"), aligned=short)
    assert not skipped.ok and "skipped" in skipped.report


def _long_aligned(n: int):
    out = [({"start": 0.0, "end": 1.0, "text": "I'm the host, Ana Ruiz. " + WORDS}, "SPEAKER_00")]
    for i in range(1, n - 1):
        out.append(
            ({"start": float(i * 10), "end": float(i * 10 + 9), "text": WORDS}, f"SPEAKER_0{i % 2}")
        )
    out.append(
        ({"start": float(n * 10), "end": 0.0, "text": "Bye, I'm Ana. " + WORDS}, "SPEAKER_00")
    )
    return out


def test_a_transcript_too_long_for_the_window_loses_middle_turns_only() -> None:
    sent = []

    def complete(prompt: str) -> str:
        sent.append(prompt)
        return SWAP_FIX

    asked = _ask(complete, aligned=_long_aligned(200), window_tokens=8000)
    assert asked.ok
    clip = asked.report["clip"]
    assert clip["turns_dropped"] > 0 and clip["prompt_tokens"] <= 8000 - tn.MAX_OUTPUT_TOKENS
    prompt = sent[0]
    assert "I'm the host, Ana Ruiz." in prompt and "Bye, I'm Ana." in prompt
    assert f"[... {clip['turns_dropped']} turns omitted for length ...]" in prompt


def test_an_unknown_window_sends_the_whole_transcript() -> None:
    asked = _ask(lambda p: SWAP_FIX, aligned=_long_aligned(200))
    assert asked.report["clip"]["turns_dropped"] == 0
    assert asked.report["clip"]["window_tokens"] is None


def test_a_window_nothing_fits_is_an_error_and_no_call() -> None:
    asked = _ask(lambda p: pytest.fail("must not be called"), window_tokens=tn.MAX_OUTPUT_TOKENS)
    assert not asked.ok and asked.report["error"] == "transcript does not fit the window"


def test_the_sidecar_compares_the_ladder_and_the_model_voice_by_voice() -> None:
    asked = _ask(lambda p: SWAP_FIX)
    ladder = SpeakerRoster(
        by_voice={
            "SPEAKER_00": SpeakerRole(name="Stephen Witt", role="guest", named=True, source="x"),
            "SPEAKER_01": SpeakerRole(name="SPEAKER_01", role="unknown", named=False, source="x"),
            "SPEAKER_02": SpeakerRole(name="Ad Voice", role="guest", named=True, source="x"),
            "SPEAKER_03": SpeakerRole(name="Russ", role="host", named=True, source="x"),
        },
        num_speakers=4,
    )
    model = tn.apply_answer(
        ladder, {**asked.pairs(), "SPEAKER_03": ("Russ", "host")}, language="en"
    )
    side = tn.sidecar("shadow", asked, ladder, model)
    assert side["version"] == tn.SIDECAR_VERSION and side["mode"] == "shadow"
    assert side["call"] == "standalone" and side["ok"] is True
    v = side["voices"]
    assert v["SPEAKER_00"]["outcome"] == "changes"
    assert v["SPEAKER_01"]["outcome"] == "adds"
    assert v["SPEAKER_02"]["outcome"] == "removes"
    assert v["SPEAKER_03"]["outcome"] == "agree"
    assert v["SPEAKER_00"]["model"]["confidence"] == "high"
    assert side["outcomes"] == {"agree": 1, "adds": 1, "removes": 1, "changes": 1}
    failed = tn.sidecar("shadow", _ask(lambda p: "x"), ladder, None)
    assert failed["ok"] is False and "outcomes" not in failed
    assert "outcome" not in failed["voices"]["SPEAKER_00"]


def test_modes() -> None:
    from types import SimpleNamespace

    from podcast_scraper import config

    assert config.Config.model_fields["transcript_naming"].default == "off"
    assert tn.mode_of(SimpleNamespace(transcript_naming="shadow")) == "shadow"
    assert tn.mode_of(SimpleNamespace(transcript_naming="Publish")) == "publish"
    assert tn.mode_of(SimpleNamespace(transcript_naming="bogus")) == "off"
    assert tn.mode_of(SimpleNamespace()) == "off"


def test_the_publish_gate_reads_the_feeds_language() -> None:
    # A Spanish role word the model returns as a name is refused on a Spanish feed (the gate's
    # Spanish row), exactly as the ladder refuses it.
    roster = SpeakerRoster(
        by_voice={
            "SPEAKER_00": SpeakerRole(name="SPEAKER_00", role="unknown", named=False, source="x")
        },
        num_speakers=1,
    )
    answer = {"SPEAKER_00": ("Anfitrión Miguel", "host")}
    assert tn.apply_answer(roster, answer, language="es").by_voice["SPEAKER_00"].named is False
