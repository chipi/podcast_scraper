"""The naming decision, pinned over every English fixture episode — D-34 / S2.6's proof obligation.

WHY THIS EXISTS. D-34 moves speaker naming to run AFTER translation instead of before it. That is
a change to the ENGLISH path, not only the Spanish one: the same code resolves both, so a refactor
that quietly alters which voice gets which name would degrade the 678-episode English corpus while
every non-English test stayed green. S2.6 therefore states the obligation as "zero diff on the
English golden — `.txt`, `.segments.json`, speaker record", and this is that golden.

WHY IT IS NOT READ OFF THE FIXTURE CORPUS. `app-validation-corpus/v3`'s `.segments.json` already
carries resolved NAMES (`"speaker_label": "Sam"`) — it is a pre-built OUTPUT. A golden read from it
would pin the fixture and pass no matter what the code did, which is the exact trap
`test_english_artifact_allowlist.py` documents for Phase 0 and the reason that test reads the
writer instead.

So this is FUNCTIONAL. For each fixture episode it reconstructs the input the naming code actually
receives — a diarization of anonymous `SPEAKER_NN` turns plus a transcript with the names stripped
back out — runs the real resolver, and records what it decided. The inputs are constructed, the
output is computed, so a behaviour change moves the golden.

DETERMINISM. `speaker_resolution_llm=False`, so `_resolve_voices_via_llm` returns empty and the
deterministic cue path is what is measured. That is also the path four shipped profiles use
(`airgapped`, `local`, `dev`, `reprocess_dgx_no_llm`), so it is not a synthetic configuration.

Regenerate with::

    REGENERATE_NAMING_GOLDEN=1 .venv/bin/python -m pytest \
        tests/integration/workflow/test_naming_golden.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from podcast_scraper import config
from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.ml.diarization.pipeline import apply_diarization_to_result

pytestmark = pytest.mark.integration

_REPO = Path(__file__).resolve().parents[3]
_CORPUS = _REPO / "tests" / "fixtures" / "app-validation-corpus" / "v3" / "feeds"
_GOLDEN = _REPO / "tests" / "fixtures" / "goldens" / "naming_decision.golden.json"


def _episodes() -> List[Tuple[str, Path]]:
    """``(episode_key, segments_path)`` for every fixture episode, in a stable order.

    D-34 DECIDES WHICH SEGMENTS THESE ARE. Naming runs AFTER translation, on the English render, so
    for a translated episode the input the naming code actually receives is `.en.segments.json` —
    the source-language sidecar is the body naming never sees.

    This used to exclude `.en.segments.json` outright, which was right while no fixture had one and
    wrong the moment p10-p14's renders landed: it fed the resolver the SPANISH (and Italian, French,
    German, Portuguese) body for precisely the five episodes D-34 exists for. The deterministic cue
    path looks for English self-introduction cues, found none, and returned bare `SPEAKER_NN` — a
    correct answer to the wrong question, and it would have pinned "non-English episodes cannot be
    named" into the golden as if it were the design.
    """
    out: List[Tuple[str, Path]] = []
    for path in sorted(_CORPUS.glob("*/run_*/transcripts/*.segments.json")):
        if path.name.endswith((".adfree.segments.json", ".en.segments.json")):
            continue
        key = path.name[: -len(".segments.json")]
        english = path.with_name(f"{key}.en.segments.json")
        out.append((key, english if english.is_file() else path))
    return out


def _anonymize(segments: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], DiarizationResult]:
    """Strip the fixture's resolved names back to anonymous voices.

    The fixture is post-naming, so its `speaker_label` is already a person's name. Naming receives
    `SPEAKER_NN`, so each distinct label is mapped to a voice id in first-appearance order — which
    is what a diarizer produces and what `relabel_only` replays.

    Returns ``(asr_segments, diarization)``: the ASR segments carry NO speaker field (the ASR
    stage does not produce one), and the diarization carries the turns.
    """
    voice_of: Dict[str, str] = {}
    asr: List[Dict[str, Any]] = []
    turns: List[DiarizationSegment] = []
    for i, seg in enumerate(segments):
        label = str(seg.get("speaker_label") or "")
        if label not in voice_of:
            voice_of[label] = f"SPEAKER_{len(voice_of):02d}"
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or 0.0)
        asr.append({"id": i, "start": start, "end": end, "text": str(seg.get("text") or "")})
        turns.append(DiarizationSegment(start=start, end=end, speaker=voice_of[label]))
    return asr, DiarizationResult(
        segments=turns, num_speakers=len(voice_of), model_name="fixture-replay"
    )


def _cfg(output_dir: str) -> config.Config:
    """Deterministic: no network, no LLM. The cue path four shipped profiles use.

    `output_dir` is a tmp path only so the recurring-text index and the cache-dir derivation have
    somewhere to point; `precomputed_diarization` means neither the diarization cache nor a
    provider is ever consulted.
    """
    return config.Config(
        rss="https://example.com/f.xml",
        output_dir=output_dir,
        speaker_resolution_llm=False,
    )


def _decide(episode_key: str, segments_path: Path, output_dir: str) -> Dict[str, Any]:
    """The naming decision for one episode, as a comparable record."""
    raw = json.loads(segments_path.read_text(encoding="utf-8"))
    segments = [s for s in raw if isinstance(s, dict)]
    asr, diarization = _anonymize(segments)
    result = {"segments": asr, "text": " ".join(s["text"] for s in asr)}

    out = apply_diarization_to_result(
        result,
        audio_path=f"/nonexistent/{episode_key}.mp3",
        cfg=_cfg(output_dir),
        detected_speaker_names=None,
        precomputed_diarization=diarization,
        episode_title=episode_key,
        detection_ran=True,
    )

    # THE SPEAKER RECORD, per voice — the thing S2.6 names as the proof. `resolved_name` and
    # `role` are the decision; `named`, `source` and `reason` are WHY, and they are included
    # because a refactor that reaches the same name by a different route has changed the code's
    # behaviour even where the label happens to match.
    #
    # `talk_time_s` is deliberately excluded: it is an INPUT derived from the fixture's times,
    # not a decision, and including it would make the diff noisy without adding a fact.
    diag = out.get("speaker_diagnostics") or {}
    voices = [
        {
            "voice": v.get("voice"),
            "resolved_name": v.get("resolved_name"),
            "role": v.get("role"),
            "named": v.get("named"),
            "source": v.get("source"),
            "voice_type": v.get("voice_type"),
            "reason": v.get("reason"),
        }
        for v in sorted(
            (x for x in (diag.get("voices") or []) if isinstance(x, dict)),
            key=lambda x: str(x.get("voice") or ""),
        )
    ]
    # The rendered labels too: the roster is what decides, but `.txt` and `.segments.json` are
    # what ship, and S2.6 asks for both.
    labels = sorted(
        {
            str(s.get("speaker_label") or "")
            for s in out.get("segments") or []
            if isinstance(s, dict)
        }
    )
    return {
        "input_voices": diarization.num_speakers,
        "resolved_labels": labels,
        "num_speakers": out.get("diarization_num_speakers"),
        "speaker_record": voices,
    }


def _build() -> Dict[str, Any]:
    import tempfile

    with tempfile.TemporaryDirectory() as out:
        return {key: _decide(key, path, out) for key, path in _episodes()}


@pytest.fixture(scope="module")
def observed() -> Dict[str, Any]:
    return _build()


@pytest.fixture(scope="module")
def baseline() -> Dict[str, Any]:
    if not _GOLDEN.is_file():
        pytest.skip(f"no golden yet; set REGENERATE_NAMING_GOLDEN=1 to write {_GOLDEN.name}")
    return dict(json.loads(_GOLDEN.read_text(encoding="utf-8")))


def test_regenerate_the_golden() -> None:
    """Writes the golden. Skipped unless REGENERATE_NAMING_GOLDEN=1 — a golden that rewrites
    itself on every run proves nothing."""
    if os.environ.get("REGENERATE_NAMING_GOLDEN") != "1":
        pytest.skip("set REGENERATE_NAMING_GOLDEN=1 to rewrite the naming golden")
    _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    _GOLDEN.write_text(
        json.dumps(_build(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


class TestTheCorpusIsWorthPinning:
    def test_it_covers_every_fixture_episode(self, observed: Dict[str, Any]) -> None:
        assert len(observed) >= 40, f"only {len(observed)} episodes; the corpus should have 40+"

    def test_the_episodes_actually_have_multiple_VOICES(self, observed: Dict[str, Any]) -> None:
        """A one-voice corpus would pin nothing about naming."""
        multi = [k for k, v in observed.items() if int(v["input_voices"]) > 1]
        assert len(multi) >= 30, f"only {len(multi)} episodes have >1 voice"

    def test_some_voices_are_actually_NAMED(self, observed: Dict[str, Any]) -> None:
        """If every voice came back `SPEAKER_NN`, the golden would pin a FAILURE rather than the
        naming behaviour, and a refactor that broke naming entirely would still match it.

        This guard has already earned itself: the first version of this file read a
        `named_count` key that does not exist in the diagnostics, so every episode recorded
        zero names and the golden would have been a monument to nothing.
        """
        named = [
            k
            for k, v in observed.items()
            if any(x.get("named") for x in v.get("speaker_record") or [])
        ]
        assert len(named) >= 10, f"only {len(named)} episode(s) named any voice"

    def test_both_roles_appear(self, observed: Dict[str, Any]) -> None:
        """Host/guest attribution is half of what the roster decides, so the golden has to
        contain both or it cannot detect a role regression."""
        roles = {
            str(x.get("role")) for v in observed.values() for x in (v.get("speaker_record") or [])
        }
        assert {"host", "guest"} <= roles, f"roles present: {sorted(roles)}"


class TestTheNamingDecisionIsUnchanged:
    def test_no_episode_moved(self, observed: Dict[str, Any], baseline: Dict[str, Any]) -> None:
        """D-34's proof obligation. Every difference is printed, because the useful output of a
        golden failure is WHICH episode moved and how."""
        drifted = {
            key: {"golden": baseline.get(key), "now": observed.get(key)}
            for key in sorted(set(baseline) | set(observed))
            if baseline.get(key) != observed.get(key)
        }
        assert drifted == {}, (
            f"the naming decision moved on {len(drifted)} episode(s). This is the English path, "
            "so a diff here is a regression for the corpus that works today — not a consequence "
            f"of the multilingual work.\n{json.dumps(drifted, indent=2, sort_keys=True)[:4000]}"
        )

    def test_the_episode_set_is_unchanged(
        self, observed: Dict[str, Any], baseline: Dict[str, Any]
    ) -> None:
        """An episode VANISHING from the golden is the quiet failure: the comparison above would
        still pass for every episode that remained."""
        assert sorted(baseline) == sorted(observed)
