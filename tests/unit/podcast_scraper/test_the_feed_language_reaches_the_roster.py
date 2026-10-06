"""The language reaches the bottom of the roster, not just the top of it.

Adding `language: str = TARGET_LANGUAGE` to a signature is the easy half. The failure mode is a
parameter that is declared everywhere and PASSED nowhere: every default fires, every row is the
English one, and the only evidence of the bug is that non-English behaviour never changes. These
tests assert the delivery, not the declaration.

The language in question is the SOURCE language. Everything the roster reads is ASR output, which
is in whatever was spoken — unlike `gi/speakers.py`, which reads the post-D-44 canonical body and
is therefore correctly English (see its own docstring).
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from podcast_scraper.languages import TARGET_LANGUAGE
from podcast_scraper.providers.ml.diarization import roster
from podcast_scraper.speaker_detectors import naming_vocabulary, resolution

_ROSTER_SRC = Path(roster.__file__)
_RESOLUTION_SRC = Path(resolution.__file__)


class TestTheSignaturesExist:
    @pytest.mark.parametrize(
        "module,name",
        [
            (roster, "resolve_speaker_roster"),
            (roster, "build_speaker_diagnostics"),
            (resolution, "resolve_voices_and_roles"),
            (resolution, "build_resolution_prompt"),
            (resolution, "retrieve_mentions"),
            (resolution, "refuted_by_third_person"),
            (resolution, "_addressed_at_open"),
        ],
    )
    def test_it_accepts_a_language(self, module: object, name: str) -> None:
        assert "language" in inspect.signature(getattr(module, name)).parameters

    def test_the_default_is_the_analysis_language(self) -> None:
        # Not None: the default has to reproduce what these functions did before the parameter
        # existed, which was to read the English rows.
        p = inspect.signature(roster.resolve_speaker_roster).parameters["language"]
        assert p.default == TARGET_LANGUAGE


def _calls_in(path: Path, callee: str) -> list[ast.Call]:
    """Every call to *callee* in *path*, by AST rather than by grep."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
        if name == callee:
            out.append(node)
    return out


def _passes_language(call: ast.Call) -> bool:
    if any(kw.arg == "language" for kw in call.keywords):
        return True
    # Positional delivery counts too — several of these are called positionally.
    return any(isinstance(a, ast.Name) and a.id == "language" for a in call.args)


class TestTheLanguageIsActuallyDelivered:
    """Every INTERNAL call must pass it on. A default taken by accident is the whole bug."""

    @pytest.mark.parametrize(
        "callee",
        [
            "_self_intros_by_voice",
            "_self_intro_voice_names",
            "_intro_reader_voice_names",
            "_voice_named_by_the_introduction",
            "_past_cue_head_name",
            "_metadata_anchored_self_intro",
            "_sign_off_self_intro",
            "_match_stated_in_span",
            "_span_has_contradicting_surname",
            "_name_host_voices",
            "_name_guest_voices",
            "_guest_voice_by_host_elimination",
            "_bind_introduced_name",
        ],
    )
    def test_every_roster_call_passes_the_language(self, callee: str) -> None:
        calls = _calls_in(_ROSTER_SRC, callee)
        assert calls, f"no call to {callee} found — did it get renamed?"
        missing = [c.lineno for c in calls if not _passes_language(c)]
        assert missing == [], (
            f"{callee} is called at line(s) {missing} without the language, so those paths "
            f"silently take the English default"
        )

    @pytest.mark.parametrize("callee", ["refuted_by_third_person", "_addressed_at_open"])
    def test_every_call_into_resolution_passes_the_language(self, callee: str) -> None:
        calls = _calls_in(_ROSTER_SRC, callee) + _calls_in(_RESOLUTION_SRC, callee)
        assert calls
        missing = [c.lineno for c in calls if not _passes_language(c)]
        assert missing == []

    def test_the_prompt_builder_gets_it(self) -> None:
        calls = _calls_in(_RESOLUTION_SRC, "build_resolution_prompt")
        assert calls
        assert all(_passes_language(c) for c in calls)


class TestThePipelineSuppliesIt:
    """`transcription_language(cfg)` is the one reader (S0.6); the pipeline must use THAT."""

    def test_the_pipeline_reads_the_sanctioned_reader(self) -> None:
        src = (Path(roster.__file__).parent / "pipeline.py").read_text(encoding="utf-8")
        assert "from ....languages import transcription_language" in src
        # Delivered to the roster, the diagnostics and the LLM resolution — all three.
        assert src.count("language=transcription_language(cfg)") >= 3

    def test_nothing_reads_cfg_language_directly(self) -> None:
        src = (Path(roster.__file__).parent / "pipeline.py").read_text(encoding="utf-8")
        # `scripts/check/lint_language_readers.py` enforces this repo-wide; asserting it here too
        # keeps the reason attached to the change that could break it.
        assert "cfg.language" not in src


class TestAnUnsupportedLanguageReadsNothing:
    """The two cases `vocabulary_row` exists to keep apart."""

    def test_none_means_the_analysis_row(self) -> None:
        # Nothing resolved -> the behaviour these call sites had before they took a language.
        assert (
            naming_vocabulary.vocabulary_row(naming_vocabulary.THIS_IS_INTRO, None)
            == naming_vocabulary.THIS_IS_INTRO[TARGET_LANGUAGE]
        )

    def test_a_resolved_language_gets_its_own_row(self) -> None:
        for lang in ("es", "it", "fr", "de", "pt"):
            row = naming_vocabulary.vocabulary_row(naming_vocabulary.THIS_IS_INTRO, lang)
            assert row == naming_vocabulary.THIS_IS_INTRO[lang]
            assert row != naming_vocabulary.THIS_IS_INTRO[TARGET_LANGUAGE]

    def test_a_region_subtag_resolves_to_its_primary(self) -> None:
        assert (
            naming_vocabulary.vocabulary_row(naming_vocabulary.THIS_IS_INTRO, "pt-PT")
            == naming_vocabulary.THIS_IS_INTRO["pt"]
        )

    def test_an_unsupported_language_is_not_given_english(self) -> None:
        # The point of the whole exercise: English cue patterns over Japanese is how a confident
        # wrong name gets made, and the English rows are gold-gated against an English set.
        assert naming_vocabulary.vocabulary_row(naming_vocabulary.THIS_IS_INTRO, "ja") is None
        assert (
            naming_vocabulary.vocabulary_row(
                naming_vocabulary.INTRO_AFFILIATION_TOKENS, "ja", default=frozenset()
            )
            == frozenset()
        )

    def test_the_guards_abstain_rather_than_refute(self) -> None:
        # `_addressed_at_open` REFUTES a name ("this voice greets them, so it is not them"). With
        # no greeting row there is no evidence either way, and returning True would refute a
        # correct name on no grounds at all.
        assert resolution._addressed_at_open("Hey, Jordan. Good morning.", "Jordan", "ja") is False
        # ...while the supported language still fires.
        assert resolution._addressed_at_open("Hey, Jordan. Good morning.", "Jordan", "en") is True

    def test_a_greeting_is_read_in_its_own_language(self) -> None:
        assert resolution._addressed_at_open("Olá, Inês. Bom dia.", "Inês", "pt") is True
        # ...and the English row does not see it, which is why the row had to exist.
        assert resolution._addressed_at_open("Olá, Inês. Bom dia.", "Inês", "en") is False


#: Callers of the publish gate that deliberately stay English-only. Both are main's one-shot
#: corpus cleanups (2026-10-03), dry-run on the prod corpus against FROZEN KEEP/REMOVE sets; that
#: corpus has no non-English feed, so a language there would change nothing they were measured on
#: and would make their result depend on a per-artifact lookup they never did.
_GATE_CALLERS_ENGLISH_BY_DESIGN = frozenset(
    {
        "upgrade/migrations/m0015_unpublishable_speaker_names_removed.py",
        "upgrade/migrations/m0017_speaker_names_canonicalised.py",
    }
)


class TestThePublishGateIsAskedInTheFeedsLanguage:
    """`is_publishable_speaker_name` took a language and was then called without one.

    Measured before this: the gate refused "Host Mike" and accepted "Anfitrión Miguel", so a
    Spanish feed minted a person named after its role word. Giving the gate a Spanish row fixed
    nothing while every caller kept the English default — the same declared-but-never-passed
    failure the module docstring describes, one layer further out.
    """

    def test_every_pipeline_call_passes_a_language(self) -> None:
        src_root = Path(roster.__file__).parents[3]
        missing = []
        for path in sorted(src_root.rglob("*.py")):
            rel = path.relative_to(src_root).as_posix()
            if rel in _GATE_CALLERS_ENGLISH_BY_DESIGN:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
                if name != "is_publishable_speaker_name":
                    continue
                if not any(kw.arg == "language" for kw in node.keywords):
                    missing.append(f"{rel}:{node.lineno}")
        assert not missing, f"publish gate called without a language: {missing}"

    def test_the_exemptions_still_exist(self) -> None:
        # An exemption for a file that is gone would silently exempt its replacement's name.
        src_root = Path(roster.__file__).parents[3]
        for rel in _GATE_CALLERS_ENGLISH_BY_DESIGN:
            assert (src_root / rel).is_file(), rel

    def test_a_spanish_role_word_does_not_reach_the_host_pool(self) -> None:
        from podcast_scraper.speaker_detectors.hosts import compose_episode_hosts

        feed_hosts = ["Anfitrión Miguel", "Lucía Herrera"]
        assert compose_episode_hosts(feed_hosts, language="es") == ["Lucía Herrera"]

    def test_the_english_path_did_not_move(self) -> None:
        from podcast_scraper.speaker_detectors.hosts import compose_episode_hosts

        # With no language the gate is the English row alone — exactly what it was before.
        assert compose_episode_hosts(["Host Mike", "Casey Rowe"]) == ["Casey Rowe"]

    def test_the_feed_language_rides_on_the_host_detection_result(self) -> None:
        from types import SimpleNamespace

        from podcast_scraper.workflow.stages.processing import hosts_for_episode
        from podcast_scraper.workflow.types import HostDetectionResult

        result = HostDetectionResult(
            {"Anfitrión Miguel", "Lucía Herrera"}, None, None, "Sesiones", language="es-ES"
        )
        episode = SimpleNamespace(title="Episodio 1", item=None)
        assert hosts_for_episode(result, episode) == {"Lucía Herrera"}

    def test_the_roster_refuses_a_spanish_role_word_on_a_voice(self) -> None:
        assert roster._is_person_voice_label("Anfitrión Miguel", "es") is False
        assert roster._is_person_voice_label("Lucía Herrera", "es") is True
