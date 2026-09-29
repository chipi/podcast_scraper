"""The translate client: the prompt it sends, the model it refuses to trust, and per-unit failure.

NOTHING HERE TOUCHES THE NETWORK. Every test monkeypatches ``urlopen``. A unit test that could
reach `dgx-llm-1:8005` would be the #1527 bug class again — a suite that silently depends on a
GPU box being up, and that translates real text when someone runs it on a laptop.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from podcast_scraper.providers.translate import (
    dgx_vllm_translate as mod,
    DgxVllmTranslateClient,
    render_translate_prompt,
    TranslateError,
    TranslateUnavailable,
)

pytestmark = pytest.mark.unit

BASE = "http://translator.invalid:8005/v1"
MODEL = "google/translategemma-12b-it"


class _Cfg:
    translate_api_base = BASE
    translate_model = MODEL
    translate_api_key = "EMPTY"


class _Resp:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_Resp":
        return self

    def __exit__(self, *_a: Any) -> None:
        return None


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    models: Optional[List[str]] = None,
    completion: Any = "Hello world.",
    fail_times: int = 0,
    finish_reason: str = "stop",
) -> List[Dict[str, Any]]:
    """Fake the endpoint. Returns the list of request bodies actually posted."""
    sent: List[Dict[str, Any]] = []
    state = {"failures": fail_times}

    def fake_urlopen(req: Any, timeout: int = 0) -> _Resp:
        url = req.full_url
        if url.endswith("/models"):
            ids = [MODEL] if models is None else models
            return _Resp({"data": [{"id": i} for i in ids]})
        sent.append(json.loads(req.data.decode("utf-8")))
        if state["failures"] > 0:
            state["failures"] -= 1
            raise OSError("connection reset")
        if completion is None:
            return _Resp({"choices": []})
        return _Resp({"choices": [{"text": completion, "finish_reason": finish_reason}]})

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
    return sent


class TestThePrompt:
    def test_it_is_byte_identical_to_the_template_the_model_documents(self) -> None:
        """Pinned exactly. The chat route is unusable (ADR-156 §3), so this string IS the API —
        a stray space or a reordered clause is a silent quality regression nobody would see in
        an artifact."""
        assert render_translate_prompt("Hola mundo.", source_language="es") == (
            "<start_of_turn>user\n"
            "You are a professional Spanish (es) to English (en) translator. Your goal is to "
            "accurately convey the meaning and nuance of the original text.\n"
            "\n"
            "Hola mundo.<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

    def test_a_regional_subtag_is_normalized_to_the_base_language(self) -> None:
        assert "Spanish (es) to English (en)" in render_translate_prompt(
            "Hola.", source_language="es-ES"
        )

    def test_an_undeclared_language_refuses_rather_than_guessing_a_name(self) -> None:
        """The failure this prevents is not an error, it is fluent wrong output.

        Handing the model a guessed language name produces a confident translation nobody can
        tell is wrong from the artifact. So a language absent from config/languages.yaml cannot
        be translated at all.
        """
        with pytest.raises(TranslateError, match="not declared in config/languages.yaml"):
            render_translate_prompt("Kaixo.", source_language="eu")

    def test_translating_a_language_into_itself_is_refused(self) -> None:
        with pytest.raises(TranslateError, match="into itself"):
            render_translate_prompt("Hello.", source_language="en", target_language="en")

    def test_no_usable_tag_is_refused(self) -> None:
        with pytest.raises(TranslateError, match="no usable language tag"):
            render_translate_prompt("x", source_language="   ")


class TestTheServedModelCheck:
    def test_a_different_model_on_the_slot_refuses_to_translate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ADR-143/144. A corpus attributed to the wrong translation model cannot be told from a
        correct one after the fact, so this fails the run rather than producing one."""
        _install(monkeypatch, models=["some/other-model"])
        client = DgxVllmTranslateClient(_Cfg())
        with pytest.raises(TranslateUnavailable, match="not 'google/translategemma-12b-it'"):
            client.translate("Hola.", source_language="es")

    def test_an_unreachable_endpoint_only_warns(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Unreachable is not a mismatch. Hard-failing here would make importing this module
        offline impossible, and the real call surfaces connectivity anyway."""

        def fake_urlopen(req: Any, timeout: int = 0) -> Any:
            raise OSError("no route to host")

        monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
        client = DgxVllmTranslateClient(_Cfg(), max_attempts=1)
        result = client.translate("Hola.", source_language="es")
        assert result.ok is False, "the call fails, but not with a mismatch error"
        assert any("could not verify the served model" in r.message for r in caplog.records)

    def test_the_check_runs_once_not_per_unit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = {"models": 0}

        def fake_urlopen(req: Any, timeout: int = 0) -> _Resp:
            if req.full_url.endswith("/models"):
                calls["models"] += 1
                return _Resp({"data": [{"id": MODEL}]})
            return _Resp({"choices": [{"text": "Hello."}]})

        monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
        client = DgxVllmTranslateClient(_Cfg())
        for _ in range(3):
            client.translate("Hola.", source_language="es")
        assert calls["models"] == 1


class TestOneUnit:
    def test_a_translation_comes_back_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, completion="  Hello world.  ")
        got = DgxVllmTranslateClient(_Cfg()).translate("Hola mundo.", source_language="es")
        assert got.ok and got.text == "Hello world."
        assert got.attempts == 1

    def test_the_request_is_deterministic_and_stops_at_the_turn_end(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Temperature 0 because a translation that changes between runs makes the provenance
        hash written onto every claim (S2.11) meaningless."""
        sent = _install(monkeypatch)
        DgxVllmTranslateClient(_Cfg()).translate("Hola.", source_language="es")
        assert sent[0]["temperature"] == 0.0
        assert sent[0]["stop"] == ["<end_of_turn>"]
        assert sent[0]["model"] == MODEL

    def test_a_transport_failure_is_retried_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch, fail_times=2)
        got = DgxVllmTranslateClient(_Cfg(), max_attempts=3).translate(
            "Hola.", source_language="es"
        )
        assert got.ok and got.attempts == 3

    def test_a_unit_that_never_succeeds_fails_WITHOUT_raising(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed unit is reported, not raised — so the CALLER can see how many failed.

        This docstring used to say "one bad unit must not cost the whole transcript", which is
        an episode-level policy the client cannot see enough to set, and which RFC-124 §5.3
        decides the other way: an episode without a COMPLETE English set skips summary, GI and
        KG. Reporting is what lets that decision happen where the counts are visible.
        """
        _install(monkeypatch, fail_times=99)
        got = DgxVllmTranslateClient(_Cfg(), max_attempts=2).translate(
            "Hola.", source_language="es"
        )
        assert got.ok is False
        assert got.text is None
        assert got.error and "OSError" in got.error

    def test_a_TRUNCATED_completion_is_a_failure_not_a_short_translation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The dangerous case, and the one this client originally got wrong.

        A unit cut off at `max_tokens` comes back non-empty, so every check built on "did we get
        text back" passes it. A silently short translation is worse than a missing one: the gate
        that would have caught a missing unit never fires, and the summary reads coherent.
        """
        _install(monkeypatch, completion="Hello wor", finish_reason="length")
        got = DgxVllmTranslateClient(_Cfg(), max_attempts=1).translate(
            "Hola mundo, esto es una frase larga.", source_language="es"
        )
        assert got.ok is False
        assert got.error and "truncated" in got.error

    def test_a_normally_finished_completion_records_its_finish_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other direction — `stop` must not be mistaken for truncation."""
        _install(monkeypatch, completion="Hello world.", finish_reason="stop")
        got = DgxVllmTranslateClient(_Cfg()).translate("Hola mundo.", source_language="es")
        assert got.ok and got.finish_reason == "stop"

    def test_an_empty_completion_is_a_failure_not_a_translation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty string is not a usable translation of non-empty input. Accepting it would
        put a silently blank turn into the English render and look successful."""
        _install(monkeypatch, completion="   ")
        got = DgxVllmTranslateClient(_Cfg(), max_attempts=1).translate(
            "Hola.", source_language="es"
        )
        assert got.ok is False

    def test_empty_input_needs_no_call_at_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sent = _install(monkeypatch)
        got = DgxVllmTranslateClient(_Cfg()).translate("   ", source_language="es")
        assert got.ok and got.text == ""
        assert sent == [], "no request should have been made"

    def test_an_unconfigured_client_raises_rather_than_silently_passing_text_through(
        self,
    ) -> None:
        """Returning the source text unchanged would put Spanish into `.en.txt`, which every
        later stage would then read as English."""

        class _Empty:
            translate_api_base = None
            translate_model = None

        with pytest.raises(TranslateUnavailable):
            DgxVllmTranslateClient(_Empty()).translate("Hola.", source_language="es")


class TestManyUnits:
    def test_order_is_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A unit's POSITION is its identity downstream — the English render emits one
        pseudo-segment per unit carrying `unit_id` (S2.4), so a reordered result set would
        silently reattribute text to the wrong turn."""

        def fake_urlopen(req: Any, timeout: int = 0) -> _Resp:
            if req.full_url.endswith("/models"):
                return _Resp({"data": [{"id": MODEL}]})
            body = json.loads(req.data.decode("utf-8"))
            source = body["prompt"].split("\n\n", 1)[1].split("<end_of_turn>")[0]
            return _Resp({"choices": [{"text": f"EN:{source}"}]})

        monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
        units = [f"unidad {i}" for i in range(12)]
        got = DgxVllmTranslateClient(_Cfg()).translate_many(
            units, source_language="es", concurrency=4
        )
        assert [r.text for r in got] == [f"EN:{u}" for u in units]

    def test_one_failed_unit_does_not_take_the_others_down(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_urlopen(req: Any, timeout: int = 0) -> _Resp:
            if req.full_url.endswith("/models"):
                return _Resp({"data": [{"id": MODEL}]})
            body = json.loads(req.data.decode("utf-8"))
            if "dos" in body["prompt"]:
                raise OSError("connection reset")
            return _Resp({"choices": [{"text": "ok"}]})

        monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
        got = DgxVllmTranslateClient(_Cfg(), max_attempts=2).translate_many(
            ["uno", "dos", "tres"], source_language="es", concurrency=2
        )
        assert [r.ok for r in got] == [True, False, True]

    def test_no_units_means_no_work(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sent = _install(monkeypatch)
        assert DgxVllmTranslateClient(_Cfg()).translate_many([], source_language="es") == []
        assert sent == []
