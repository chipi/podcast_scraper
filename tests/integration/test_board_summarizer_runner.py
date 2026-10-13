"""The board-summarizer sidecar runner (RFC-127), loaded by path as the digest runner's tests do.

These lock what the sidecar adds on top of ``board_summaries.sweep``: the kill switch is read
before the model is even built, a model that cannot be reached ends the cycle quietly, a provider
that would fail over to another (unpriced) one is refused, so is a summarizer that carries no use case, BART/LED summarise the material as a document,
a paid profile never makes a call, and a
bad cycle never kills the loop. The full cycle runs the real ``prod_dgx_full`` provider with only its
HTTP client mocked.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

pytest.importorskip("fastapi")  # _run_once imports the server package

pytestmark = pytest.mark.integration

_RUNNER = Path(__file__).resolve().parents[2] / "infra" / "deploy" / "board_summarizer.py"
_CORPUS = Path(__file__).resolve().parents[1] / "fixtures" / "app-validation-corpus" / "v3"


def _load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("BOARD_SUMMARY_HEARTBEAT_FILE", str(tmp_path / "hb" / "tick"))
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "app"))
    monkeypatch.setenv("BOARD_SUMMARY_CORPUS_ROOT", str(tmp_path / "corpus"))
    spec = importlib.util.spec_from_file_location("board_summarizer_under_test", _RUNNER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_profile_is_the_dgx_one_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BOARD_SUMMARY_PROFILE", raising=False)
    assert _load(tmp_path, monkeypatch).PROFILE == "prod_dgx_full"


def test_switched_off_never_builds_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _load(tmp_path, monkeypatch)
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "board_summaries_settings.json").write_text(json.dumps({"enabled": False}))

    def boom() -> Any:
        raise AssertionError("the model was built while switched off")

    monkeypatch.setattr(mod, "_provider", boom)
    mod._run_once()
    assert '"stopped": "switched_off"' in capsys.readouterr().out


def test_a_model_that_is_away_ends_the_cycle_quietly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _load(tmp_path, monkeypatch)

    def away() -> Any:
        raise ConnectionError("dgx-llm-1:8003 refused")

    monkeypatch.setattr(mod, "_provider", away)
    mod._run_once()
    assert '"stopped": "model_unreachable"' in capsys.readouterr().out


def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The served-model check is the one network call ``initialize`` makes; nothing else here
    leaves the process."""
    from podcast_scraper.providers.vllm.vllm_provider import VLLMProvider

    monkeypatch.setattr(VLLMProvider, "_verify_served_model", lambda self: None)


def test_the_dgx_profile_builds_the_vllm_provider_with_no_failover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from podcast_scraper.providers.vllm.vllm_provider import VLLMProvider

    _offline(monkeypatch)
    mod = _load(tmp_path, monkeypatch)
    monkeypatch.setattr(mod, "KEY_FILE", tmp_path / "absent")
    cfg, provider = mod._provider()
    assert type(provider) is VLLMProvider
    assert cfg.summary_provider == "vllm" and list(cfg.summary_fallback_providers or []) == []


def test_a_provider_that_would_fail_over_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from podcast_scraper.summarization.fallback import FallbackAwareSummarizationProvider

    mod = _load(tmp_path, monkeypatch)
    primary = Mock()
    wrapped = FallbackAwareSummarizationProvider(primary, ["openai"], Mock())
    monkeypatch.setattr(mod, "_provider", lambda: (Mock(summary_model="m"), wrapped))
    mod._run_once()
    assert '"stopped": "failover_configured"' in capsys.readouterr().out
    primary.summarize.assert_not_called()


def test_a_summarizer_without_the_use_case_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every summarizer the factory builds carries the use case; one that does not would summarise
    an empty transcript and ignore the board, so it stops the cycle."""
    from podcast_scraper import config

    mod = _load(tmp_path, monkeypatch)
    bare = Mock(spec=["summarize", "initialize"])
    kw: dict[str, Any] = {"rss_url": "https://example.com/feed.xml"}
    cfg = config.Config(**kw)
    monkeypatch.setattr(mod, "_provider", lambda: (cfg, bare))
    mod._run_once()
    assert '"stopped": "unsupported_provider"' in capsys.readouterr().out
    bare.summarize.assert_not_called()


def test_a_bart_led_profile_summarises_the_material_as_a_document_at_no_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from podcast_scraper import config
    from podcast_scraper.providers.ml.ml_provider import MLProvider
    from podcast_scraper.server import board_summaries

    monkeypatch.setenv("BOARD_SUMMARY_CORPUS_ROOT", str(_CORPUS))
    mod = _load(tmp_path, monkeypatch)
    monkeypatch.setattr(mod, "CORPUS_ROOT", _CORPUS)
    uid, cid = _board_with_two_episodes(tmp_path)
    kw: dict[str, Any] = {
        "rss_url": "https://example.com/feed.xml",
        "summary_provider": "transformers",
        "transcribe_missing": False,
        "auto_speakers": False,
    }
    cfg = config.Config(**kw)
    provider = MLProvider(cfg)
    provider._transformers_initialized = True
    provider._map_model = Mock(model_name="facebook/bart-large-cnn", device="cpu")
    provider._reduce_model = provider._map_model
    run = Mock(return_value="A board about risk, summarised by BART.")
    monkeypatch.setattr(
        "podcast_scraper.providers.ml.ml_provider.summarizer.summarize_long_text", run
    )
    monkeypatch.setattr(
        "podcast_scraper.providers.ml.ml_provider.ModelRegistry.get_capabilities",
        lambda *a: Mock(max_input_tokens=1024),
    )
    monkeypatch.setattr(mod, "_provider", lambda: (cfg, provider))
    mod._run_once()

    assert '"event": "board_summary.written"' in capsys.readouterr().out
    stored = board_summaries.read_summaries(tmp_path / "app", uid)[cid]
    assert stored["text"] == "A board about risk, summarised by BART."
    assert stored["prompt"].startswith("transformers/summarization/board_document_v1@")
    assert "+" not in stored["prompt"]  # the chat templates are not what BART reads
    assert run.call_args.kwargs["reduce_max_new_tokens"] == 400
    assert board_summaries.spent_today(tmp_path / "app", time.time()) == 0.0


def _board_with_two_episodes(tmp_path: Path) -> tuple[str, str]:
    from podcast_scraper.server import app_collections_store
    from podcast_scraper.server.app_slugs import _slug_index

    uid = "u_" + "a" * 24
    data_dir = tmp_path / "app"
    row, _ = app_collections_store.create_collection(data_dir, uid, "B")
    for slug in sorted(_slug_index(_CORPUS))[:2]:
        app_collections_store.add_item(data_dir, uid, row["id"], {"kind": "episode", "ref": slug})
    path = data_dir / "users" / uid / "collections.json"
    doc = json.loads(path.read_text("utf-8"))
    for c in doc["collections"]:
        c["updated_at"] = int(time.time()) - 3600
    path.write_text(json.dumps(doc), "utf-8")
    return uid, str(row["id"])


def _chat_reply(text: str) -> Mock:
    resp = Mock()
    resp.choices = [Mock()]
    resp.choices[0].message.content = text
    resp.choices[0].finish_reason = "stop"
    resp.usage = Mock(prompt_tokens=900, completion_tokens=60)
    return resp


def test_a_full_cycle_on_the_dgx_profile_writes_the_summary_at_no_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from podcast_scraper.server import board_summaries

    _offline(monkeypatch)
    monkeypatch.setenv("BOARD_SUMMARY_CORPUS_ROOT", str(_CORPUS))
    mod = _load(tmp_path, monkeypatch)
    monkeypatch.setattr(mod, "CORPUS_ROOT", _CORPUS)
    monkeypatch.setattr(mod, "KEY_FILE", tmp_path / "absent")
    uid, cid = _board_with_two_episodes(tmp_path)
    client = Mock()
    client.chat.completions.create.return_value = _chat_reply("A board about risk and couplings.")
    real = mod._provider

    def provider() -> Any:
        cfg, prov = real()
        prov.client = client
        return cfg, prov

    monkeypatch.setattr(mod, "_provider", provider)
    mod._run_once()

    out = capsys.readouterr().out
    assert '"event": "board_summary.written"' in out
    stored = board_summaries.read_summaries(tmp_path / "app", uid)[cid]
    assert stored["text"] == "A board about risk and couplings."
    assert stored["model"] == client.chat.completions.create.call_args.kwargs["model"]
    assert stored["prompt"].startswith("openai/summarization/board_system_v1@")
    assert "+openai/summarization/board_user_v1@" in stored["prompt"]
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["max_tokens"] == 400
    assert "a short summary of a listener's board" in kwargs["messages"][0]["content"]
    assert board_summaries.spent_today(tmp_path / "app", time.time()) == 0.0


def test_a_summllama_profile_runs_the_cycle_locally_at_no_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SummLlama is a local chat model: the board use case runs on it as on the DGX, priced at
    $0 from the pricing table, recorded under its own model id."""
    import sys

    from podcast_scraper import config
    from podcast_scraper.providers.ml.summllama_provider import SummLlamaProvider
    from podcast_scraper.server import board_summaries

    monkeypatch.setenv("BOARD_SUMMARY_CORPUS_ROOT", str(_CORPUS))
    mod = _load(tmp_path, monkeypatch)
    monkeypatch.setattr(mod, "CORPUS_ROOT", _CORPUS)
    uid, cid = _board_with_two_episodes(tmp_path)
    kw: dict[str, Any] = {
        "rss_url": "https://example.com/feed.xml",
        "summary_provider": "summllama",
    }
    cfg = config.Config(**kw)
    provider = SummLlamaProvider(cfg)
    provider._tokenizer = MagicMock()
    provider._tokenizer.apply_chat_template.side_effect = lambda m, **_: repr(m)
    provider._tokenizer.decode.return_value = "A board about risk, read locally."
    provider._tokenizer.return_value.input_ids.shape = (1, 7)
    provider._model = MagicMock()
    provider._initialized = True
    fake_torch = MagicMock()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(mod, "_provider", lambda: (cfg, provider))
    mod._run_once()

    out = capsys.readouterr().out
    assert '"event": "board_summary.written"' in out
    stored = board_summaries.read_summaries(tmp_path / "app", uid)[cid]
    assert stored["text"] == "A board about risk, read locally."
    assert stored["model"] == "DISLab/SummLlama3.2-3B"
    assert stored["prompt"].startswith("summllama/summarization/board_system_v1@")
    assert provider._model.generate.call_args.kwargs["max_new_tokens"] == 400
    assert board_summaries.spent_today(tmp_path / "app", time.time()) == 0.0


def test_a_paid_profile_is_priced_and_never_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Budget zero (operator 2026-10-13): the real pricing table prices a cloud model above zero,
    so the sweep stops before the first call."""
    from podcast_scraper import config
    from podcast_scraper.providers.openai.openai_provider import OpenAIProvider

    monkeypatch.setenv("BOARD_SUMMARY_CORPUS_ROOT", str(_CORPUS))
    mod = _load(tmp_path, monkeypatch)
    monkeypatch.setattr(mod, "CORPUS_ROOT", _CORPUS)
    _board_with_two_episodes(tmp_path)
    paid: dict[str, Any] = {
        "rss_url": "https://example.com/feed.xml",
        "summary_provider": "openai",
        "openai_api_key": "sk-test123",
        "openai_summary_model": "gpt-4o-mini",
    }
    cfg = config.Config(**paid)
    provider = OpenAIProvider(cfg)
    provider.client = Mock()
    provider.initialize()
    monkeypatch.setattr(mod, "_provider", lambda: (cfg, provider))
    mod._run_once()
    out = capsys.readouterr().out
    # Priced on the model the provider calls (not cfg.summary_model, the local-ML field).
    assert '"stopped": "budget"' in out and '"model": "gpt-4o-mini"' in out
    provider.client.chat.completions.create.assert_not_called()


def test_a_bad_cycle_never_kills_the_loop_and_still_beats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load(tmp_path, monkeypatch)

    def broken() -> None:
        raise RuntimeError("something unexpected")

    monkeypatch.setattr(mod, "_run_once", broken)
    mod._cycle()
    assert (tmp_path / "hb" / "tick").read_text().isdigit()


def test_the_sweep_runs_every_five_minutes_and_never_with_the_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("BOARD_SUMMARY_INTERVAL_SECONDS", "BOARD_SUMMARY_INTERVAL_OFFSET_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    mod = _load(tmp_path, monkeypatch)
    assert (mod.INTERVAL_S, mod.OFFSET_S) == (300, 60)
    slept: list[float] = []
    mod._sleep_to_next_interval(slept.append)
    assert 0 < slept[0] <= 360
    # The digest wakes 120 s past the hour; 60 + 300k never lands there.
    assert all((60 + 300 * k) % 3600 != 120 for k in range(12))


def test_a_cycle_with_nothing_changed_never_talks_to_the_model_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """As much a success criterion as summarising a changed board (operator 2026-10-13): with
    nothing to summarise, no request reaches the DGX — not the summary, and not the served-model
    check ``initialize`` makes. Every five minutes, that would be 288 useless requests a day."""
    from podcast_scraper.providers.vllm.vllm_provider import VLLMProvider

    checks: list[str] = []
    monkeypatch.setattr(VLLMProvider, "_verify_served_model", lambda self: checks.append("models"))
    monkeypatch.setenv("BOARD_SUMMARY_CORPUS_ROOT", str(_CORPUS))
    mod = _load(tmp_path, monkeypatch)
    monkeypatch.setattr(mod, "CORPUS_ROOT", _CORPUS)
    monkeypatch.setattr(mod, "KEY_FILE", tmp_path / "absent")
    client = Mock()
    client.chat.completions.create.return_value = _chat_reply("A board about risk.")
    real = mod._provider

    def provider() -> Any:
        cfg, prov = real()
        prov.client = client
        return cfg, prov

    monkeypatch.setattr(mod, "_provider", provider)

    mod._run_once()  # no boards at all
    assert checks == [] and client.chat.completions.create.call_count == 0

    _board_with_two_episodes(tmp_path)
    mod._run_once()  # one changed board: one check, one call
    assert checks == ["models"] and client.chat.completions.create.call_count == 1

    capsys.readouterr()
    mod._run_once()  # nothing changed since
    assert checks == ["models"] and client.chat.completions.create.call_count == 1
    assert '"unchanged": 1' in capsys.readouterr().out


def test_a_provider_that_cannot_get_ready_ends_the_sweep_as_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from podcast_scraper.providers.vllm.vllm_provider import VLLMProvider

    def wrong_model(self: Any) -> None:
        raise RuntimeError("the slot serves another model")

    monkeypatch.setattr(VLLMProvider, "_verify_served_model", wrong_model)
    monkeypatch.setenv("BOARD_SUMMARY_CORPUS_ROOT", str(_CORPUS))
    mod = _load(tmp_path, monkeypatch)
    monkeypatch.setattr(mod, "CORPUS_ROOT", _CORPUS)
    monkeypatch.setattr(mod, "KEY_FILE", tmp_path / "absent")
    _board_with_two_episodes(tmp_path)
    mod._run_once()
    assert '"stopped": "model_unreachable"' in capsys.readouterr().out
