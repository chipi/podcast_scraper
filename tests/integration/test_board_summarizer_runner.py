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
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

pytest.importorskip("fastapi")  # _run_once imports the server package

pytestmark = pytest.mark.integration

_RUNNER = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "podcast_scraper"
    / "server"
    / "board_summarizer_runner.py"
)
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
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_API_BASE", raising=False)
    monkeypatch.setenv("LITELLM_API_KEY", "sk-gateway-test")
    cfg, provider = mod._provider()
    assert type(provider) is VLLMProvider
    assert cfg.summary_provider == "vllm" and list(cfg.summary_fallback_providers or []) == []
    # As the prod pipeline (#2342): the DGX through the LiteLLM gateway, with the pipeline's
    # gateway key; the wire model is still the DGX's real id.
    assert cfg.vllm_api_base == "http://homelab:4001/v1"
    assert provider._resolve_api_key(cfg) == "sk-gateway-test"
    assert provider.summary_model == "NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4"


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
    assert stored["model"] == client.chat.completions.create.call_args_list[0].kwargs["model"]
    assert stored["prompt"].startswith("openai/summarization/board_system_v1@")
    assert "+openai/summarization/board_user_v1@" in stored["prompt"]
    # The cycle also summarises the fixture's storylines (#2344); the board's call is the one
    # with the board prompt.
    calls = [c.kwargs for c in client.chat.completions.create.call_args_list]
    board_calls = [k for k in calls if "a listener's board" in k["messages"][0]["content"]]
    assert len(board_calls) == 1 and board_calls[0]["max_tokens"] == 400
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
    client = Mock()
    client.chat.completions.create.return_value = _chat_reply("A board about risk.")
    real = mod._provider

    def provider() -> Any:
        cfg, prov = real()
        prov.client = client
        return cfg, prov

    monkeypatch.setattr(mod, "_provider", provider)

    # The first cycle writes the fixture's three storyline summaries (#2344): one served-model
    # check, three calls.
    mod._run_once()
    assert checks == ["models"] and client.chat.completions.create.call_count == 3

    mod._run_once()  # no boards, storylines unchanged: nothing reaches the server
    assert checks == ["models"] and client.chat.completions.create.call_count == 3

    _board_with_two_episodes(tmp_path)
    mod._run_once()  # one changed board: one check, one call
    assert checks == ["models", "models"] and client.chat.completions.create.call_count == 4

    capsys.readouterr()
    mod._run_once()  # nothing changed since
    assert checks == ["models", "models"] and client.chat.completions.create.call_count == 4
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
    _board_with_two_episodes(tmp_path)
    mod._run_once()
    assert '"stopped": "model_unreachable"' in capsys.readouterr().out


def test_it_starts_up_as_the_pipeline_cli_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cli.main: Sentry as ``pipeline``, then OTEL, then console logging at INFO."""
    from podcast_scraper.utils import otel_init, sentry_init
    from podcast_scraper.workflow import orchestration

    mod = _load(tmp_path, monkeypatch)
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(sentry_init, "init_sentry", lambda c: calls.append(("sentry", c)))
    monkeypatch.setattr(otel_init, "init_otel", lambda: calls.append(("otel",)))
    monkeypatch.setattr(orchestration, "apply_log_level", lambda lvl: calls.append(("log", lvl)))
    mod._startup()
    assert calls == [("sentry", "pipeline"), ("otel",), ("log", "INFO")]


def test_a_summary_emits_the_llm_cost_event_with_its_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The episode summary's own cost event (``record_provider_call_cost``), tagged
    ``stage=board_summary`` so dashboards and the obs MCP tell the two apart."""
    _offline(monkeypatch)
    monkeypatch.setenv("BOARD_SUMMARY_CORPUS_ROOT", str(_CORPUS))
    mod = _load(tmp_path, monkeypatch)
    monkeypatch.setattr(mod, "CORPUS_ROOT", _CORPUS)
    _board_with_two_episodes(tmp_path)
    client = Mock()
    client.chat.completions.create.return_value = _chat_reply("A board about risk.")
    real = mod._provider

    def provider() -> Any:
        cfg, prov = real()
        prov.client = client
        return cfg, prov

    monkeypatch.setattr(mod, "_provider", provider)
    with caplog.at_level("INFO", logger="podcast_scraper.workflow.cost_monitoring"):
        mod._run_once()
    events = [json.loads(r.getMessage()) for r in caplog.records if '"llm_cost"' in r.getMessage()]
    stages = [e["stage"] for e in events]
    # One board, and the fixture's three storylines (#2344), each tagged with its own stage.
    assert stages.count("board_summary") == 1 and stages.count("storyline_summary") == 3
    assert {e["provider"] for e in events} == {"vllm"}


def test_after_startup_info_lines_reach_the_console(tmp_path: Path) -> None:
    """In the image nothing else gives root a console handler, so before ``_startup`` the INFO
    ``llm_cost`` line was dropped (Python's last-resort handler prints WARNING and up only) and
    Alloy had nothing to ship. A fresh interpreter, as in the container."""
    import subprocess
    import sys

    program = f"""
import importlib.util, logging
spec = importlib.util.spec_from_file_location("bsr", {str(_RUNNER)!r})
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
logging.getLogger("podcast_scraper.workflow.cost_monitoring").info("BEFORE-STARTUP")
mod._startup()
logging.getLogger("podcast_scraper.workflow.cost_monitoring").info("AFTER-STARTUP")
"""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OTEL_", "PODCAST_SENTRY"))}
    env.update(
        BOARD_SUMMARY_HEARTBEAT_FILE=str(tmp_path / "hb"),
        APP_DATA_DIR=str(tmp_path / "app"),
    )
    out = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=120, env=env
    )
    assert out.returncode == 0, out.stderr[-2000:]
    logs = out.stdout + out.stderr
    assert "AFTER-STARTUP" in logs and "BEFORE-STARTUP" not in logs


def test_a_failed_cycle_is_logged_at_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """ERROR through logging is what Sentry's logging integration reports, as for the pipeline."""
    mod = _load(tmp_path, monkeypatch)

    def broken() -> None:
        raise RuntimeError("something unexpected")

    monkeypatch.setattr(mod, "_run_once", broken)
    with caplog.at_level("ERROR"):
        mod._cycle()
    assert any(r.levelname == "ERROR" and r.exc_info for r in caplog.records)


def test_each_cycle_runs_the_reference_pass_first_and_its_failure_never_stops_the_summaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#2345: the reference pass runs before the summaries, whether or not they are on."""
    mod = _load(tmp_path, monkeypatch)
    order: list[str] = []

    def failing_refs() -> None:
        order.append("references")
        raise RuntimeError("corpus unreadable")

    monkeypatch.setattr(mod, "_references_once", failing_refs)
    monkeypatch.setattr(mod, "_run_once", lambda: order.append("summaries"))
    monkeypatch.setattr(mod, "_beat", lambda: order.append("beat"))
    mod._cycle()
    assert order == ["references", "summaries", "beat"]


def test_the_reference_pass_rekeys_a_renamed_storyline_from_the_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end through the sidecar: id_history in the storyline artifact → a follow re-keyed."""
    from podcast_scraper.server import app_user_state

    mod = _load(tmp_path, monkeypatch)
    root = tmp_path / "corpus"
    (root / "enrichments").mkdir(parents=True)
    (root / "enrichments" / "topic_theme_clusters.json").write_text(
        json.dumps(
            {
                "data": {
                    "clusters": [{"graph_compound_parent_id": "thc:new", "members": []}],
                    "id_history": {"successors": {"thc:old": "thc:new"}, "gone": []},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "CORPUS_ROOT", root)
    app_user_state.add_interest(mod.DATA_DIR, "u1", "thc:old")
    mod._references_once()
    assert app_user_state.get_interests(mod.DATA_DIR, "u1") == ["thc:new"]


def test_the_storyline_backfill_dry_runs_then_fills_every_storyline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """#2344: the operator's one-shot fill after a deploy, dry run first."""
    from podcast_scraper.server import storyline_summaries

    _offline(monkeypatch)
    mod = _load(tmp_path, monkeypatch)
    monkeypatch.setattr(mod, "CORPUS_ROOT", _CORPUS)
    n = len(storyline_summaries.load_storylines(_CORPUS))
    assert mod.backfill_storylines(apply=False) == 0
    assert f"would summarise {n} storylines" in capsys.readouterr().out
    assert storyline_summaries.read_summaries(tmp_path / "app") == {}

    client = Mock()
    client.chat.completions.create.return_value = _chat_reply("A thread on risk.")
    real = mod._provider

    def provider() -> Any:
        cfg, prov = real()
        prov.client = client
        return cfg, prov

    monkeypatch.setattr(mod, "_provider", provider)
    assert mod.backfill_storylines(apply=True) == 0
    assert len(storyline_summaries.read_summaries(tmp_path / "app")) == n
    calls = client.chat.completions.create.call_args_list
    assert len(calls) == n and "a storyline" in calls[0].kwargs["messages"][0]["content"]


def test_each_cycle_is_a_run_with_its_own_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 2026-10-15: the sidecar's storyline summaries reached Langfuse with no run, episode or trace
    # (112 of them), so nothing grouped them and nothing joined them to a trace. Each cycle is now
    # a run (`board-summarizer-<UTC>`): the gateway calls share its session, and its events and
    # logs carry the run id.
    from podcast_scraper.utils import correlation

    mod = _load(tmp_path, monkeypatch)
    seen: list = []
    monkeypatch.setattr(mod, "_references_once", lambda: None)
    monkeypatch.setattr(mod, "_run_once", lambda: seen.append(correlation.get_run_id()))
    monkeypatch.setattr(mod, "_beat", lambda: None)
    mod._cycle()
    mod._cycle()
    assert len(seen) == 2
    assert all(r and r.startswith("board-summarizer-") for r in seen)
    assert correlation.get_run_id() is None  # nothing leaks past the cycle
