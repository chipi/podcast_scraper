"""#1878 fix 2: a double schema failure on the primary walks the RFC-106 chain.

The gap this pins: a schema failure happens at PARSE time, after the primary's call returned
"successfully", so the FallbackAwareSummarizationProvider's exception contract never fires. In
prod (probe #2, 2026-08-30) that meant an episode shipped with bullets=0 while a healthy
fallback vendor sat unused — probe #1 got its summary only because its failure happened to be
provider-shaped.

Tests here cover the new ``call_via_fallback`` escalation surface on the wrapper itself. The
metadata_generation wiring is exercised by the existing ADR-148 suite plus integration runs;
the wrapper method is the load-bearing new contract.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from podcast_scraper.summarization.fallback import FallbackAwareSummarizationProvider

pytestmark = [pytest.mark.unit]


class _Recorder:
    """Minimal provider double; records calls, returns or raises per configuration."""

    def __init__(self, name: str, result: Any = None, raises: bool = False) -> None:
        self.name = name
        self.result = result
        self.raises = raises
        self.calls: List[Dict[str, Any]] = []

    def initialize(self) -> None:  # noqa: D102 - protocol stub
        pass

    def summarize(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.raises:
            raise RuntimeError(f"{self.name} exploded")
        return self.result


def _wrapper_with_fakes(monkeypatch, primary: _Recorder, tiers: List[_Recorder]):
    # Typed Any on purpose: the double bypasses __init__ (no Config, pre-built tiers), which is
    # exactly the point — these tests exercise chain-walking, not construction.
    wrapper: Any = FallbackAwareSummarizationProvider.__new__(FallbackAwareSummarizationProvider)
    wrapper._primary = primary
    wrapper._fallback_names = [t.name for t in tiers]
    wrapper._cfg = None
    wrapper._fallbacks = {t.name: t for t in tiers}  # pre-built: no factory involved
    wrapper._fallback_recorded = False
    return wrapper


def test_call_via_fallback_skips_the_primary_entirely(monkeypatch):
    primary = _Recorder("primary", result={"summary": "primary would win"})
    tier = _Recorder("deepseek", result={"summary": '{"bullets": ["from the fallback"]}'})
    w = _wrapper_with_fakes(monkeypatch, primary, [tier])

    out = w.call_via_fallback("summarize", text="t")

    assert out == tier.result
    assert primary.calls == [], (
        "call_via_fallback must NOT touch the primary — it has already failed schema twice; "
        "a third identical call is exactly what the escalation exists to avoid"
    )
    assert len(tier.calls) == 1


def test_call_via_fallback_walks_past_a_failing_tier(monkeypatch):
    t1 = _Recorder("deepseek", raises=True)
    t2 = _Recorder("gemini", result={"summary": "tier two delivers"})
    w = _wrapper_with_fakes(monkeypatch, _Recorder("primary"), [t1, t2])

    out = w.call_via_fallback("summarize", text="t")

    assert out == t2.result
    assert len(t1.calls) == 1 and len(t2.calls) == 1


def test_call_via_fallback_raises_when_chain_empty(monkeypatch):
    w = _wrapper_with_fakes(monkeypatch, _Recorder("primary"), [])
    with pytest.raises(RuntimeError, match="no fallback tier available"):
        w.call_via_fallback("summarize", text="t")


def test_call_via_fallback_raises_when_all_tiers_fail(monkeypatch):
    t1 = _Recorder("deepseek", raises=True)
    w = _wrapper_with_fakes(monkeypatch, _Recorder("primary"), [t1])
    with pytest.raises(RuntimeError, match="no fallback tier available"):
        w.call_via_fallback("summarize", text="t")
    assert len(t1.calls) == 1


def test_exception_path_still_walks_the_chain(monkeypatch):
    """The refactor shares one chain-walker; the original exception contract must be intact."""
    primary = _Recorder("primary", raises=True)
    tier = _Recorder("deepseek", result={"summary": "saved"})
    w = _wrapper_with_fakes(monkeypatch, primary, [tier])

    out = w.summarize(text="t")

    assert out == tier.result
    assert len(primary.calls) == 1 and len(tier.calls) == 1
