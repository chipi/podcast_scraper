"""The run's Config must be FINAL before any object captures it (#2172 / S2.14).

THE BUG THIS CLOSES, found by review on 2026-10-03. `run_pipeline` builds its providers at what
used to be Step 1.6 and only later — after the RSS fetch, which is the first point the channel tag
is known — replaces the Config with `cfg.model_copy(update={"feed_declared_language": ...})`. A
`model_copy` produces a NEW instance, so every object constructed before it kept the OLD one.

`MLProvider.__init__` stores `self.cfg`, and `providers/ml/ml_provider.py` resolves
`transcription_language(self.cfg)` from that stored copy to supply `text_language=` — the S2.14
guard — when it names speakers from the episode title and description. With the stale Config the
guard resolved from the profile default for the entire run: a feed declaring `es-ES` got English
NER over its Spanish title, and §5.2 measured what that produces — recall 2/2 with precision
falling 67% -> 18%. Phantom people, each minted as a person node with a `SPOKEN_BY` edge. A
missing name is visible; a phantom one is not, which is the whole reason S2.14 exists.

`preload_ml_models_if_needed` has the same shape one step earlier: it caches a provider in a
module global that keeps its own `cfg`.

WHY THE FIX IS THE ORDER AND NOT A SECOND WRITE. Re-assigning `provider.cfg` after the copy would
work for the two providers that exist today and fail silently for the third one someone adds —
the same class of defect as the catalog builders that each had to remember a check. One Config
instance per run, settled before anything can hold it, is the version that stays true.

WHY THIS TEST READS THE AST rather than running a pipeline. What regressed is a STATEMENT ORDER
inside one function. A behavioural test would need a real feed, a real provider and a real ML
stack to observe the consequence, and it would then be an integration test that could not run in
the unit tier at all (rule U1). The order is the invariant; this asserts the order.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import List, Optional

import pytest

from podcast_scraper.workflow import orchestration as orch

pytestmark = [pytest.mark.unit]

#: Everything that captures the Config into state outliving the call. Each must sit AFTER the
#: `feed_declared_language` write. Add to this list when a new such call appears in `run_pipeline`.
CAPTURING_CALLS = {
    "_create_all_providers": (
        "MLProvider stores `self.cfg` and resolves the S2.14 guard language from it"
    ),
    "preload_ml_models_if_needed": (
        "the preloaded provider is cached in a module global that keeps its own cfg"
    ),
}


def _run_pipeline_tree() -> ast.FunctionDef:
    src = Path(inspect.getsourcefile(orch) or "").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == "run_pipeline":
            return node
    raise AssertionError("run_pipeline not found in orchestration.py")


def _declared_language_write_line(fn: ast.FunctionDef) -> Optional[int]:
    """The line where `feed_declared_language` is written into a `model_copy` update."""
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "model_copy":
            continue
        for kw in node.keywords:
            if kw.arg == "update" and isinstance(kw.value, ast.Dict):
                keys = [k.value for k in kw.value.keys if isinstance(k, ast.Constant)]
                if "feed_declared_language" in keys:
                    return node.lineno
    return None


def _call_lines(fn: ast.FunctionDef, name: str) -> List[int]:
    return sorted(
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
        and (node.func.id if isinstance(node.func, ast.Name) else node.func.attr) == name
    )


class TestNothingCapturesTheConfigBeforeItIsFinal:
    def test_the_feed_language_is_still_written_onto_the_config(self) -> None:
        """Guards the guard: if the write is gone or renamed, every ordering assertion below
        becomes vacuous rather than failing."""
        assert _declared_language_write_line(_run_pipeline_tree()) is not None, (
            "run_pipeline no longer writes `feed_declared_language` into a model_copy update — "
            "either the field moved, in which case this test must follow it, or THE ONE READER "
            "(S0.6) is back to answering from the profile default"
        )

    @pytest.mark.parametrize("call", sorted(CAPTURING_CALLS))
    def test_the_capturing_call_happens_after_the_write(self, call: str) -> None:
        fn = _run_pipeline_tree()
        write = _declared_language_write_line(fn)
        assert write is not None  # covered by the test above
        lines = _call_lines(fn, call)
        assert lines, (
            f"run_pipeline no longer calls {call}() — if it was removed, drop it from "
            "CAPTURING_CALLS; if it was renamed, rename it here"
        )
        early = [ln for ln in lines if ln < write]
        assert not early, (
            f"{call}() is called at line(s) {early}, BEFORE the feed's declared language is "
            f"written onto the config at line {write}. {CAPTURING_CALLS[call]}, so it would hold "
            "the pre-copy Config for the whole run and the S2.14 guard would resolve the "
            "language from the profile default instead of the feed's channel tag."
        )


class TestTheGuardStillReadsTheCapturedConfig:
    """The premise above, asserted where it lives.

    If `ml_provider` stopped resolving the language from `self.cfg` — threading it in per call,
    say — the ordering requirement would be gone and this file would be pinning an order that no
    longer protects anything. Better to fail here and be deleted deliberately.
    """

    def test_ml_provider_resolves_the_s2_14_language_from_self_cfg(self) -> None:
        from podcast_scraper.providers.ml import ml_provider

        src = Path(inspect.getsourcefile(ml_provider) or "").read_text(encoding="utf-8")
        assert "text_language=transcription_language(self.cfg)" in src, (
            "ml_provider no longer feeds the S2.14 guard from its STORED cfg. If the language is "
            "now threaded in per call, the capture hazard is gone and this module should be "
            "deleted rather than relaxed."
        )
