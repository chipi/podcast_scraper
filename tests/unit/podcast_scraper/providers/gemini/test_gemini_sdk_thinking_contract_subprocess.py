#!/usr/bin/env python3
"""Optional subprocess check: real ``google.genai`` types accept ``thinking_budget``.

Keeps this module separate from ``test_gemini_provider.py``, which patches
``sys.modules`` for ``google.genai`` and would hide SDK shape regressions.
"""

import subprocess
import sys
import unittest

import pytest

_SUBPROCESS_SNIPPET = r"""
import sys
try:
    from google.genai import types
except ImportError:
    sys.exit(2)
try:
    tc = types.ThinkingConfig(thinking_budget=0)
    cfg = types.GenerateContentConfig(thinking_config=tc)
    assert cfg.thinking_config is not None
    assert getattr(cfg.thinking_config, "thinking_budget", None) == 0
except Exception:
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""


@pytest.mark.unit
class TestGeminiSdkThinkingContractSubprocess(unittest.TestCase):
    """Runs in a clean interpreter; skips when ``google-genai`` is not installed."""

    def test_thinking_config_types_accept_thinking_budget(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_SNIPPET],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if proc.returncode == 2:
            self.skipTest("google.genai not installed (optional [llm] extra)")
        self.assertEqual(
            proc.returncode,
            0,
            msg=(proc.stderr or "") + (proc.stdout or ""),
        )
