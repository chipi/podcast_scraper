#!/usr/bin/env python3
"""Subprocess check: real ``google.genai`` types accept ``thinking_budget``.

Runs in :mod:`tests.integration` with ``.[llm]`` installed (CI integration jobs).
Kept separate from ``test_gemini_provider.py``, which patches ``sys.modules`` for
``google.genai`` and would hide SDK shape regressions.
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

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]


class TestGeminiSdkThinkingContractSubprocess(unittest.TestCase):
    """Runs in a clean interpreter with ``google-genai`` from the integration venv."""

    def test_thinking_config_types_accept_thinking_budget(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_SNIPPET],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertNotEqual(
            proc.returncode,
            2,
            msg="google.genai must be installed (integration uses .[dev,ml,llm])",
        )
        self.assertEqual(
            proc.returncode,
            0,
            msg=(proc.stderr or "") + (proc.stdout or ""),
        )
