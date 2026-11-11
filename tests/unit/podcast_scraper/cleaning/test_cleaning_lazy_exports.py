"""Importing ``podcast_scraper.cleaning`` loads neither the LLM nor the hybrid cleaner.

They leave public with ad detection (E04 IP extraction plan, item 5); until then they load on first
use, so every ``from podcast_scraper.cleaning import HybridCleaner`` keeps working.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit

PROBE = """
import sys
import podcast_scraper.cleaning as cleaning
loaded = sorted(m for m in ("podcast_scraper.cleaning.hybrid", "podcast_scraper.cleaning.llm_based")
                if m in sys.modules)
print(",".join(loaded))
from podcast_scraper.cleaning import HybridCleaner, LLMBasedCleaner
print(HybridCleaner.__module__, LLMBasedCleaner.__module__)
"""


def test_the_package_loads_without_the_llm_and_hybrid_cleaners() -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("COV")}
    out = subprocess.run(
        [sys.executable, "-c", PROBE], capture_output=True, text=True, env=env, check=True
    ).stdout.splitlines()
    assert out[0] == ""  # neither module loaded by the package import
    assert out[1] == "podcast_scraper.cleaning.hybrid podcast_scraper.cleaning.llm_based"


def test_an_unknown_name_is_an_attribute_error() -> None:
    import podcast_scraper.cleaning as cleaning

    with pytest.raises(AttributeError):
        cleaning.NoSuchCleaner  # noqa: B018
