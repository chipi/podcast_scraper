"""Tests for cross-process resource sampling."""

from __future__ import annotations

import os
import time

from podcast_scraper.monitor.sampler import CrossProcessSampler


def test_sampler_collects_for_current_process() -> None:
    sampler = CrossProcessSampler(os.getpid(), interval_s=0.05)
    sampler.start()
    time.sleep(0.12)
    sampler.stop()
    assert len(sampler.samples) >= 1
    _mono, rss, cpu = sampler.samples[-1]
    assert rss > 0.0
    assert isinstance(cpu, float)
