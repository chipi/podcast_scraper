"""``/metrics`` with ``serve --workers N`` keeps the scrape-time collectors (build and schema info,
job age, extensions' gauges) beside the workers' samples (``server/metrics_multiprocess.py``)."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit]

prometheus_client = pytest.importorskip("prometheus_client")


class _ScrapeTime:
    def collect(self):  # noqa: ANN201 - prometheus_client's collector protocol
        from prometheus_client.core import GaugeMetricFamily

        yield GaugeMetricFamily("s5_scrape_time_probe", "probe", value=1)


def test_the_private_registry_attribute_it_reads_still_exists() -> None:
    assert isinstance(prometheus_client.REGISTRY._collector_to_names, dict)


def test_scrape_time_collectors_are_kept_and_per_process_ones_are_not() -> None:
    from prometheus_client import CollectorRegistry, Counter, ProcessCollector

    from podcast_scraper.server.metrics_multiprocess import scrape_time_collectors

    reg = CollectorRegistry()
    probe = _ScrapeTime()
    reg.register(probe)
    Counter("s5_counter_total", "c", registry=reg)
    ProcessCollector(registry=reg)
    assert scrape_time_collectors(reg) == [probe]


def test_the_multiprocess_registry_serves_files_and_scrape_time_collectors(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from prometheus_client import generate_latest, REGISTRY

    from podcast_scraper.server import metrics_multiprocess

    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    probe = _ScrapeTime()
    REGISTRY.register(probe)
    try:
        text = generate_latest(metrics_multiprocess.multiprocess_registry()).decode()
    finally:
        REGISTRY.unregister(probe)
    assert "s5_scrape_time_probe 1.0" in text
    assert "process_resident_memory_bytes" not in text
