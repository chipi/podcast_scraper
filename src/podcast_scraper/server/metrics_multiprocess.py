"""``/metrics`` for ``serve --workers N``: every worker's samples, and the scrape-time collectors.

With ``PROMETHEUS_MULTIPROC_DIR`` set, each worker writes its counters, histograms and gauges to
files there, and a scrape adds them up (``prometheus_client.multiprocess``). The instrumentator's
own multiprocess endpoint returns only those files. That drops the collectors that compute their
samples at scrape time and register on the default registry: build and schema info, job age,
naming coverage, and any an extension adds (digest health). Those read the database or the corpus,
not process memory, so the worker serving the scrape computes them for all.

Which collectors are scrape-time is read from the default registry: everything except metric
objects (whose samples are in the files) and the per-process collectors (process, platform, GC),
which describe one worker. That lookup reads ``CollectorRegistry._collector_to_names``, a private
attribute; ``tests/unit/podcast_scraper/server/test_metrics_multiprocess.py`` fails if it goes.
"""

from __future__ import annotations

from typing import Any, List


def scrape_time_collectors(registry: Any = None) -> List[Any]:
    """The default registry's collectors that compute their samples when scraped."""
    from prometheus_client import (
        gc_collector,
        metrics,
        platform_collector,
        process_collector,
        REGISTRY,
    )

    per_process = (
        metrics.MetricWrapperBase,
        process_collector.ProcessCollector,
        platform_collector.PlatformCollector,
        gc_collector.GCCollector,
    )
    reg = registry if registry is not None else REGISTRY
    return [c for c in list(reg._collector_to_names) if not isinstance(c, per_process)]


def multiprocess_registry() -> Any:
    """A registry for one scrape: the workers' files plus the scrape-time collectors."""
    from prometheus_client import CollectorRegistry, multiprocess

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    for collector in scrape_time_collectors():
        registry.register(collector)
    return registry


def install(app: Any) -> None:
    """Add ``GET /metrics`` serving :func:`multiprocess_registry`."""
    from fastapi import Response
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    @app.get("/metrics", include_in_schema=False)
    def metrics_multiprocess() -> Response:
        return Response(generate_latest(multiprocess_registry()), media_type=CONTENT_TYPE_LATEST)
