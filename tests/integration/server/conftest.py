"""Auto-mark the consumer-app / server-route integration tests as ``app``.

Every ``test_app_*.py`` file under ``tests/integration/server/`` is tagged with
the ``app`` marker at collection time, so the fast PR suite can run them
(``-m "integration and (critical_path or app)"``) and codecov credits the route
coverage on the PR — not only post-merge on ``main``. ``critical_path`` stays
pipeline-only (RSS → … → Files); ``app`` is the app-route peer. Convention-driven
so new ``test_app_*`` files are covered automatically, with no per-test marking.
"""

from __future__ import annotations


def pytest_collection_modifyitems(config, items):  # noqa: ANN001, ARG001
    for item in items:
        name = getattr(item.path, "name", "")
        if name.startswith("test_app_"):
            item.add_marker("app")
