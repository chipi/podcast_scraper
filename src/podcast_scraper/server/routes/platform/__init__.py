"""Reserved package for future megasketch platform routes (#50, #347).

**Not used for RFC-077.** Corpus RSS list file, viewer-safe operator YAML, and HTTP
pipeline jobs are **top-level** routers next to this package:

- ``routes/feeds.py`` — ``GET``/``PUT /api/feeds`` (``rss_urls.list.txt``); gated by
  ``enable_feeds_api`` / env ``PODCAST_SERVE_ENABLE_FEEDS_API``.
- ``routes/operator_config.py`` — ``GET``/``PUT /api/operator-config``; gated by
  ``enable_operator_config_api`` / ``PODCAST_SERVE_ENABLE_OPERATOR_CONFIG_API``.
- ``routes/jobs.py`` — ``POST``/``GET /api/jobs``, cancel, reconcile, etc.; gated by
  ``enable_jobs_api`` / ``PODCAST_SERVE_ENABLE_JOBS_API``.

Normative docs: RFC-077 (viewer feeds, operator config, pipeline jobs) and
``docs/guides/SERVER_GUIDE.md``. The ``enable_platform`` argument to ``create_app``
remains reserved until #50/#347 ship separate catalog or DB-backed surfaces.
"""
