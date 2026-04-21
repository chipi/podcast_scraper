"""Unit tests for single_feed_uses_corpus_layout (#644).

Since #646 the wrapping happens in a Config ``@model_validator(mode="after")``,
not in ``service.run``. That means every entry point (CLI, service.run, direct
construction) gets the same wrapped ``output_dir`` without the caller needing
to know. These tests cover both the Config level (guaranteed wrapping) and the
service level (backwards-compatible behaviour).
"""

from __future__ import annotations

from unittest.mock import patch

from podcast_scraper import service
from podcast_scraper.config import Config


def _make_cfg(output_dir: str, rss_url: str, corpus_layout: bool) -> Config:
    return Config.model_validate(
        {
            "rss_url": rss_url,
            "output_dir": output_dir,
            "single_feed_uses_corpus_layout": corpus_layout,
        }
    )


class TestSingleFeedCorpusLayout:
    def test_legacy_default_writes_to_output_dir_unchanged(self, tmp_path):
        cfg = _make_cfg(str(tmp_path), "https://feeds.example.com/podcast.xml", False)
        with patch.object(service, "workflow") as mock_workflow:
            mock_workflow.run_pipeline.return_value = (0, "ok")
            result = service.run(cfg)

        assert result.success
        # The cfg passed to run_pipeline should have the *original* output_dir.
        call_cfg = mock_workflow.run_pipeline.call_args[0][0]
        assert str(call_cfg.output_dir) == str(tmp_path)

    def test_corpus_layout_wraps_output_dir(self, tmp_path):
        cfg = _make_cfg(str(tmp_path), "https://feeds.example.com/podcast.xml", True)
        with patch.object(service, "workflow") as mock_workflow:
            mock_workflow.run_pipeline.return_value = (0, "ok")
            service.run(cfg)

        call_cfg = mock_workflow.run_pipeline.call_args[0][0]
        # Output dir should now be <tmp_path>/feeds/<slug>/
        assert str(call_cfg.output_dir).startswith(str(tmp_path.resolve()))
        # feeds/ path segment must be present.
        assert "/feeds/" in str(call_cfg.output_dir)
        assert "rss_feeds.example.com_" in str(call_cfg.output_dir)

    def test_corpus_layout_slug_is_deterministic(self, tmp_path):
        rss = "https://feeds.example.com/podcast.xml"
        cfg1 = _make_cfg(str(tmp_path), rss, True)
        cfg2 = _make_cfg(str(tmp_path), rss, True)

        with patch.object(service, "workflow") as mock_workflow:
            mock_workflow.run_pipeline.return_value = (0, "ok")
            service.run(cfg1)
            first = mock_workflow.run_pipeline.call_args[0][0].output_dir
            service.run(cfg2)
            second = mock_workflow.run_pipeline.call_args[0][0].output_dir
        assert first == second

    def test_config_validator_wraps_at_construction(self, tmp_path):
        """Every caller — CLI, service.run, programmatic — goes through Config
        construction, so wrapping must happen there (not in a single caller).
        Prevents the #646-discovered bug where CLI path bypassed service.run."""
        cfg = Config.model_validate(
            {
                "rss_url": "https://feeds.example.com/podcast.xml",
                "output_dir": str(tmp_path),
                "single_feed_uses_corpus_layout": True,
            }
        )
        assert "/feeds/rss_feeds.example.com_" in cfg.output_dir
        # Original tmp_path is preserved as the prefix.
        assert cfg.output_dir.startswith(str(tmp_path.resolve())) or cfg.output_dir.startswith(
            str(tmp_path)
        )

    def test_config_validator_idempotent(self, tmp_path):
        """Validator must not double-wrap if output_dir already contains a
        /feeds/ segment (matters when someone runs Config.model_validate on
        a previously-wrapped dict, e.g. round-tripping through yaml)."""
        cfg1 = Config.model_validate(
            {
                "rss_url": "https://feeds.example.com/podcast.xml",
                "output_dir": str(tmp_path),
                "single_feed_uses_corpus_layout": True,
            }
        )
        # Round-trip through model_dump: should NOT accumulate another /feeds/ segment.
        dumped = cfg1.model_dump()
        cfg2 = Config.model_validate(dumped)
        assert cfg2.output_dir == cfg1.output_dir
        assert cfg2.output_dir.count("/feeds/") == 1

    def test_config_validator_skips_when_flag_off(self, tmp_path):
        cfg = Config.model_validate(
            {
                "rss_url": "https://feeds.example.com/podcast.xml",
                "output_dir": str(tmp_path),
                "single_feed_uses_corpus_layout": False,
            }
        )
        assert cfg.output_dir == str(tmp_path)
        assert "/feeds/" not in cfg.output_dir

    def test_corpus_layout_does_not_affect_multi_feed(self, tmp_path):
        # Multi-feed path already uses corpus layout. The flag must not alter
        # its behavior (multi-feed routes through _run_multi_feed which is
        # separate code).
        cfg = Config.model_validate(
            {
                "rss_url": "https://feeds.example.com/a.xml",
                "rss_urls": [
                    "https://feeds.example.com/a.xml",
                    "https://feeds.example.com/b.xml",
                ],
                "output_dir": str(tmp_path),
                "single_feed_uses_corpus_layout": True,
            }
        )
        # Multi-feed path is taken when len(rss_urls) >= 2. Route stays the
        # same; our flag is a no-op for multi-feed.
        with (
            patch.object(service, "_run_multi_feed") as mock_multi,
            patch.object(service, "workflow") as mock_workflow,
        ):
            from podcast_scraper.service import ServiceResult

            mock_multi.return_value = ServiceResult(
                episodes_processed=2, summary="", success=True, error=None
            )
            service.run(cfg)
        mock_multi.assert_called_once()
        mock_workflow.run_pipeline.assert_not_called()
