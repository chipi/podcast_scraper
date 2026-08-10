"""Unit tests for ``jobs.build_pipeline_argv``."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.rss.feeds_spec import FEEDS_SPEC_DEFAULT_BASENAME
from podcast_scraper.server.jobs import build_pipeline_argv

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]


def test_argv_includes_feeds_spec_when_present(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    op = corpus / "viewer_operator.yaml"
    op.write_text("profile: local\n", encoding="utf-8")
    spec_path = corpus / FEEDS_SPEC_DEFAULT_BASENAME
    spec_path.write_text(
        "feeds:\n  - https://example.com/podcast.xml\n",
        encoding="utf-8",
    )
    argv = build_pipeline_argv(corpus, op)
    idx = argv.index("--feeds-spec")
    assert argv[idx + 1] == str(spec_path.resolve())


def test_argv_passes_run_id_when_given(tmp_path: Path) -> None:
    """P1.6: run_id → --run-id so the pipeline run id == the Jobs API job_id (one join key)."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    op = corpus / "viewer_operator.yaml"
    op.write_text("profile: local\n", encoding="utf-8")
    argv = build_pipeline_argv(corpus, op, run_id="job-abc-123")
    i = argv.index("--run-id")
    assert argv[i + 1] == "job-abc-123"


def test_argv_omits_run_id_when_absent(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    op = corpus / "viewer_operator.yaml"
    op.write_text("profile: local\n", encoding="utf-8")
    assert "--run-id" not in build_pipeline_argv(corpus, op)


def test_argv_multi_feed_still_passes_feeds_spec(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    op = corpus / "viewer_operator.yaml"
    op.write_text("profile: local\n", encoding="utf-8")
    (corpus / FEEDS_SPEC_DEFAULT_BASENAME).write_text(
        "feeds:\n  - https://a.example/1.xml\n  - https://b.example/2.xml\n",
        encoding="utf-8",
    )
    argv = build_pipeline_argv(corpus, op)
    assert "--feeds-spec" in argv


def test_argv_feed_scope_runs_single_feed_not_batch(tmp_path: Path) -> None:
    """P1.4: a feed_url scopes the run to one feed (corpus layout) + knobs, NOT the whole batch."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    op = corpus / "viewer_operator.yaml"
    op.write_text("profile: cloud_balanced\n", encoding="utf-8")
    (corpus / FEEDS_SPEC_DEFAULT_BASENAME).write_text(
        "feeds:\n  - https://a.example/1.xml\n", encoding="utf-8"
    )
    argv = build_pipeline_argv(
        corpus,
        op,
        feed_url="https://a.example/1.xml",
        skip_existing=True,
        max_episodes=5,
        episode_order="newest",
    )
    assert "--feeds-spec" not in argv  # NOT the batch
    # The URL is the POSITIONAL ``rss`` arg (what populates ``config.rss_url``, which the
    # single-feed scraping stage reads), NOT ``--rss`` — that binds to ``rss_extra`` (the
    # plural multi-feed list) and leaves ``config.rss_url`` None, so the run dies at the
    # scraping stage with "RSS URL is required". A single ``rss_extra`` entry does not
    # trigger the multi-feed loop that would set rss_url per feed. Regression guard.
    assert "--rss" not in argv
    assert "https://a.example/1.xml" in argv
    assert "--single-feed-uses-corpus-layout" in argv
    assert "--skip-existing" in argv
    assert argv[argv.index("--max-episodes") + 1] == "5"
    assert argv[argv.index("--episode-order") + 1] == "newest"


def test_argv_feed_url_maps_to_positional_rss_not_rss_extra(tmp_path: Path) -> None:
    """The feed URL must parse into ``args.rss`` (the positional) — the value
    ``config.rss_url`` is derived from — not ``args.rss_extra``. Passing it via
    ``--rss`` was the prod bug: the single-feed run got ``config.rss_url = None``.
    """
    import argparse

    from podcast_scraper.cli import _add_common_arguments

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    op = corpus / "viewer_operator.yaml"
    op.write_text("profile: cloud_balanced\n", encoding="utf-8")
    url = "https://a.example/1.xml"
    argv = build_pipeline_argv(corpus, op, feed_url=url, skip_existing=True)

    parser = argparse.ArgumentParser()
    _add_common_arguments(parser)
    # Parse only the tokens the common parser owns: the positional URL + the single-feed flag.
    ns, _unknown = parser.parse_known_args([url, "--single-feed-uses-corpus-layout"])
    assert (
        ns.rss == url
    ), f"URL must land in positional args.rss (config.rss_url source); got {ns.rss!r}"
    assert not getattr(ns, "rss_extra", None), "URL must NOT go to rss_extra (the plural list)"
    # And the built argv must not resurrect the broken form.
    assert "--rss" not in argv and url in argv


def test_argv_feed_scope_omits_unset_knobs(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    op = corpus / "viewer_operator.yaml"
    op.write_text("profile: cloud_balanced\n", encoding="utf-8")
    argv = build_pipeline_argv(corpus, op, feed_url="https://a.example/1.xml")
    for flag in ("--skip-existing", "--append", "--max-episodes", "--episode-offset"):
        assert flag not in argv


def test_argv_omits_feeds_spec_when_missing(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    op = corpus / "viewer_operator.yaml"
    op.write_text("profile: local\n", encoding="utf-8")
    argv = build_pipeline_argv(corpus, op)
    assert "--feeds-spec" not in argv


def test_argv_includes_profile_before_config_when_present(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    op = corpus / "viewer_operator.yaml"
    op.write_text("profile: cloud_balanced\nmax_episodes: 1\n", encoding="utf-8")
    argv = build_pipeline_argv(corpus, op)
    i_prof = argv.index("--profile")
    i_cfg = argv.index("--config")
    assert argv[i_prof + 1] == "cloud_balanced"
    assert argv[i_cfg + 1] == str(op)
    assert i_prof < i_cfg


def test_argv_omits_profile_when_operator_has_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No on-disk profile + no env default → no --profile in argv (legacy)."""
    monkeypatch.delenv("PODCAST_DEFAULT_PROFILE", raising=False)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    op = corpus / "viewer_operator.yaml"
    op.write_text("max_episodes: 2\n", encoding="utf-8")
    argv = build_pipeline_argv(corpus, op)
    assert "--profile" not in argv
    assert "--config" in argv


# --- env-default fallback (#692, RFC-081 §Layer 1) ---------------------------


def test_argv_uses_env_default_when_operator_has_no_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No on-disk profile + ``PODCAST_DEFAULT_PROFILE=cloud_thin`` → argv adds it.

    Catches the case where the operator triggers a pipeline run on a fresh
    corpus before clicking Save in the profile menu — the run still goes
    through the right preset instead of whatever Config._resolve_profile
    would default to.
    """
    monkeypatch.delenv("PODCAST_AVAILABLE_PROFILES", raising=False)
    monkeypatch.setenv("PODCAST_DEFAULT_PROFILE", "cloud_thin")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    op = corpus / "viewer_operator.yaml"
    op.write_text("max_episodes: 2\n", encoding="utf-8")
    argv = build_pipeline_argv(corpus, op)
    i_prof = argv.index("--profile")
    assert argv[i_prof + 1] == "cloud_thin"


def test_argv_on_disk_profile_takes_priority_over_env_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When operator has saved a profile, env default does NOT override.

    On-disk profile is the operator's explicit choice; env default is the
    fallback for "operator hasn't picked yet". Don't second-guess saved choices.
    """
    monkeypatch.delenv("PODCAST_AVAILABLE_PROFILES", raising=False)
    monkeypatch.setenv("PODCAST_DEFAULT_PROFILE", "cloud_thin")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    op = corpus / "viewer_operator.yaml"
    op.write_text("profile: cloud_balanced\n", encoding="utf-8")
    argv = build_pipeline_argv(corpus, op)
    i_prof = argv.index("--profile")
    assert argv[i_prof + 1] == "cloud_balanced"


def test_argv_env_default_typo_does_not_fall_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Typo'd env default (not on disk) → no --profile, not --profile <typo>.

    ``env_default_profile`` validates against on-disk names; if the env value
    is a typo, the validator returns None and the argv stays clean. Better
    than passing a bogus profile to the CLI which would crash with a
    less-clear error.
    """
    monkeypatch.delenv("PODCAST_AVAILABLE_PROFILES", raising=False)
    monkeypatch.setenv("PODCAST_DEFAULT_PROFILE", "cloud_thinn")  # typo
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    op = corpus / "viewer_operator.yaml"
    op.write_text("max_episodes: 2\n", encoding="utf-8")
    argv = build_pipeline_argv(corpus, op)
    assert "--profile" not in argv


# ---------------------------------------------------------------------------
# Per-feed incremental knobs — argv contract (#1542 regression guard)
# ---------------------------------------------------------------------------


def test_argv_feed_scope_all_knobs_present(tmp_path: Path) -> None:
    """append, episode_offset, episode_order, skip_existing all appear in argv when set."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    op = corpus / "viewer_operator.yaml"
    op.write_text("profile: cloud_balanced\n", encoding="utf-8")
    argv = build_pipeline_argv(
        corpus,
        op,
        feed_url="https://a.example/1.xml",
        skip_existing=True,
        append=True,
        max_episodes=3,
        episode_offset=2,
        episode_order="oldest",
    )
    assert "--skip-existing" in argv
    assert "--append" in argv
    assert argv[argv.index("--max-episodes") + 1] == "3"
    assert argv[argv.index("--episode-offset") + 1] == "2"
    assert argv[argv.index("--episode-order") + 1] == "oldest"


def test_argv_feed_scope_skip_existing_only(tmp_path: Path) -> None:
    """skip_existing=True → --skip-existing; others omitted when not set."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    op = corpus / "viewer_operator.yaml"
    op.write_text("profile: cloud_balanced\n", encoding="utf-8")
    argv = build_pipeline_argv(corpus, op, feed_url="https://a.example/1.xml", skip_existing=True)
    assert "--skip-existing" in argv
    assert "--append" not in argv
    assert "--episode-offset" not in argv


def test_argv_feed_scope_append_only(tmp_path: Path) -> None:
    """append=True → --append; others omitted when not set."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    op = corpus / "viewer_operator.yaml"
    op.write_text("profile: cloud_balanced\n", encoding="utf-8")
    argv = build_pipeline_argv(corpus, op, feed_url="https://a.example/1.xml", append=True)
    assert "--append" in argv
    assert "--skip-existing" not in argv


def test_argv_feed_scope_episode_offset_zero_omitted(tmp_path: Path) -> None:
    """episode_offset=0 is the default — build_pipeline_argv should NOT emit --episode-offset 0."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    op = corpus / "viewer_operator.yaml"
    op.write_text("profile: cloud_balanced\n", encoding="utf-8")
    argv = build_pipeline_argv(corpus, op, feed_url="https://a.example/1.xml", episode_offset=None)
    assert "--episode-offset" not in argv


# ---------------------------------------------------------------------------
# Airgapped single-feed pipeline smoke (#1542) — no network, no real ML
# ---------------------------------------------------------------------------


def _cli_argv_tokens(full_argv: list[str]) -> list[str]:
    """Strip the interpreter + module prefix from a build_pipeline_argv result.

    ``build_pipeline_argv`` returns:
        [sys.executable, '-m', 'podcast_scraper.cli', ...]
    ``parse_args`` expects only the tokens after the module invocation.
    """
    # Locate '-m' and skip past 'podcast_scraper.cli'.
    try:
        m_idx = full_argv.index("-m")
        return full_argv[m_idx + 2 :]
    except ValueError:
        return full_argv[3:]  # fallback: drop first 3 tokens


def test_per_feed_job_config_resolves_rss_url(tmp_path: Path) -> None:
    """Regression guard for #1542: the per-feed job path must produce a Config
    where ``rss_url`` is set to the requested feed URL so the scraping stage
    never sees None and does not raise "RSS URL is required".

    No network, no ML — pure CLI arg parsing + Config construction."""
    from podcast_scraper.cli import _build_config, parse_args  # type: ignore[attr-defined]

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    op = corpus / "viewer_operator.yaml"
    # Use a profile-free operator so parse_args stays offline (no YAML load
    # that touches packaged profile presets which pull in optional deps).
    op.write_text("max_episodes: 1\n", encoding="utf-8")

    feed_url = "https://example.com/podcast.xml"
    full_argv = build_pipeline_argv(corpus, op, feed_url=feed_url)
    tokens = _cli_argv_tokens(full_argv)

    # parse_args may call validate_args which raises "RSS URL is required"
    # if the positional rss arg is missing — this is what we're guarding.
    args = parse_args(tokens)
    cfg = _build_config(args)

    assert cfg.rss_url == feed_url, (
        f"config.rss_url must equal the requested feed URL to prevent "
        f"'RSS URL is required' in the scraping stage; got {cfg.rss_url!r}"
    )


def test_per_feed_job_config_resolves_rss_url_with_knobs(tmp_path: Path) -> None:
    """Same as above but with incremental-add knobs set — prove they survive
    the CLI round-trip and land in Config correctly."""
    from podcast_scraper.cli import _build_config, parse_args  # type: ignore[attr-defined]

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    op = corpus / "viewer_operator.yaml"
    op.write_text("max_episodes: 1\n", encoding="utf-8")

    feed_url = "https://example.com/podcast.xml"
    full_argv = build_pipeline_argv(
        corpus,
        op,
        feed_url=feed_url,
        skip_existing=True,
        append=True,
        max_episodes=5,
        episode_offset=2,
        episode_order="oldest",
    )
    tokens = _cli_argv_tokens(full_argv)
    args = parse_args(tokens)
    cfg = _build_config(args)

    assert cfg.rss_url == feed_url
    assert cfg.skip_existing is True
    assert cfg.append is True
    assert cfg.max_episodes == 5
    assert cfg.episode_offset == 2
    assert cfg.episode_order == "oldest"
