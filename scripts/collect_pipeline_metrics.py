#!/usr/bin/env python3
"""
Collect pipeline performance metrics by running a minimal pipeline.

This script runs a sample pipeline with test fixtures to generate pipeline metrics
that can be included in the metrics dashboard.

Usage:
    python scripts/collect_pipeline_metrics.py --output reports/pipeline_metrics.json
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from podcast_scraper import config, workflow
from tests.e2e.fixtures.e2e_http_server import E2EHTTPServer


def collect_pipeline_metrics(output_path: Path, max_episodes: int = 1) -> None:
    """Run a minimal pipeline to collect performance metrics.

    Args:
        output_path: Path to save pipeline metrics JSON file
        max_episodes: Maximum number of episodes to process (default: 1 for speed)
    """
    # Start E2E server to serve test fixtures
    server = E2EHTTPServer()
    try:
        server.start()
        print(f"✅ E2E server started on {server.base_url}")

        # Create config for minimal pipeline run
        # Use fast feed (podcast1) with minimal processing
        rss_url = f"{server.base_url}/podcast1/feed.xml"
        output_dir = str(PROJECT_ROOT / ".tmp" / "pipeline-metrics")

        cfg = config.Config(
            rss_url=rss_url,
            output_dir=output_dir,
            max_episodes=max_episodes,
            transcribe_missing=False,  # Use existing transcripts (faster)
            generate_metadata=True,
            generate_summaries=False,  # Disable to avoid loading large models
            auto_speakers=False,  # Disable to avoid loading spaCy
            metrics_output=str(output_path),  # Save metrics to specified path
            log_level="WARNING",  # Reduce log noise
        )

        print("📊 Running minimal pipeline to collect metrics...")
        print(f"   RSS URL: {rss_url}")
        print(f"   Max episodes: {max_episodes}")
        print(f"   Output: {output_path}")

        # Run pipeline
        saved, summary = workflow.run_pipeline(cfg)

        print(f"✅ Pipeline completed: {saved} episodes processed")
        print(f"✅ Metrics saved to: {output_path}")

        # Verify metrics file was created
        if not output_path.exists():
            print(f"⚠️  Warning: Metrics file not found at {output_path}")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error collecting pipeline metrics: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        server.stop()
        print("✅ E2E server stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Collect pipeline performance metrics by running a minimal pipeline"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/pipeline_metrics.json"),
        help="Output path for pipeline metrics JSON (default: reports/pipeline_metrics.json)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=1,
        help="Maximum number of episodes to process (default: 1)",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    args.output.parent.mkdir(parents=True, exist_ok=True)

    collect_pipeline_metrics(args.output, args.max_episodes)


if __name__ == "__main__":
    main()
