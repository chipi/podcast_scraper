"""``python -m podcast_obs`` — the "basics" CLI: probe any deploy directly, no MCP needed.

Examples::

    python -m podcast_obs health  --target local
    python -m podcast_obs runs    --target prod --limit 5
    python -m podcast_obs summary --target prod

Output is pretty JSON (scriptable). Exit code is 0 when the probe succeeded, 1 when it
failed (e.g. unreachable or not configured), 2 on a usage/config error.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from .aggregate import summary as _summary
from .config import ObservabilityConfig, ObservabilityConfigError
from .sources import github, grafana, langfuse, loki, prod_api, sentry


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="podcast_obs",
        description="Prod observability control plane (#803) — probe any deploy.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML config (else $PODCAST_OBS_CONFIG, else PODCAST_OBS_* env vars).",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Named target to observe (default: the config's default_target).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("health", help="Full /api/health payload from the target deploy.")
    sub.add_parser("version", help="Running code version + corpus stamp the deploy serves.")
    runs = sub.add_parser("runs", help="Recent pipeline runs (/api/jobs), newest first.")
    runs.add_argument("--limit", type=int, default=10, help="Max runs to return (default 10).")
    deploys = sub.add_parser("deploys", help="Recent deploy-prod.yml runs (GitHub Actions).")
    deploys.add_argument("--limit", type=int, default=10, help="Max deploys (default 10).")
    sub.add_parser("cost-today", help="LLM spend over the last 24h (Loki).")
    logs = sub.add_parser("logs", help="Recent container logs — error-ish by default (Loki).")
    logs.add_argument("--level", default="error", help="'error' filters error-ish lines; else all.")
    logs.add_argument(
        "--service", default=None, help="Filter to one compose service (api/pipeline/…)."
    )
    logs.add_argument(
        "--window", default="1h", help="Lookback window, e.g. 30m/1h/24h (default 1h)."
    )
    logs.add_argument("--limit", type=int, default=50, help="Max log lines (default 50).")
    logs.add_argument("--contains", default=None, help="Also require this substring in the line.")
    errors = sub.add_parser("errors", help="Recent unresolved Sentry issues for env=prod.")
    errors.add_argument("--window", default="24h", help="statsPeriod, e.g. 24h/14d (default 24h).")
    errors.add_argument(
        "--limit", type=int, default=10, help="Max issues per project (default 10)."
    )
    alerts = sub.add_parser("alerts", help="Current Grafana alerts.")
    alerts.add_argument("--limit", type=int, default=20, help="Max alerts (default 20).")
    traces = sub.add_parser("traces", help="Recent Langfuse LLM traces for the deploy.")
    traces.add_argument("--limit", type=int, default=10, help="Max traces (default 10).")
    sub.add_parser("summary", help="Control-plane glance: every source for the target.")
    serve = sub.add_parser("serve", help="Run the MCP server (agent-facing) over the core.")
    serve.add_argument(
        "--transport",
        choices=["stdio", "sse", "http"],
        default="stdio",
        help="stdio (local), or sse/http for a networked control plane (default stdio).",
    )
    serve.add_argument("--host", default="127.0.0.1", help="Bind host for sse/http.")
    serve.add_argument("--port", type=int, default=8848, help="Bind port for sse/http.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        config = ObservabilityConfig.load(args.config)
        target = config.target(args.target)
    except ObservabilityConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.command == "health":
        result = prod_api.health(target)
    elif args.command == "version":
        result = prod_api.deployed_version(target)
    elif args.command == "runs":
        result = prod_api.recent_pipeline_runs(target, limit=args.limit)
    elif args.command == "deploys":
        result = github.recent_deploys(target, limit=args.limit)
    elif args.command == "cost-today":
        result = loki.cost_today(target)
    elif args.command == "logs":
        result = loki.recent_logs(
            target,
            level=args.level,
            service=args.service,
            window=args.window,
            limit=args.limit,
            contains=args.contains,
        )
    elif args.command == "errors":
        result = sentry.recent_errors(target, window=args.window, limit=args.limit)
    elif args.command == "alerts":
        result = grafana.recent_alerts(target, limit=args.limit)
    elif args.command == "traces":
        result = langfuse.recent_traces(target, limit=args.limit)
    elif args.command == "summary":
        result = _summary(target)
    elif args.command == "serve":
        from .mcp_server import run_server

        transport = "streamable-http" if args.transport == "http" else args.transport
        run_server(config, transport=transport, host=args.host, port=args.port)
        return 0
    else:  # pragma: no cover — argparse enforces the choices
        parser.error(f"unknown command {args.command!r}")
        return 2

    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
