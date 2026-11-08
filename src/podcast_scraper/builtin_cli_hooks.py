"""Today's in-tree subcommands and batch-end refreshes on the CLI hooks (E04 S7).

Each entry imports its code inside its body. An entry goes when its code moves to an extension,
which then contributes it instead (``Extension.cli_commands``, ``pipeline_hooks``).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from .extensions import CliCommand


def _parse_gi(argv: Any) -> Any:
    from podcast_scraper.insight_stage import cli

    return cli._parse_gi_args(list(argv))


def _run_gi(args: Any, log: Any) -> int:
    from podcast_scraper.insight_stage import cli

    return cli._run_gi(args, log=log)


def _parse_gi_repair(argv: Any) -> Any:
    from podcast_scraper.insight_stage import cli

    return cli._parse_gi_repair_argv(list(argv))


def _run_gi_repair(args: Any, log: Any) -> int:
    from podcast_scraper.insight_stage import cli

    return cli._run_gi_repair_cli(args, log)


def _parse_identity_ledger(argv: Any) -> Any:
    from podcast_scraper.identity.ledger_cli import parse_identity_ledger_argv

    return parse_identity_ledger_argv(list(argv))


def _run_identity_ledger(args: Any, log: Any) -> int:
    from podcast_scraper.identity.ledger_cli import run_identity_ledger_cli

    return run_identity_ledger_cli(args, log)


def _parse_enrich_edges(argv: Any) -> Any:
    from podcast_scraper.insight_stage import edges_cli

    return edges_cli.parse_enrich_edges_argv(list(argv))


def _run_enrich_edges(args: Any, log: Any) -> int:
    from podcast_scraper.insight_stage import edges_cli

    return edges_cli.run_enrich_edges_cli(args, log)


def _parse_cluster_corpus_topics(argv: Any) -> Any:
    from podcast_scraper.kg import topic_clustering_cli

    return topic_clustering_cli.parse_cluster_corpus_topics_argv(list(argv))


def _run_cluster_corpus_topics(args: Any, log: Any) -> int:
    from podcast_scraper.kg import topic_clustering_cli

    return topic_clustering_cli.run_cluster_corpus_topics_cli(args, log)


def _refresh_ledger_after_batch(corpus_parent: Any, cfg: Any, log: Any) -> None:
    from podcast_scraper.identity.refresh_guard import refresh_ledger_after_batch

    refresh_ledger_after_batch(corpus_parent, cfg, log)


def _ledger_refresh(root: Any, *, after: str, log: Any = None) -> bool:
    from podcast_scraper.identity.refresh_guard import refresh_ledger_after

    return refresh_ledger_after(root, after=after, log=log)


CLI_COMMANDS: Dict[str, CliCommand] = {
    "gi": CliCommand(parse=_parse_gi, run=_run_gi),
    "gi-repair": CliCommand(parse=_parse_gi_repair, run=_run_gi_repair),
    "identity-ledger": CliCommand(parse=_parse_identity_ledger, run=_run_identity_ledger),
    "enrich-edges": CliCommand(parse=_parse_enrich_edges, run=_run_enrich_edges),
    "cluster-corpus-topics": CliCommand(
        parse=_parse_cluster_corpus_topics, run=_run_cluster_corpus_topics
    ),
}
BATCH_COMPLETED: Tuple[Callable[[Any, Any, Any], None], ...] = (_refresh_ledger_after_batch,)
LEDGER_REFRESH: Optional[Callable[..., bool]] = _ledger_refresh
