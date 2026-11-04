"""The file inventory of an app data dir (ADR-169): who owns every file under ``APP_DATA_DIR``.

``db import --from-files`` walks the whole directory and classifies each file as

- ``imported``: an importer in ``db.importers`` reads it into the database;
- ``file``: still a file, its store not yet migrated (declared in ``_FILE_CLAIMS`` below);
- ``ignored``: a lock file;
- ``retired``: data a removed store left behind; no code reads it, it stays with the frozen files
  and is not imported;
- ``ops``: a backup or receipt an ops script in ``scripts/ops/`` writes for its own undo;
- ``unclaimed``: matches nothing declared. That is a store nobody told the importer about, which
  would be lost by a cut-over, so it fails the command.

Names come from the stores' own constants (never a catch-all glob). When a store migrates, its
entries move from ``_file_claims`` to its importer's ``claims()``.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Kind = Literal["imported", "file", "ignored", "retired", "ops"]

#: Per-user files of stores that were removed; the commit that removed each store is the owner.
RETIRED_USER_FILES = {
    "ranking_events.jsonl": "ranking_telemetry (removed in 6c6c08c6, #2327)",
}

#: ``scripts/ops/dedupe_playback_milestones.py``: its per-file backups and its receipts. Literal
#: here because scripts are not importable from the package; a test pins them to the script.
DEDUPE_LOG_NAME = "playback_events.jsonl"
DEDUPE_BACKUP_SUFFIX = ".bak-dedupe-milestones"
DEDUPE_RECEIPTS_FILE = "dedupe_playback_milestones.jsonl"


@dataclass(frozen=True)
class Claim:
    """One declared file shape: a regex over the path relative to the data dir (posix)."""

    pattern: re.Pattern[str]
    kind: Kind
    owner: str


def user_file(name: str, kind: Kind, owner: str) -> Claim:
    """A claim for ``users/<id>/<name>``."""
    return Claim(re.compile(rf"users/[^/]+/{re.escape(name)}"), kind, owner)


def root_file(name: str, kind: Kind, owner: str) -> Claim:
    """A claim for a file directly in the data dir."""
    return Claim(re.compile(re.escape(name)), kind, owner)


def dir_files(dirname: str, file_regex: str, kind: Kind, owner: str) -> Claim:
    """A claim for files of one shape directly under ``<dirname>/``."""
    return Claim(re.compile(rf"{re.escape(dirname)}/{file_regex}"), kind, owner)


def _file_claims() -> list[Claim]:
    """Everything under APP_DATA_DIR that is still a file, by owning store."""
    # Imported here: the server modules import the database layer.
    from podcast_scraper.server import (
        app_access_store,
        app_account_deletion,
        app_collections_store,
        app_comms_store,
        app_corpus_revision,
        app_digest_health,
        app_mcp_tokens,
        app_new_episode_digest,
        app_notifications_store,
        app_oauth_server,
        app_pkm_export,
        app_push_store,
        app_user_preferences,
        app_user_state,
        board_summaries,
        reference_integrity,
        rekey_references,
        storyline_summaries,
    )
    from podcast_scraper.server.routes import app_profile

    claims: list[Claim] = []

    def per_user(owner: str, *names: str) -> None:
        claims.extend(user_file(n, "file", owner) for n in names)

    def at_root(owner: str, *names: str) -> None:
        claims.extend(root_file(n, "file", owner) for n in names)

    per_user("user_state", *(f"{n}.json" for n in app_user_state.STATE_NAMES))
    per_user("user_state", *app_user_state.EVENT_LOGS)
    per_user("collections", app_collections_store._FILE_NAME)
    per_user("comms", app_comms_store._FILE_NAME)
    per_user("corpus_revision", app_corpus_revision._FILE_NAME)
    per_user("graph_telemetry", "graph_events.jsonl")
    per_user("mcp_tokens", app_mcp_tokens._TOKENS_FILE)
    per_user("new_episode_digest", app_new_episode_digest._LEDGER_NAME)
    per_user("notifications", app_notifications_store._FILE_NAME)
    per_user("pkm_export", app_pkm_export._STATE_FILE)
    per_user("push", app_push_store._FILE_NAME)
    per_user("user_preferences", app_user_preferences._FILE_NAME)
    per_user("account_deletion", app_account_deletion._APPLE_TOKEN_FILE)
    per_user("board_summaries", board_summaries.SUMMARIES_FILE)
    per_user("reference_integrity", f"{reference_integrity.FLAGS_STORE}.json")
    claims.extend(
        Claim(re.compile(rf"users/[^/]+/avatar\.{re.escape(ext)}"), "file", "avatar")
        for ext, _media in app_profile._ALLOWED.values()
    )

    at_root("access_policy", app_access_store._FILE_NAME)
    at_root("audit", "audit.jsonl")
    at_root("ranking_config", "ranking_config.json")
    at_root("releases", "app_releases.json", "player_release.json")
    at_root("digest_health", app_digest_health.STATE_FILENAME)
    at_root("storyline_summaries", storyline_summaries.SUMMARIES_FILE)
    at_root("mcp_tokens", app_mcp_tokens._INDEX_FILE)
    at_root(
        "oauth_server",
        app_oauth_server._CLIENTS_FILE,
        app_oauth_server._GRANTS_FILE,
        app_oauth_server._CONSENTS_FILE,
        app_oauth_server._USE_FILE,
    )
    at_root("board_summaries", board_summaries.SETTINGS_FILE, board_summaries.LEDGER_FILE)
    at_root("rekey_references", rekey_references.RECEIPTS_FILE)

    # Dynamic names, each a shape the writing code produces.
    claims.append(dir_files("outbox", r"[0-9a-f]{64}\.json", "file", "outbox"))
    claims.append(dir_files("magic_link_used", r"[A-Za-z0-9_-]+\.json", "file", "magic_link"))
    claims.append(dir_files("magic_link_recent", r"[0-9a-f]{32}\.txt", "file", "magic_link"))

    claims.extend(user_file(n, "retired", owner) for n, owner in RETIRED_USER_FILES.items())
    claims.append(
        user_file(DEDUPE_LOG_NAME + DEDUPE_BACKUP_SUFFIX, "ops", "dedupe_playback_milestones")
    )
    claims.append(root_file(DEDUPE_RECEIPTS_FILE, "ops", "dedupe_playback_milestones"))
    return claims


_LOCK = Claim(re.compile(r"(?:.*/)?[^/]*\.lock"), "ignored", "lock")


def declared_claims() -> list[Claim]:
    """Every declared file shape: imported, still a file, ignored."""
    from podcast_scraper.db.importers import import_claims

    return [*import_claims(), *_file_claims(), _LOCK]


@dataclass
class Inventory:
    """The result of classifying every file under a data dir."""

    #: kind -> count; ``unclaimed`` is its own key.
    by_class: Counter[str] = field(default_factory=Counter)
    #: (kind, owner) -> count
    by_owner: Counter[tuple[str, str]] = field(default_factory=Counter)
    unclaimed: list[str] = field(default_factory=list)


def scan(data_dir: Path, claims: list[Claim] | None = None) -> Inventory:
    """Classify every file under ``data_dir``."""
    declared = claims if claims is not None else declared_claims()
    out = Inventory()
    for dirpath, _dirs, files in os.walk(data_dir):
        for fname in files:
            rel = (Path(dirpath) / fname).relative_to(data_dir).as_posix()
            hit = next((c for c in declared if c.pattern.fullmatch(rel)), None)
            if hit is None:
                out.by_class["unclaimed"] += 1
                out.unclaimed.append(rel)
            else:
                out.by_class[hit.kind] += 1
                out.by_owner[(hit.kind, hit.owner)] += 1
    out.unclaimed.sort()
    return out
