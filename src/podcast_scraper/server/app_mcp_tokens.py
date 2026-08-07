"""Per-user MCP personal-access tokens (PATs) — RFC-112 slice 1 (#1471).

The secondary auth path for the remote MCP server (Claude Code / Cursor / API / local dev; OAuth is
the primary path for claude.ai per-user connectors). A PAT is a high-entropy bearer token shown
**once** at creation and stored **SHA-256-hashed** — the plaintext is never persisted. Verification
is **O(1)** via a global ``hash → user_id`` index (RFC-112 §4), avoiding an all-users scan per
connect. A dead/rotated token is revoked by id.

Layout:
- ``<data_dir>/users/<id>/mcp_tokens.json`` = ``[{id, label, hash, created_at, last_used_at}]``.
- ``<data_dir>/mcp_token_index.json`` = ``{hash: user_id}`` (global, lock-serialised).

Only the hash is comparable at rest — a leaked store cannot reconstruct a token.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
import uuid
from pathlib import Path
from typing import Any

from filelock import FileLock

from podcast_scraper.server.app_user_store import _is_safe_user_id
from podcast_scraper.server.atomic_write import atomic_write_text

_LOCK_TIMEOUT_S = 5.0
_TOKENS_FILE = "mcp_tokens.json"
_INDEX_FILE = "mcp_token_index.json"
_PREFIX = "clp_mcp_"


def _tokens_path(data_dir: Path, user_id: str) -> Path:
    return data_dir / "users" / user_id / _TOKENS_FILE


def _tokens_lock(data_dir: Path, user_id: str) -> FileLock:
    path = _tokens_path(data_dir, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(path.with_name(f".{_TOKENS_FILE}.lock")), timeout=_LOCK_TIMEOUT_S)


def _index_path(data_dir: Path) -> Path:
    return data_dir / _INDEX_FILE


def _index_lock(data_dir: Path) -> FileLock:
    data_dir.mkdir(parents=True, exist_ok=True)
    return FileLock(str(_index_path(data_dir).with_suffix(".lock")), timeout=_LOCK_TIMEOUT_S)


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _read_tokens(data_dir: Path, user_id: str) -> list[dict[str, Any]]:
    data = _read_json(_tokens_path(data_dir, user_id), [])
    return (
        [t for t in data if isinstance(t, dict) and t.get("id")] if isinstance(data, list) else []
    )


def list_tokens(data_dir: Path, user_id: str) -> list[dict[str, Any]]:
    """Token metadata (never the hash/plaintext): ``[{id, label, created_at, last_used_at}]``."""
    if not _is_safe_user_id(user_id):
        return []
    return [
        {k: t.get(k) for k in ("id", "label", "created_at", "last_used_at")}
        for t in _read_tokens(data_dir, user_id)
    ]


def create_token(data_dir: Path, user_id: str, label: str) -> tuple[str, dict[str, Any]]:
    """Mint a token: returns ``(plaintext_shown_once, metadata)``. Stores only the hash + index."""
    if not _is_safe_user_id(user_id):
        raise ValueError("unsafe user id")
    plaintext = _PREFIX + secrets.token_urlsafe(32)
    token_hash = _hash(plaintext)
    record = {
        "id": f"mtk_{uuid.uuid4().hex[:12]}",
        "label": (label or "").strip()[:120] or "agent",
        "hash": token_hash,
        "created_at": int(time.time()),
        "last_used_at": None,
    }
    with _tokens_lock(data_dir, user_id):
        tokens = _read_tokens(data_dir, user_id)
        tokens.append(record)
        atomic_write_text(
            _tokens_path(data_dir, user_id), json.dumps(tokens, ensure_ascii=False, indent=2)
        )
    with _index_lock(data_dir):
        index = _read_json(_index_path(data_dir), {})
        index = index if isinstance(index, dict) else {}
        index[token_hash] = user_id
        atomic_write_text(_index_path(data_dir), json.dumps(index, ensure_ascii=False, indent=2))
    meta = {k: record[k] for k in ("id", "label", "created_at", "last_used_at")}
    return plaintext, meta


def revoke_token(data_dir: Path, user_id: str, token_id: str) -> bool:
    """Revoke a token by id (drops it from the store + the index). Returns True when removed."""
    if not _is_safe_user_id(user_id):
        return False
    removed_hash: str | None = None
    with _tokens_lock(data_dir, user_id):
        tokens = _read_tokens(data_dir, user_id)
        kept = []
        for t in tokens:
            if t.get("id") == token_id:
                removed_hash = t.get("hash")
            else:
                kept.append(t)
        if removed_hash is None:
            return False
        atomic_write_text(
            _tokens_path(data_dir, user_id), json.dumps(kept, ensure_ascii=False, indent=2)
        )
    with _index_lock(data_dir):
        index = _read_json(_index_path(data_dir), {})
        if isinstance(index, dict) and index.pop(removed_hash, None) is not None:
            atomic_write_text(
                _index_path(data_dir), json.dumps(index, ensure_ascii=False, indent=2)
            )
    return True


def verify_token(data_dir: Path, token: str) -> str | None:
    """Resolve a presented token to its owning ``user_id`` (O(1) via the index), else None.

    Also stamps ``last_used_at`` on the matching token record. A token whose index entry is stale
    (record gone) returns None.
    """
    if not token:
        return None
    token_hash = _hash(token)
    index = _read_json(_index_path(data_dir), {})
    user_id = index.get(token_hash) if isinstance(index, dict) else None
    if not user_id or not _is_safe_user_id(str(user_id)):
        return None
    with _tokens_lock(data_dir, str(user_id)):
        tokens = _read_tokens(data_dir, str(user_id))
        match = next((t for t in tokens if t.get("hash") == token_hash), None)
        if match is None:
            return None
        match["last_used_at"] = int(time.time())
        atomic_write_text(
            _tokens_path(data_dir, str(user_id)), json.dumps(tokens, ensure_ascii=False, indent=2)
        )
    return str(user_id)
