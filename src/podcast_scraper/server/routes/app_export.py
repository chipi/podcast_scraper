"""PKM export routes — the graph-aware Obsidian vault (RFC-113, #1472).

Auth-gated. `GET /api/app/export?format=obsidian` streams a zip of the user's `closelistening/`
vault (highlight notes wikilinked to id-keyed entity/episode notes) + a `manifest.json`. v1 is a
full export the client applies by replacing its `closelistening/` folder. Extractive, bridge-only.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from podcast_scraper.server import app_pkm_export
from podcast_scraper.server.app_corpus_access import corpus_root_or_503
from podcast_scraper.server.app_user_store import User
from podcast_scraper.server.routes.app_auth import get_current_user

router = APIRouter(tags=["app"])


def _data_dir(request: Request) -> Path:
    return Path(request.app.state.app_data_dir)


#: A fixed timestamp for every zip entry. `zipfile.writestr` stamps WALL-CLOCK time per entry, so
#: two exports of an identical vault produced different bytes — no ETag or content-addressed
#: caching is possible, and any test asserting on zip bytes would flake by construction. The
#: content is already deterministic (#44); this makes the container match it. The value is
#: arbitrary and only has to be constant and valid for the DOS date format zip uses (>= 1980).
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


def _write_entry(zf: zipfile.ZipFile, path: str, content: str) -> None:
    """Add one entry with a fixed timestamp, so identical vaults zip to identical bytes."""
    info = zipfile.ZipInfo(path, date_time=_ZIP_EPOCH)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    zf.writestr(info, content)


@router.get(
    "/export",
    # This returns a zip, and without saying so the generated schema claims application/json —
    # FastAPI's default for a handler annotated `-> Response`. Nothing in this repo breaks on it
    # (the web client reads the body as a blob), but /docs then shows a JSON example for a zip, and
    # a client generated from the spec would call .json() on zip bytes and fail at runtime against
    # a server that is behaving correctly.
    response_class=Response,
    responses={
        200: {
            "content": {"application/zip": {}},
            "description": (
                "Zip archive of the vault: the changed `closelistening/…` notes plus a "
                "`manifest.json` listing `written` and `removed` paths. The `X-Export-*` headers "
                "carry the cursor and counts so a client can summarise without unzipping."
            ),
        },
        400: {"description": "Unsupported `format` (only `obsidian` is supported)."},
    },
)
async def export_vault(
    request: Request,
    format: str = Query(default="obsidian"),
    since: int = Query(default=0, ge=0, description="Last revision the client applied (0 = full)."),
    epoch: str | None = Query(
        default=None,
        description=(
            "Vault identity from the client's last export (`X-Export-Epoch`). A revision number "
            "only identifies a snapshot within one epoch; omit it and you get a full export."
        ),
    ),
    user: User = Depends(get_current_user),
) -> Response:
    """Graph-aware vault export as a zip. Incremental when ``since`` matches the last export.

    The zip carries the changed `closelistening/…` notes + a `manifest.json` listing `removed`
    (tombstone) paths. The response headers (`X-Export-*`) let the client advance its cursor and
    show a summary without unzipping. `since=0` (or a mismatch) → a full export.
    """
    if format != "obsidian":
        raise HTTPException(status_code=400, detail="only format=obsidian is supported")
    root = corpus_root_or_503(request)
    bundle = app_pkm_export.export_bundle(
        root, _data_dir(request), user.user_id, since=since, epoch=epoch
    )

    manifest = {
        "format": "obsidian",
        "mode": bundle["mode"],
        "revision": bundle["revision"],
        # The revision is only meaningful WITH this (#41): the counter restarts at 0 whenever the
        # server's export state is lost or unreadable, so a bare integer can collide with one a
        # client is still holding from before the reset. Echo both back next time.
        "epoch": bundle["epoch"],
        "namespace": bundle["namespace"],
        "written": bundle["written"],
        "removed": bundle["removed"],
        # Full export: the client deletes everything under `namespace` before writing `written`
        # (a fallen-behind client can't be sent per-file tombstones — replace all). RFC-113 M8.
        "replace_namespace": bundle["replace_namespace"],
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, content in bundle["files"].items():
            _write_entry(zf, path, content)
        _write_entry(zf, "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="closelistening-obsidian.zip"',
            "X-Export-Mode": bundle["mode"],
            "X-Export-Revision": str(bundle["revision"]),
            "X-Export-Epoch": str(bundle["epoch"]),
            "X-Export-Written": str(len(bundle["written"])),
            "X-Export-Removed": str(len(bundle["removed"])),
            "Access-Control-Expose-Headers": (
                "X-Export-Mode, X-Export-Revision, X-Export-Epoch, "
                "X-Export-Written, X-Export-Removed"
            ),
        },
    )
