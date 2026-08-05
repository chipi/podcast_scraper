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


@router.get("/export")
async def export_vault(
    request: Request,
    format: str = Query(default="obsidian"),
    since: int = Query(default=0, ge=0, description="Last revision the client applied (0 = full)."),
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
    bundle = app_pkm_export.export_bundle(root, _data_dir(request), user.user_id, since=since)

    manifest = {
        "format": "obsidian",
        "mode": bundle["mode"],
        "revision": bundle["revision"],
        "namespace": bundle["namespace"],
        "written": bundle["written"],
        "removed": bundle["removed"],
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, content in bundle["files"].items():
            zf.writestr(path, content)
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="closelistening-obsidian.zip"',
            "X-Export-Mode": bundle["mode"],
            "X-Export-Revision": str(bundle["revision"]),
            "X-Export-Written": str(len(bundle["written"])),
            "X-Export-Removed": str(len(bundle["removed"])),
            "Access-Control-Expose-Headers": (
                "X-Export-Mode, X-Export-Revision, X-Export-Written, X-Export-Removed"
            ),
        },
    )
