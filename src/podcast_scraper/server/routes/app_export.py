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
    user: User = Depends(get_current_user),
) -> Response:
    """Full graph-aware vault export as a zip (`closelistening/…` notes + `manifest.json`)."""
    if format != "obsidian":
        raise HTTPException(status_code=400, detail="only format=obsidian is supported")
    root = corpus_root_or_503(request)
    bundle = app_pkm_export.build_obsidian_bundle(root, _data_dir(request), user.user_id)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, content in bundle["files"].items():
            zf.writestr(path, content)
        zf.writestr("manifest.json", json.dumps(bundle["manifest"], ensure_ascii=False, indent=2))
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="closelistening-obsidian.zip"'},
    )
