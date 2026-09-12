"""Profile avatar upload + serve (Area E) — a NARROW, dedicated surface, not a general /me update.

A signed-in user uploads their own avatar, which overrides the OAuth-captured one. The upload is
validated hard — content-type allow-list, a magic-byte sniff (never trust the declared type), and a
size cap — then stored in the user's own data dir and served back through a route (the data dir is
not web-mounted). Only the signed-in user can write their own avatar, and both the write and the
read are auth-gated (a signed-in surface); the id is validated and the filename fixed, so the path
cannot traverse out of the user's own dir.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from filelock import FileLock

from podcast_scraper.server import app_user_store
from podcast_scraper.server.app_user_store import is_safe_user_id, User
from podcast_scraper.server.routes.app_auth import get_current_user

router = APIRouter(tags=["app"])

_MAX_BYTES = 2 * 1024 * 1024  # 2 MB
_LOCK_TIMEOUT_S = 15.0
#: declared content-type → (stored extension, served media type). Anything else is rejected.
_ALLOWED = {
    "image/png": ("png", "image/png"),
    "image/jpeg": ("jpg", "image/jpeg"),
    "image/webp": ("webp", "image/webp"),
}
_EXT_MEDIA = {ext: media for (_ct, (ext, media)) in _ALLOWED.items()}


def _data_dir(request: Request) -> Path:
    return Path(request.app.state.app_data_dir)


def _sniff_matches(ext: str, data: bytes) -> bool:
    """Magic-byte check — the file's bytes must match its declared type (defence-in-depth)."""
    if ext == "png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if ext == "jpg":
        return data.startswith(b"\xff\xd8\xff")
    if ext == "webp":
        return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    return False


@router.post("/profile/avatar")
async def upload_avatar(
    request: Request,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
) -> dict[str, str]:
    """Store the signed-in user's uploaded avatar (overrides the OAuth one). 415/413/400 on a bad
    type / oversize / content-type mismatch."""
    # Fast-fail on a declared oversize before reading the spooled body (advisor M3). This is
    # defence-in-depth; the authoritative pre-spool cap is at the edge — `request_body max_size 3MB`
    # scoped to this path in infra/caddy/player.caddy (Caddy 413s a multi-GB body before the app).
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > _MAX_BYTES + 65536:
        raise HTTPException(status_code=413, detail="Image too large (max 2 MB).")
    ctype = (file.content_type or "").split(";", 1)[0].strip().lower()
    if ctype not in _ALLOWED:
        raise HTTPException(status_code=415, detail="Unsupported type; use PNG, JPEG, or WebP.")
    ext, _media = _ALLOWED[ctype]
    data = await file.read(_MAX_BYTES + 1)
    if len(data) > _MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 2 MB).")
    if not _sniff_matches(ext, data):
        raise HTTPException(status_code=400, detail="File content does not match its type.")

    user_dir = _data_dir(request) / "users" / user.user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    # Serialize under a per-user lock (advisor L1) so two concurrent uploads can't leave two
    # avatar.* files, and write atomically via os.replace (advisor L2) so a concurrent GET never
    # serves a half-written image.
    with FileLock(str(user_dir / ".avatar.lock"), timeout=_LOCK_TIMEOUT_S):
        for old in user_dir.glob("avatar.*"):
            old.unlink(missing_ok=True)
        final = user_dir / f"avatar.{ext}"
        # Leading dot so neither the delete glob nor serve_avatar's `avatar.*` glob can match the
        # half-written temp (they'd otherwise serve a partial on a first upload).
        tmp = user_dir / f".avatar.{ext}.tmp"
        tmp.write_bytes(data)
        os.replace(tmp, final)

    # Point the profile at the served route, version-stamped so the client (and PWA cache) refetch.
    served = f"/api/app/profile/{user.user_id}/avatar?v={int(time.time())}"
    app_user_store.set_image(_data_dir(request), user.user_id, served)
    return {"image": served}


@router.get("/profile/{user_id}/avatar")
def serve_avatar(
    user_id: str, request: Request, _user: User = Depends(get_current_user)
) -> FileResponse:
    """Serve a user's stored avatar. Auth-gated (advisor L4 — a signed-in surface); the id is
    validated and the filename is fixed, so the path cannot escape the user's own dir."""
    if not is_safe_user_id(user_id):
        raise HTTPException(status_code=404, detail="No avatar.")
    user_dir = _data_dir(request) / "users" / user_id
    matches = sorted(user_dir.glob("avatar.*")) if user_dir.is_dir() else []
    if not matches:
        raise HTTPException(status_code=404, detail="No avatar.")
    path = matches[0]
    # glob/is_file follow symlinks — require the resolved target stays inside the user's dir, so a
    # link escaping the corpus can't be served (defense-in-depth; same guard as artwork/photo/logo).
    from podcast_scraper.utils.path_validation import resolves_under_root

    if not resolves_under_root(path, user_dir):
        raise HTTPException(status_code=404, detail="No avatar.")
    media = _EXT_MEDIA.get(path.suffix.lstrip("."), "application/octet-stream")
    # codeql[py/path-injection] -- user_id is _is_safe_user_id-validated; filename is a fixed glob.
    # nosniff so a browser can't reinterpret the bytes as anything but the allow-listed image type.
    return FileResponse(
        path=str(path), media_type=media, headers={"X-Content-Type-Options": "nosniff"}
    )
