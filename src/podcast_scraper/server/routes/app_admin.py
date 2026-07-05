"""Admin user-management routes for the platform (#1128).

Admin-only CRUD over the shared identity store (``app_user_store``) — the surface the viewer's
Admin → Users view drives. Every route depends on :func:`get_admin_user` (403 for non-admins) and
every mutation is appended to the audit log. A **self-lockout guard** prevents an admin from
demoting, deactivating, or deleting their own account (so the platform can't be locked out of
administration).

Auth/session is the *same* mechanism the Learning Player uses (RFC-098 §2); the viewer reuses it via
the shared ``lp_session`` cookie. Mounted at the ``/api/app`` prefix alongside ``app_auth``.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from podcast_scraper.server import app_roles
from podcast_scraper.server.app_audit import append_audit
from podcast_scraper.server.app_user_store import (
    create_user,
    delete_user,
    get_user,
    list_users,
    set_disabled,
    set_role,
    User,
    user_id_for,
)
from podcast_scraper.server.routes.app_auth import get_admin_user

router = APIRouter(tags=["app-admin"])


class UserOut(BaseModel):
    """A user as seen by the admin surface."""

    user_id: str
    email: str
    name: str
    role: str
    disabled: bool
    provider: str


class CreateUserBody(BaseModel):
    """Pre-provision a user with a chosen role (local-first; mock identity)."""

    email: str = Field(..., min_length=1)
    name: str = Field(default="")
    role: str = Field(default=app_roles.CREATOR)


class PatchUserBody(BaseModel):
    """Partial update — set role and/or active state. Omitted fields are left unchanged."""

    role: str | None = None
    disabled: bool | None = None


def _data_dir(request: Request) -> Path:
    raw = getattr(request.app.state, "app_data_dir", None)
    if raw is None:
        raise HTTPException(status_code=503, detail="User store is not configured.")
    return Path(raw)


def _audit(request: Request, **record: object) -> None:
    append_audit(getattr(request.app.state, "audit_path", None), record)


def _out(user: User) -> UserOut:
    return UserOut(
        user_id=user.user_id,
        email=user.email,
        name=user.name,
        role=user.role,
        disabled=user.disabled,
        provider=user.provider,
    )


@router.get("/admin/users", response_model=list[UserOut])
async def admin_list_users(
    request: Request, admin: User = Depends(get_admin_user)
) -> list[UserOut]:
    """List every platform user (admin-only)."""
    return [_out(u) for u in list_users(_data_dir(request))]


@router.post("/admin/users", response_model=UserOut, status_code=201)
async def admin_create_user(
    body: CreateUserBody, request: Request, admin: User = Depends(get_admin_user)
) -> UserOut:
    """Pre-provision a user with a role. 409 if that identity already exists, 422 on a bad role."""
    if not app_roles.is_role(body.role):
        raise HTTPException(status_code=422, detail=f"Unknown role: {body.role!r}.")
    data_dir = _data_dir(request)
    provider, subject = "mock", f"admin-created:{body.email.strip().lower()}"
    uid = user_id_for(provider, subject)
    if get_user(data_dir, uid) is not None:
        raise HTTPException(status_code=409, detail="A user with that email already exists.")
    user = create_user(
        data_dir,
        provider=provider,
        subject=subject,
        email=body.email.strip(),
        name=body.name.strip() or body.email.strip(),
        role=body.role,
    )
    _audit(request, action="admin.user.create", by=admin.user_id, user=uid, role=user.role)
    return _out(user)


@router.patch("/admin/users/{user_id}", response_model=UserOut)
async def admin_patch_user(
    user_id: str,
    body: PatchUserBody,
    request: Request,
    admin: User = Depends(get_admin_user),
) -> UserOut:
    """Change a user's role and/or active state (admin-only, self-lockout-guarded)."""
    data_dir = _data_dir(request)
    user = get_user(data_dir, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="No such user.")
    is_self = user_id == admin.user_id

    if body.role is not None:
        if not app_roles.is_role(body.role):
            raise HTTPException(status_code=422, detail=f"Unknown role: {body.role!r}.")
        if is_self and not app_roles.is_admin(body.role):
            raise HTTPException(status_code=400, detail="You cannot remove your own admin role.")
        if app_roles.normalize_role(body.role) != user.role:
            set_role(data_dir, user_id, body.role)
            _audit(
                request,
                action="admin.user.role",
                by=admin.user_id,
                user=user_id,
                role=app_roles.normalize_role(body.role),
            )

    if body.disabled is not None:
        if is_self and body.disabled:
            raise HTTPException(status_code=400, detail="You cannot deactivate your own account.")
        if bool(body.disabled) != user.disabled:
            set_disabled(data_dir, user_id, body.disabled)
            _audit(
                request,
                action="admin.user.disabled",
                by=admin.user_id,
                user=user_id,
                disabled=bool(body.disabled),
            )

    updated = get_user(data_dir, user_id)
    assert updated is not None  # we hold no lock, but the user was just present
    return _out(updated)


@router.delete("/admin/users/{user_id}", status_code=204)
async def admin_delete_user(
    user_id: str, request: Request, admin: User = Depends(get_admin_user)
) -> None:
    """Hard-delete a user (admin-only). An admin cannot delete their own account."""
    if user_id == admin.user_id:
        raise HTTPException(status_code=400, detail="You cannot delete your own account.")
    data_dir = _data_dir(request)
    if not delete_user(data_dir, user_id):
        raise HTTPException(status_code=404, detail="No such user.")
    _audit(request, action="admin.user.delete", by=admin.user_id, user=user_id)
