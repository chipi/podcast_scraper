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

import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from podcast_scraper.server import app_access_store, app_roles
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_audit import append_audit
from podcast_scraper.server.app_user_store import (
    create_user,
    delete_user,
    get_user,
    list_users,
    set_disabled,
    set_mcp_access,
    set_role,
    User,
    user_id_for,
)
from podcast_scraper.server.routes.app_auth import get_admin_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["app-admin"])


class UserOut(BaseModel):
    """A user as seen by the admin surface."""

    user_id: str
    email: str
    name: str
    role: str
    disabled: bool
    provider: str
    mcp_access: bool = False


class CreateUserBody(BaseModel):
    """Pre-provision a user with a chosen role (local-first; mock identity)."""

    email: str = Field(..., min_length=1)
    name: str = Field(default="")
    role: str = Field(default=app_roles.CREATOR)


class PatchUserBody(BaseModel):
    """Partial update — role, active state, and/or the MCP entitlement. Omitted fields untouched."""

    role: str | None = None
    disabled: bool | None = None
    mcp_access: bool | None = None


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
        mcp_access=user.mcp_access,
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

    if body.mcp_access is not None and bool(body.mcp_access) != user.mcp_access:
        set_mcp_access(data_dir, user_id, body.mcp_access)
        _audit(
            request,
            action="admin.user.mcp_access",
            by=admin.user_id,
            user=user_id,
            mcp_access=bool(body.mcp_access),
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


class AccessPolicyOut(BaseModel):
    """The sign-in access policy as the admin surface sees it."""

    mode: str
    allowed_emails: list[str]
    allowed_domains: list[str]
    #: False when no policy file exists yet and the env policy is in force. Worth surfacing: it is
    #: the difference between "nobody has set this" and "someone set exactly this".
    persisted: bool


class AccessPolicyBody(BaseModel):
    """A replacement access policy. Absent lists are treated as empty, not as "leave unchanged"."""

    #: Literal, so a typo 422s instead of being silently coerced. A misspelled mode used to
    #: return 200 with mode `allowlist` — fail-closed, but the caller was told nothing and would
    #: reasonably believe they had opened signup.
    mode: Literal["allowlist", "open"] = "allowlist"
    allowed_emails: list[str] = Field(default_factory=list)
    allowed_domains: list[str] = Field(default_factory=list)


def _policy_out(policy: AccessPolicy | None, *, persisted: bool) -> AccessPolicyOut:
    if policy is None:  # no env policy configured either
        return AccessPolicyOut(
            mode="allowlist", allowed_emails=[], allowed_domains=[], persisted=persisted
        )
    return AccessPolicyOut(
        mode=policy.mode,
        allowed_emails=sorted(policy.allowed_emails),
        allowed_domains=sorted(policy.allowed_domains),
        persisted=persisted,
    )


@router.get("/admin/access-policy", response_model=AccessPolicyOut)
async def admin_get_access_policy(
    request: Request, admin: User = Depends(get_admin_user)
) -> AccessPolicyOut:
    """Who may sign in — the persisted policy when there is one, else the startup env policy."""
    data_dir = getattr(request.app.state, "app_data_dir", None)
    persisted = app_access_store.load_policy(data_dir)
    if persisted is not None:
        return _policy_out(persisted, persisted=True)
    return _policy_out(getattr(request.app.state, "access_policy", None), persisted=False)


@router.put("/admin/access-policy", response_model=AccessPolicyOut)
async def admin_put_access_policy(
    body: AccessPolicyBody, request: Request, admin: User = Depends(get_admin_user)
) -> AccessPolicyOut:
    """Replace the sign-in access policy. Takes effect on the NEXT sign-in — no redeploy.

    This is a full replacement, not a merge, so it can revoke as well as grant. That makes it
    possible to lock everyone out with one bad call, which is why the guard below exists — the same
    self-lockout protection the user routes have, for the same reason: the platform must not be
    able to lock itself out of its own administration.
    """
    policy = app_access_store.policy_from_dict(body.model_dump())
    # RFC-108: on the operator-public deployment the allowlist is the ONLY authZ boundary — any
    # signed-in account self-grants `creator` over the operator-read corpus via `?grant=creator`
    # (`app_roles.resolve_login_role`). `app.py` refuses to BOOT that surface under open signup for
    # exactly this reason; without this check the same surface could be opened at runtime through
    # an endpoint, and would then boot again clean because the guard used to read only env.
    if policy.mode == "open" and getattr(request.app.state, "operator_public", False):
        raise HTTPException(
            status_code=400,
            detail=(
                "Open signup is refused on the operator-public surface: it would expose the "
                "operator-read corpus to any authenticated Google account. Use an allowlist."
            ),
        )
    if not policy.is_allowed(admin.email):
        raise HTTPException(
            status_code=400,
            detail=(
                "That policy would lock you out: "
                f"{admin.email} is not permitted by it. Add your own address, or use mode 'open'."
            ),
        )
    # The guard above only protects the CALLER. Another bootstrap admin can still be excluded, and
    # being on APP_ADMIN_EMAILS does not get you past the sign-in gate — the policy is checked
    # first. Refusing would make one admin unable to manage the list without the others, so this
    # warns rather than blocks, and the audit entry below names who was dropped.
    shut_out = sorted(
        e
        for e in getattr(request.app.state, "admin_emails", frozenset())
        if not policy.is_allowed(e)
    )
    if shut_out:
        logger.warning(
            "access policy written by %s excludes bootstrap admin(s) %s — they will not be able "
            "to sign in again once their current session expires",
            admin.email,
            ", ".join(shut_out),
        )
    data_dir = _data_dir(request)
    before = app_access_store.effective_policy(
        data_dir, getattr(request.app.state, "access_policy", None)
    )
    saved = app_access_store.save_policy(data_dir, policy)
    # The ADDRESSES, not counts. "3 emails -> 2 emails" cannot answer "who did we just cut off",
    # which is the only question anyone asks this log after an access incident.
    _audit(
        request,
        action="admin.access_policy.replace",
        by=admin.user_id,
        mode=saved.mode,
        was_mode=before.mode if before is not None else None,
        allowed_emails=sorted(saved.allowed_emails),
        allowed_domains=sorted(saved.allowed_domains),
        removed_emails=sorted(before.allowed_emails - saved.allowed_emails) if before else [],
        added_emails=sorted(saved.allowed_emails - before.allowed_emails) if before else [],
    )
    return _policy_out(saved, persisted=True)
