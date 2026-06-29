/**
 * Auth + admin user-management API for the viewer (#1128).
 *
 * The viewer reuses the Learning Player's auth: one backend, and the `lp_session` cookie is
 * host-scoped so it's shared across the player/viewer/api origins. These call the same
 * `/api/app/*` routes the player uses; the viewer adds the role-gated `/admin/users` surface.
 */
import { fetchWithTimeout } from './httpClient'

export type Role = 'listener' | 'creator' | 'admin'

export interface Me {
  user_id: string
  email: string
  name: string
  role: Role
  disabled: boolean
}

export interface AdminUser {
  user_id: string
  email: string
  name: string
  role: Role
  disabled: boolean
  provider: string
}

const BASE = '/api/app'

/** The current user, or `null` when not signed in (the `/me` route 401s for anonymous). */
export async function getMe(): Promise<Me | null> {
  const res = await fetchWithTimeout(`${BASE}/me`)
  if (res.status === 401) return null
  if (!res.ok) throw new Error(`GET /me failed: ${res.status}`)
  return (await res.json()) as Me
}

/** Full-page login URL. The viewer grants `creator` to new users by default. */
export function loginUrl(grant: Role | null = 'creator', as?: string): string {
  const params = new URLSearchParams()
  if (grant) params.set('grant', grant)
  if (as) params.set('as', as)
  const qs = params.toString()
  return `${BASE}/auth/login${qs ? `?${qs}` : ''}`
}

export async function logout(): Promise<void> {
  await fetchWithTimeout(`${BASE}/auth/logout`, { method: 'POST' })
}

// --- admin (role-gated; 403 for non-admins) -----------------------------------------------------

export async function listUsers(): Promise<AdminUser[]> {
  const res = await fetchWithTimeout(`${BASE}/admin/users`)
  if (!res.ok) throw new Error(`GET /admin/users failed: ${res.status}`)
  return (await res.json()) as AdminUser[]
}

export async function createUser(body: {
  email: string
  name?: string
  role?: Role
}): Promise<AdminUser> {
  const res = await fetchWithTimeout(`${BASE}/admin/users`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(await errorMessage(res, 'create user'))
  return (await res.json()) as AdminUser
}

export async function patchUser(
  userId: string,
  body: { role?: Role; disabled?: boolean },
): Promise<AdminUser> {
  const res = await fetchWithTimeout(`${BASE}/admin/users/${encodeURIComponent(userId)}`, {
    method: 'PATCH',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(await errorMessage(res, 'update user'))
  return (await res.json()) as AdminUser
}

export async function deleteUser(userId: string): Promise<void> {
  const res = await fetchWithTimeout(`${BASE}/admin/users/${encodeURIComponent(userId)}`, {
    method: 'DELETE',
  })
  if (!res.ok && res.status !== 204) throw new Error(await errorMessage(res, 'delete user'))
}

/** Pull a FastAPI `{detail}` message off an error response for a readable toast. */
async function errorMessage(res: Response, action: string): Promise<string> {
  try {
    const body = (await res.json()) as { detail?: string }
    if (body?.detail) return body.detail
  } catch {
    /* non-JSON body */
  }
  return `Failed to ${action} (${res.status})`
}
