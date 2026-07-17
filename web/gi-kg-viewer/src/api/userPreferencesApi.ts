/**
 * USERPREFS-1 — HTTP wrappers for the user preferences endpoint.
 *
 * Server: GET / PUT / PATCH `/api/app/preferences` (see
 * `src/podcast_scraper/server/routes/app_user_preferences.py`). Payload
 * is a free-form JSON object owned by the client; the server just
 * round-trips it. PATCH with `null` value deletes the key.
 *
 * Every helper degrades gracefully when the endpoint is unavailable
 * (unauthenticated user, offline, dev server down) — returns `null` so
 * callers can fall back to localStorage-only mode without try/catch
 * ceremony.
 */

import { fetchWithTimeout } from './httpClient'

export interface UserPreferencesResponse {
  preferences: Record<string, unknown>
}

const PREFS_URL = '/api/app/preferences'

async function safeParseJson(res: Response): Promise<UserPreferencesResponse | null> {
  try {
    const doc = (await res.json()) as UserPreferencesResponse
    if (!doc || typeof doc.preferences !== 'object' || doc.preferences === null) {
      return { preferences: {} }
    }
    return { preferences: doc.preferences }
  } catch {
    return null
  }
}

/** GET the current user's preferences payload; returns null when the endpoint
 *  is unreachable (401 unauthenticated / 404 / network). Callers should treat
 *  a null result the same as "no preferences yet" — the local defaults win. */
export async function fetchUserPreferences(): Promise<UserPreferencesResponse | null> {
  try {
    const res = await fetchWithTimeout(PREFS_URL, undefined, {
      timeoutDetail: 'app/preferences',
    })
    if (res.status === 401 || res.status === 404) return null
    if (!res.ok) return null
    return await safeParseJson(res)
  } catch {
    return null
  }
}

/** PATCH shallow-merges `updates` into the stored payload. Keys with value
 *  `null` are DELETED server-side. Returns the merged state, or null on
 *  failure (caller keeps the local write). */
export async function patchUserPreferences(
  updates: Record<string, unknown>,
): Promise<UserPreferencesResponse | null> {
  try {
    const res = await fetchWithTimeout(
      PREFS_URL,
      {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ preferences: updates }),
      },
      { timeoutDetail: 'app/preferences' },
    )
    if (!res.ok) return null
    return await safeParseJson(res)
  } catch {
    return null
  }
}

/** PUT replaces the ENTIRE payload — used by the "reset from other device"
 *  path. Prefer PATCH for single-key changes to avoid last-write-wins races
 *  between concurrent tabs. */
export async function replaceUserPreferences(
  preferences: Record<string, unknown>,
): Promise<UserPreferencesResponse | null> {
  try {
    const res = await fetchWithTimeout(
      PREFS_URL,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ preferences }),
      },
      { timeoutDetail: 'app/preferences' },
    )
    if (!res.ok) return null
    return await safeParseJson(res)
  } catch {
    return null
  }
}
