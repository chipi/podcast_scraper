/**
 * USERPREFS-1 — global vitest setup.
 *
 * The user-preferences store fires PATCH `/api/app/preferences` from
 * every feature-store write-through (graphLenses, theme, panels, corpus
 * path, …). Under happy-dom those relative fetches resolve against
 * `http://localhost:3000` and hang for the httpClient's 120s timeout —
 * the dangling promises bleed between test files running in the same
 * worker and time out unrelated tests. Mocking the API module at the
 * global setup level short-circuits every write into a resolved-null
 * no-op regardless of which test file forgets to mock it locally.
 *
 * Individual tests remain free to `vi.mock('../api/userPreferencesApi',
 * ...)` locally when they want to observe the arguments passed.
 */
import { vi } from 'vitest'

vi.mock('../api/userPreferencesApi', () => ({
  fetchUserPreferences: vi.fn().mockResolvedValue(null),
  patchUserPreferences: vi.fn().mockResolvedValue(null),
  replaceUserPreferences: vi.fn().mockResolvedValue(null),
}))
