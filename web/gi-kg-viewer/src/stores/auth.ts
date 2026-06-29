/**
 * Viewer auth store (#1128) — the signed-in user + role gates.
 *
 * Roles: `listener` (player only, no viewer access) < `creator` (viewer) < `admin` (+ ops,
 * configuration, user management). The viewer shell renders only for `creator`/`admin`; a signed-in
 * `listener` gets a no-access screen, and an anonymous visitor gets the login landing.
 */
import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { getAuthStatus, loginUrl, logout as apiLogout, type Me, type Role } from '../api/authApi'

export const useAuthStore = defineStore('auth', () => {
  const user = ref<Me | null>(null)
  const loaded = ref(false)
  /** Whether the backend has auth configured. When false, the viewer renders open (no gate). */
  const enabled = ref(false)

  const isAuthenticated = computed(() => user.value !== null)
  const role = computed<Role | null>(() => user.value?.role ?? null)
  const isAdmin = computed(() => role.value === 'admin')
  /** Creator or admin — the gate for any viewer access at all. */
  const canUseViewer = computed(() => role.value === 'creator' || role.value === 'admin')
  /** Gate the login/no-access screens: only when auth is enabled AND the user can't use the viewer. */
  const gated = computed(() => enabled.value && !canUseViewer.value)

  async function refresh(): Promise<void> {
    const status = await getAuthStatus()
    enabled.value = status.enabled
    user.value = status.user
    loaded.value = true
  }

  async function ensureLoaded(): Promise<void> {
    if (!loaded.value) await refresh()
  }

  /** Full-page redirect into the OAuth flow (new users land as `creator`). */
  function login(as?: string): void {
    window.location.assign(loginUrl('creator', as))
  }

  async function logout(): Promise<void> {
    await apiLogout()
    user.value = null
    // Reload to a clean, unauthenticated shell (drops any role-gated in-memory state).
    window.location.reload()
  }

  return {
    user,
    loaded,
    enabled,
    isAuthenticated,
    role,
    isAdmin,
    canUseViewer,
    gated,
    refresh,
    ensureLoaded,
    login,
    logout,
  }
})
