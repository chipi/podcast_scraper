import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAuthStore } from '../stores/auth'

/**
 * Per-user actions for signed-out visitors (#1590).
 *
 * Every auth-gated affordance used to render `v-if="auth.isAuthenticated"`, so a signed-out listener
 * sitting in the player saw **no evidence that capture, queue, favourites or follows existed at
 * all**. The capabilities that differentiate this product were invisible to exactly the people who
 * had not yet decided to sign up, and the only prompts were the two header buttons.
 *
 * Hiding is also the wrong shape of honesty: the control is not unavailable, it is *deferred*. So
 * show it, and let the tap explain the requirement — routing to sign-in with a redirect back to
 * where they were, so the action they wanted is one step away rather than lost.
 *
 * Usage:
 *   const { gated, isGated } = useSignInGate()
 *   <button :aria-label="isGated ? t('auth.signInTo', {...}) : t('queue.add')" @click="gated(doIt)">
 */
export function useSignInGate() {
  const auth = useAuthStore()
  const router = useRouter()
  const route = useRoute()

  /**
   * True when the visitor is signed out, i.e. the control is a teaser rather than a live action.
   *
   * Drives labels only. `gated()` re-checks after resolving the session, so a label that renders
   * "Sign in to…" for the instant before hydration completes still performs the real action.
   */
  const isGated = computed(() => !auth.isAuthenticated)

  /**
   * Run `action` when signed in; otherwise send them to sign-in with a redirect back here.
   *
   * Returns a handler, so it composes directly with `@click`. The redirect uses `fullPath` so query
   * and hash survive — landing back on a search result or a timestamped episode, not just the route.
   */
  function gated(action: () => void | Promise<void>) {
    return (): void => {
      void (async () => {
        // Resolve the session BEFORE deciding. `isAuthenticated` is false until App.vue's onMounted
        // refresh lands, so a tap in that window sent a signed-in user to the login page — their
        // own session, thrown away because they were quick. `ensureLoaded()` is a no-op once the
        // session is known, so the common path stays synchronous in effect.
        await auth.ensureLoaded().catch(() => {})
        if (auth.isAuthenticated) {
          await action()
          return
        }
        // A navigation can legitimately fail — a guard redirects, the user navigates away
        // mid-flight, or the route is not registered. Unhandled it becomes an unhandled promise
        // rejection: console noise in the browser, and in Vitest an "unhandled error" that can
        // fail an unrelated test file. Staying put is already the right fallback.
        //
        // try/catch rather than `.catch()`: an unmatched route makes `router.push` throw
        // SYNCHRONOUSLY from `resolve()`, before it ever returns a promise, so a `.catch()` on its
        // result is never reached. Measured — adding one left the rejection count unchanged at 1.
        try {
          await router.push({ name: 'login', query: { redirect: route.fullPath } })
        } catch {
          /* stayed put */
        }
      })()
    }
  }

  return { isGated, gated }
}
