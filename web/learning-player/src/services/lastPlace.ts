/**
 * Reopen where you left off after iOS/Android ends the app in the background (#2278).
 *
 * A backgrounded app gets ended — the whole process, or just the WebView's content process — and
 * the next open used to start at Home with an empty mini-player. Sign-in survived (device token +
 * snapshot) and the playback position survived (server), but nothing knew which screen you were on
 * or which episode was loaded, so it felt like starting over.
 *
 * So: on every navigation and when the app goes to the background, one record goes to device
 * storage. On the first navigation of a native boot it is read back:
 *
 *   - the ROUTE is restored only when the boot landed on Home — the default start. A WebView reload
 *     re-opens the URL it was on, so it never lands on Home and is left alone;
 *   - the EPISODE is reloaded PAUSED at its position into an empty player, unless the restored route
 *     is that episode's own page (PlayerView loads it and resumes from the server itself).
 *
 * Only within RESTORE_WINDOW_MS and only for the SAME account: a day-old place is not "where you
 * left off", and another person's last screen must never open for you.
 */
import { START_LOCATION, type Router } from 'vue-router'
import type { NextUp } from '../stores/player'
import { getDeviceJson, setDeviceJson } from './deviceStore'
import { isNative } from './native'

export const LAST_PLACE_KEY = 'lastPlace.v1'
/** 12 hours (operator, 2026-10-04): past that, Home is the better start. */
export const RESTORE_WINDOW_MS = 12 * 60 * 60 * 1000

export type SavedEpisode = NextUp & { position: number }

export interface LastPlace {
  /** Full path of the last NON-public route. */
  path: string
  userId: string
  at: number
  episode: SavedEpisode | null
}

export type RestoreOutcome = 'route' | 'episode' | 'both' | 'none'

export interface RestorePlan {
  route: string | null
  episode: SavedEpisode | null
}

/**
 * What to restore, decided from plain values so it can be tested without a router or a device.
 *
 * `routeIsRestorable` is the caller's answer for `saved.path` (resolved route, not public, not
 * Home). `landedOnHome` is whether this boot's first navigation was the default start.
 */
export function planRestore(args: {
  saved: LastPlace | null
  now: number
  userId: string | null
  landedOnHome: boolean
  routeIsRestorable: boolean
  playerEmpty: boolean
}): RestorePlan {
  const { saved, now, userId, landedOnHome, routeIsRestorable, playerEmpty } = args
  const none: RestorePlan = { route: null, episode: null }
  if (!saved || !userId || saved.userId !== userId) return none
  const age = now - saved.at
  if (!(age >= 0 && age <= RESTORE_WINDOW_MS)) return none
  const route = landedOnHome && routeIsRestorable ? saved.path : null
  const episodePage = saved.episode ? `/episode/${encodeURIComponent(saved.episode.slug)}` : null
  const onItsOwnPage = route !== null && episodePage !== null && route.split('?')[0] === episodePage
  const episode = saved.episode && playerEmpty && !onItsOwnPage ? saved.episode : null
  return { route, episode }
}

export function outcomeOf(plan: RestorePlan): RestoreOutcome {
  if (plan.route && plan.episode) return 'both'
  if (plan.route) return 'route'
  if (plan.episode) return 'episode'
  return 'none'
}

let outcome: RestoreOutcome = 'none'
/** What the boot restored — reported once by `app_launch` (#2277). */
export function restoreOutcome(): RestoreOutcome {
  return outcome
}

interface Deps {
  userId: () => string | null
  nowPlaying: () => SavedEpisode | null
  loadAt: (episode: SavedEpisode, seconds: number) => void
  ensureAuthLoaded: () => Promise<void>
  /**
   * Make this account's DOWNLOADS visible before anything is restored: the player's local-source
   * resolver and the downloads registry. The restore runs on the first navigation, before the shell
   * mounts and wires either, so without this a restored episode page or player load cannot see an
   * episode sitting on the device and falls back to the origin URL — offline, "Couldn't load the
   * audio from the source" (`make test-ios` phase 2, 2026-10-04).
   */
  prepareLocalSources?: (userId: string) => Promise<void>
}

/** Write the current place. Public routes (landing, login, magic-link) are never recorded. */
export async function recordPlace(router: Router, deps: Deps): Promise<void> {
  try {
    const route = router.currentRoute.value
    const userId = deps.userId()
    if (!userId || route.meta.public || route === START_LOCATION) return
    const place: LastPlace = {
      path: route.fullPath,
      userId,
      at: Date.now(),
      episode: deps.nowPlaying(),
    }
    await setDeviceJson(LAST_PLACE_KEY, place)
  } catch {
    // Losing one write loses a restore, never the app.
  }
}

/**
 * Wire recording and the one-time restore. Native only: a browser restores its own tabs, and a web
 * reload keeps its URL.
 *
 * Must be installed BEFORE `app.use(router)`, which is what starts the initial navigation.
 */
export function installLastPlace(router: Router, deps: Deps): void {
  if (!isNative()) return
  let consumed = false
  router.beforeEach(async (to, from) => {
    if (consumed || from !== START_LOCATION) return true
    consumed = true
    try {
      await deps.ensureAuthLoaded()
      const saved = await getDeviceJson<LastPlace>(LAST_PLACE_KEY)
      const resolved = saved ? router.resolve(saved.path) : null
      const plan = planRestore({
        saved,
        now: Date.now(),
        userId: deps.userId(),
        landedOnHome: to.name === 'home',
        routeIsRestorable:
          !!resolved && resolved.matched.length > 0 && !resolved.meta.public && resolved.name !== 'home',
        playerEmpty: deps.nowPlaying() === null,
      })
      outcome = outcomeOf(plan)
      const userId = deps.userId()
      if ((plan.episode || plan.route) && userId) {
        // Before the page or the player looks for a local copy — see Deps.prepareLocalSources.
        await deps.prepareLocalSources?.(userId).catch(() => {})
      }
      if (plan.episode) deps.loadAt(plan.episode, plan.episode.position)
      if (plan.route) return plan.route
    } catch {
      // A failed restore is a normal start at Home.
    }
    return true
  })
  router.afterEach(() => {
    void recordPlace(router, deps)
  })
}
