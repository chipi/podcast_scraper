import { computed, ref } from 'vue'
import { readCached, writeCached } from '../services/contentCache'

/** What a data-backed section is currently doing. */
export type SectionPhase = 'loading' | 'ready' | 'error'

/**
 * Load state for a Home section (#1591), with the per-account content cache wired in (#1909).
 *
 * ## The bug this exists to fix
 *
 * Every section swallowed failure into emptiness — `.catch(() => [])` — and then hid itself when
 * empty. So a cold corpus, a brand-new account, and **a total API outage all rendered the same
 * page**: a hero, a search box, and two chips. Nothing distinguished "nothing yet" from "loading"
 * from "broken", for the user or for us.
 *
 * ## This reverses a deliberate decision, on purpose
 *
 * Self-hiding was a documented design goal (`HomeView.vue` docblock) and is asserted by
 * `e2e/your-week.spec.ts`. The replacement contract:
 *
 * - **loading** → a skeleton, so the section's existence is visible before its content is
 * - **ready + empty** → hide, UNLESS the emptiness is *actionable* by the user. A section empty
 *   because the system has nothing (no corpus activity, no history) offers no move and should stay
 *   hidden; a section empty because the user hasn't done something yet should render and carry that
 *   action. See the "Your shows" empty state for the first instance.
 * - **error** → say so, and offer retry. Never silently equal to empty.
 *
 * Rule of thumb: hide when the SYSTEM is empty, render when the USER is.
 *
 * ## Why the cache lives HERE (#1909)
 *
 * #1909 scoped "snapshot-on-successful-load + hydrate-then-revalidate for library, queue, favourites
 * **and the Home rails**". The first three landed in their stores; the Home rails did not, so with
 * no network Home rendered a column of identical "Couldn't load this right now" cards — five ways of
 * saying one thing, none of them the content the user had already loaded. The requirement was
 * "everything I loaded last time is still there, just stale", and an error card is the opposite of
 * that.
 *
 * Doing it per-section, by hand, in eight places is how the first four keys drifted apart. A section
 * opts in with a `cacheKey` and gets the whole contract:
 *
 * - **Hydrate before the fetch answers.** Last-known content paints immediately; the request is
 *   already in flight, so caching costs the network nothing.
 * - **A failure NEVER replaces content.** This is the arc's governing rule ("only a 401/403 may
 *   destroy cached state; a transport error never may") applied to the render, not just to stores.
 *   A section with something to show reports `stale`; only a section with NOTHING reports `error`.
 * - **Revalidate in place.** A reload with data already on screen does not drop back to `loading`,
 *   so returning to Home cannot flicker a skeleton over content that is already correct.
 */

/**
 * Which cached sections are currently showing stale data. Module-scope because the notice that
 * says so belongs at the top of the PAGE, not inside each rail — eight rails each announcing their
 * own staleness is the same wall of repeated text this change exists to remove.
 */
const staleKeys = ref<ReadonlySet<string>>(new Set())

function markStale(key: string, stale: boolean): void {
  const has = staleKeys.value.has(key)
  if (has === stale) return
  const next = new Set(staleKeys.value)
  if (stale) next.add(key)
  else next.delete(key)
  staleKeys.value = next
}

/** True when at least one cached section is showing what it had last time rather than fresh data. */
export const anyStale = computed(() => staleKeys.value.size > 0)

/** Forget the staleness tally — sign-out, or a namespace change. */
export function resetStaleness(): void {
  staleKeys.value = new Set()
}

export function useSectionState<T>(initial: T, options: { cacheKey?: string } = {}) {
  const data = ref<T>(initial)
  const phase = ref<SectionPhase>('loading')
  /** Showing content that a fetch did not confirm this session. */
  const stale = ref(false)
  /** Whether anything worth keeping is on screen — the thing a failure must not destroy. */
  const hasData = ref(false)
  let hydrated = false

  function accept(value: T, isStale: boolean): void {
    data.value = value
    hasData.value = true
    phase.value = 'ready'
    stale.value = isStale
    if (options.cacheKey) markStale(options.cacheKey, isStale)
  }

  /**
   * Run a fetch and record what actually happened.
   *
   * Deliberately does NOT swallow into `initial` — that collapse is the defect. A rejected promise
   * lands in `error`, which is a different render from an empty success — unless there is cached or
   * already-loaded content, in which case the content stays and only its freshness changes.
   */
  async function load(fetcher: () => Promise<T>): Promise<void> {
    // Start the request BEFORE reading the cache, so hydration never delays the network. Settled
    // into a result object rather than left to reject: the cache read below is a real async gap,
    // and a promise that rejects across it with no handler attached is an unhandled rejection.
    const inflight = fetcher().then(
      (value) => ({ ok: true, value }) as const,
      (error: unknown) => ({ ok: false, error }) as const,
    )
    // A revalidation with content on screen must not drop back to a skeleton (#1909: "revalidate in
    // place, never wipe").
    if (!hasData.value) phase.value = 'loading'
    if (options.cacheKey && !hydrated) {
      hydrated = true
      // RACED against the request, not awaited in front of it. The snapshot exists to fill the wait
      // — so when there is no wait it must not create one. Reading it first put device storage on
      // the critical path of every section, and on web that read spans a lazy module load, so a
      // request that had already failed still sat behind it.
      const first = await Promise.race([
        readCached<T>(options.cacheKey).then((value) => ({ from: 'cache', value }) as const),
        inflight.then(() => ({ from: 'network', value: null }) as const),
      ])
      // Fresh always beats stale: if the request got there first, the snapshot is already obsolete
      // and painting it would be a flicker backwards.
      if (first.from === 'cache' && first.value !== null) accept(first.value, true)
    }
    const result = await inflight
    if (result.ok) {
      accept(result.value, false)
      if (options.cacheKey) void writeCached(options.cacheKey, result.value)
      return
    }
    if (hasData.value) {
      // The rule: a transport error may not destroy what the user already has. Say it is stale.
      stale.value = true
      phase.value = 'ready'
      if (options.cacheKey) markStale(options.cacheKey, true)
      return
    }
    phase.value = 'error'
  }

  const isLoading = computed(() => phase.value === 'loading')
  const isError = computed(() => phase.value === 'error')
  const isReady = computed(() => phase.value === 'ready')

  return { data, phase, load, isLoading, isError, isReady, stale }
}
