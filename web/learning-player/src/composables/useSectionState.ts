import { computed, ref } from 'vue'

/** What a data-backed section is currently doing. */
export type SectionPhase = 'loading' | 'ready' | 'error'

/**
 * Load state for a Home section (#1591).
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
 */
export function useSectionState<T>(initial: T) {
  const data = ref<T>(initial)
  const phase = ref<SectionPhase>('loading')

  /**
   * Run a fetch and record what actually happened.
   *
   * Deliberately does NOT swallow into `initial` — that collapse is the defect. A rejected promise
   * lands in `error`, which is a different render from an empty success.
   */
  async function load(fetcher: () => Promise<T>): Promise<void> {
    phase.value = 'loading'
    try {
      data.value = await fetcher()
      phase.value = 'ready'
    } catch {
      phase.value = 'error'
    }
  }

  const isLoading = computed(() => phase.value === 'loading')
  const isError = computed(() => phase.value === 'error')
  const isReady = computed(() => phase.value === 'ready')

  return { data, phase, load, isLoading, isError, isReady }
}
