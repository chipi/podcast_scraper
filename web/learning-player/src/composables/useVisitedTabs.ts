import { reactive, watch, type Ref } from 'vue'

/**
 * Tabs whose panel content has been opened at least once — mount a panel's CONTENT on first visit,
 * then keep it (the panel itself stays `v-show`).
 *
 * Every tabbed page mounted all of its panels up front and hid the inactive ones with `v-show`, so
 * opening Browse also built the Shows grid, and Library built Following, Saved, Boards and Revisit —
 * each fetching and decoding artwork nobody was looking at (measured on a Pixel 8 emulator against
 * prod, 2026-10-08: a hidden Shows panel held a 3000px image). Mounting on first visit keeps what
 * `v-show` was chosen for — no refetch and no lost scroll on switching back — without paying for
 * tabs never opened. The wrapper keeps rendering so a tab's `aria-controls` always resolves.
 */
export function useVisitedTabs<T extends string>(tab: Ref<T>): { has: (t: T) => boolean } {
  const seen = reactive(new Set<T>()) as Set<T>
  watch(tab, (t) => seen.add(t), { immediate: true })
  return { has: (t: T) => seen.has(t) }
}
