import { reactive } from 'vue'

/**
 * Keep the Library hub scannable at 100+ items (#2042 follow-up): each per-type section shows the
 * top {@link SECTION_CAP} by default and reveals more in place. A `force` flag lifts every cap at
 * once — used while a search query is active, so a match is never hidden behind a cap.
 *
 * ## Two modes, because "show all" is wrong past a point (operator 2026-09-18)
 *
 * - **expand-all** (default, `step` unset): one toggle swaps between the first `cap` and the whole
 *   list. Right for a section of six-ish where the full set is still one screen.
 * - **incremental** (`step` set): each press reveals another `step` items, so a long list is walked
 *   a page at a time rather than dumped. Downloads uses 5 + 5; the Saved episode sections and the
 *   highlights inside an episode use 10. "Show all" on a hundred captures is not a page, it is a
 *   scroll with no landmarks.
 *
 * Both share one state map, so a caller reads the same `visible` / `overflows` pair either way and
 * the section markup does not care which mode it is in.
 *
 * Presentation-only, per-view state (how far the user has walked a section); nothing persists.
 */
export const SECTION_CAP = 6

export function useCappedSections(cap: number = SECTION_CAP, step?: number) {
  /** Keys the user expanded (expand-all mode). */
  const expanded = reactive(new Set<string>())
  /** key -> how many are currently revealed (incremental mode). */
  const revealed = reactive(new Map<string, number>())

  const incremental = typeof step === 'number' && step > 0

  /** How many items this key currently shows, before `force` is considered. */
  function shown(key: string): number {
    if (!incremental) return expanded.has(key) ? Number.POSITIVE_INFINITY : cap
    return revealed.get(key) ?? cap
  }

  /**
   * Expand-all: flip between capped and whole.
   * Incremental: reveal one more `step`, and wrap back to `cap` once everything is out — the same
   * control does "Show more" then "Show less" rather than growing a second button.
   */
  function toggle(key: string, total?: number): void {
    if (!incremental) {
      if (expanded.has(key)) expanded.delete(key)
      else expanded.add(key)
      return
    }
    const current = revealed.get(key) ?? cap
    if (total != null && current >= total) revealed.set(key, cap)
    else revealed.set(key, current + (step as number))
  }

  /** The slice to render: everything when forced, else however far this key has been revealed. */
  function visible<T>(key: string, items: readonly T[], force = false): T[] {
    if (force) return [...items]
    const n = shown(key)
    return n === Number.POSITIVE_INFINITY ? [...items] : items.slice(0, n)
  }

  /**
   * Whether the reveal control is warranted.
   *
   * The key is optional so the expand-all callers that predate incremental mode keep working
   * unchanged; incremental callers must pass it, since "is there more" depends on how far THIS
   * section has been walked.
   */
  function overflows(len: number, force = false, key?: string): boolean {
    if (force) return false
    if (!incremental || key == null) return len > cap
    return len > (revealed.get(key) ?? cap) || (revealed.get(key) ?? cap) > cap
  }

  /** Items still hidden for this key — the number a "Show more (N)" label wants. */
  function remaining(key: string, len: number): number {
    const n = shown(key)
    return n === Number.POSITIVE_INFINITY ? 0 : Math.max(0, len - n)
  }

  return { expanded, revealed, toggle, visible, overflows, remaining, shown, cap, incremental }
}
