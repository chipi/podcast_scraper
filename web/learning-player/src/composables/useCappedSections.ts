import { reactive } from 'vue'

/**
 * Keep the Library hub scannable at 100+ items (#2042 follow-up): each per-type section shows the
 * top {@link SECTION_CAP} by default and expands in place ("Show all (N)"). A `force` flag lets a
 * caller lift every cap at once — used while a search query is active, so a match is never hidden
 * behind a cap.
 *
 * Presentation-only, per-view state (which sections the user expanded); nothing here persists.
 */
export const SECTION_CAP = 6

export function useCappedSections(cap: number = SECTION_CAP) {
  const expanded = reactive(new Set<string>())

  function toggle(key: string): void {
    if (expanded.has(key)) expanded.delete(key)
    else expanded.add(key)
  }

  /** The slice to render: the whole list when expanded or forced, else the first `cap`. */
  function visible<T>(key: string, items: readonly T[], force = false): T[] {
    return force || expanded.has(key) ? [...items] : items.slice(0, cap)
  }

  /** Whether a "Show all / Show less" toggle is warranted (more than `cap`, and not forced open). */
  function overflows(len: number, force = false): boolean {
    return !force && len > cap
  }

  return { expanded, toggle, visible, overflows, cap }
}
