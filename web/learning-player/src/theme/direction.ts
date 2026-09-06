/**
 * Visual-direction switch (#1949) — `?direction=paper`, and `?direction=` to go back.
 *
 * Sets `data-direction` on <html>, which is the only hook `theme/directions.css` needs. Nothing
 * else in the app reads it, no component branches on it, and with no query parameter the shipping
 * design renders exactly as before.
 *
 * The choice is persisted for the session so an in-app navigation does not silently drop back to
 * the default mid-review — comparing two directions is impossible if one of them keeps resetting.
 *
 * ## Why the empty value is a distinct case
 *
 * `?direction=` parses to `''`, not `null`, and `'' ?? stored` yields `''` — so the previous
 * implementation neither stored it nor applied it, and the stored direction survived untouched to
 * be restored on the next navigation. The switch could be turned on and moved between directions,
 * but never turned OFF: the only exit was clearing sessionStorage by hand, which is not a thing a
 * reviewer looking at a palette will think to do. A persistent toggle needs an off, and the empty
 * value is the obvious spelling of it.
 *
 * Lives here rather than inline in `main.ts` because behaviour worth three branches is behaviour
 * worth testing, and none of this was reachable from a test while it sat in module top-level code.
 */
export const DIRECTION_KEY = 'lp.direction'

/**
 * Resolve the direction for this page load from the URL and the stored session choice.
 *
 * Returns the direction to apply, or `null` for "the default palette". Writes the session store as
 * a side effect, because setting and clearing are the same user gesture with different values.
 */
export function resolveDirection(search: string, store: Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>): string | null {
  const q = new URLSearchParams(search).get('direction')

  // Present and empty: an explicit request for the default. Clear, do not fall through to storage.
  if (q === '') {
    store.removeItem(DIRECTION_KEY)
    return null
  }
  if (q) {
    store.setItem(DIRECTION_KEY, q)
    return q
  }
  // Absent: whatever the session already chose.
  return store.getItem(DIRECTION_KEY) || null
}

/** Apply the resolved direction to the document root, removing the attribute for the default. */
export function applyDirection(root: HTMLElement, direction: string | null): void {
  if (direction) root.dataset.direction = direction
  else delete root.dataset.direction
}
