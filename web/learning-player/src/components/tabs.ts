/**
 * Shared ids for the tab↔panel pair (#1594 item 7). See {@link ../components/Tabs.vue}.
 *
 * These live in their own module rather than in `Tabs.vue` because a `<script setup>` block cannot
 * declare named exports, and PANELS need them too: a tab's `aria-controls` and its panel's
 * `aria-labelledby` have to agree exactly, and that is the pairing seven hand-written tab strips
 * all failed to establish. Generating both ends from one prefix means a typo produces a compile
 * error or a visibly wrong prefix, not a silently unlinked panel that still looks fine on screen.
 */

/** The tab button's id. */
export function tabId(prefix: string, key: string): string {
  return `${prefix}-tab-${key}`
}

/** The panel's id, referenced by its tab's `aria-controls`. */
export function panelId(prefix: string, key: string): string {
  return `${prefix}-panel-${key}`
}

export interface TabSpec<K extends string> {
  key: K
  /** Already translated — `Tabs.vue` does no i18n, callers own their strings. */
  label: string
  /** Optional `data-testid`, so existing specs keep their selectors. */
  testid?: string
  /**
   * Overrides the accessible name when the visible label is not the whole story — e.g. Search's
   * "Mine" scope, which for a signed-out visitor means "sign in to search yours".
   */
  ariaLabel?: string
}

/**
 * Everything a tabpanel needs to be linked back to its tab.
 *
 * `tabindex="0"` is part of the contract, not decoration: when a panel holds no focusable element
 * of its own, an arrow-key user who selects the tab has nowhere to go next. The panel itself has to
 * be reachable.
 */
export function panelAttrs(
  prefix: string,
  key: string,
): { id: string; role: 'tabpanel'; 'aria-labelledby': string; tabindex: string } {
  return {
    id: panelId(prefix, key),
    role: 'tabpanel',
    'aria-labelledby': tabId(prefix, key),
    tabindex: '0',
  }
}
