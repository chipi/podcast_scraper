<script setup lang="ts" generic="K extends string">
/**
 * The one tab strip (#1594 item 7).
 *
 * ## What was wrong
 *
 * The app had SEVEN tab implementations and not one of them was complete:
 *
 * | where | roles | tab↔panel linkage | arrow keys |
 * |---|---|---|---|
 * | `LibraryView` | none at all | — | — |
 * | `SearchView`, `ProfileView`, `EntityCardBody`, `TrendWindowTabs` | yes | no | no |
 * | `BrowseView`, `HomeView` | yes, incl. `role="tabpanel"` | no `aria-controls` / `aria-labelledby` | no |
 *
 * Seven copies is seven chances to miss a rule, and every one of them missed the same two. A
 * screen-reader user reached a strip that announced "tab" without saying what it controlled, and a
 * keyboard user could not move between tabs the way the platform has worked since forever.
 *
 * ## The two rules everyone forgot, and why they are easy to forget
 *
 * **Roving tabindex.** In a tablist, exactly ONE tab is in the page's tab order; the arrow keys move
 * between them. Everybody writes plain buttons instead, which is *usable* — you can Tab to each one
 * — so nothing looks broken. It just means a five-tab strip costs five Tab presses to traverse
 * instead of one, and the arrow keys, which is what a screen-reader user will actually reach for,
 * do nothing at all. Invisible in review, invisible in a mouse test.
 *
 * **`aria-controls` / `aria-labelledby`.** Without the pair, "tab" and "tabpanel" are two unrelated
 * announcements; with it, the panel is named by its tab and the tab says what it opens. This needs
 * matching ids on both sides, which is exactly the kind of bookkeeping that gets dropped when the
 * markup is copy-pasted. So the ids are generated here, from `idPrefix`, by {@link tabId} and
 * {@link panelId} — panels import the same helpers, and a typo cannot silently produce a mismatch.
 *
 * ## Why `variant` instead of one look
 *
 * Three shapes already exist — an underline strip, `.lp-segment`, and a pill group — and they are
 * deliberate: the underline is a page-level section switcher, the pill is a compact in-card
 * control. Item 7 asks for one COMPONENT, not one appearance; unifying the visuals would be a
 * redesign nobody asked for, and it would have made this change impossible to review against the
 * behaviour it is actually fixing.
 */
import { computed, ref } from 'vue'
import { panelId, tabId, type TabSpec } from './tabs'

const props = withDefaults(
  defineProps<{
    tabs: ReadonlyArray<TabSpec<K>>
    /** Names the tablist for assistive tech. Required — an unnamed tablist is a bare "tab list". */
    label: string
    /** Must be unique on the page: it namespaces the generated tab and panel ids. */
    idPrefix: string
    variant?: 'underline' | 'segment' | 'pill'
    /**
     * `tabs` for a strip that switches a PANEL; `radio` for one that sets a parameter.
     *
     * The trend-window selector and the Your-Week layout preference were both marked up as
     * tablists, and neither controls a panel: the window re-queries a rail the parent owns, the
     * layout is a saved preference with no region at all. A `role="tab"` whose `aria-controls`
     * names nothing is a dangling promise — worse than the missing linkage it would replace — and
     * "tab" is the wrong announcement for "set this to one of four values" either way.
     *
     * The KEYBOARD contract is identical (roving tabindex, arrows move and select), which is why
     * both live here rather than in a second component that would drift from this one.
     */
    pattern?: 'tabs' | 'radio'
    /** `true` stretches each tab to an equal share of the row (LibraryView's five-up strip). */
    equalWidth?: boolean
  }>(),
  { variant: 'underline', equalWidth: false, pattern: 'tabs' },
)

const model = defineModel<K>({ required: true })

const listEl = ref<HTMLElement | null>(null)

const listClass = computed(() => {
  if (props.variant === 'segment') return 'lp-segment'
  if (props.variant === 'pill')
    return 'inline-flex gap-1 rounded-full border border-border bg-surface p-1'
  return 'flex flex-wrap gap-1 border-b border-border'
})

function tabClass(key: K): string {
  const on = model.value === key
  if (props.variant === 'segment') return 'lp-segment-option'
  if (props.variant === 'pill')
    return [
      'rounded-full px-3 py-1 text-xs font-bold transition',
      on ? 'bg-accent text-accent-foreground' : 'text-muted hover:text-canvas-foreground',
    ].join(' ')
  return [
    '-mb-px shrink-0 whitespace-nowrap border-b-2 py-2 text-sm font-bold transition',
    props.equalWidth ? 'flex-1 px-1 text-center text-xs' : 'px-3',
    on
      ? 'border-accent text-canvas-foreground'
      : 'border-transparent text-muted hover:text-canvas-foreground',
  ].join(' ')
}

/**
 * Arrow-key movement, per the WAI-ARIA tabs pattern.
 *
 * Selection FOLLOWS focus, which is the right default for these strips: every panel here is already
 * rendered (`v-show`), so moving the selection costs nothing and matches what a mouse user sees. It
 * would be the wrong default for tabs whose panels are expensive to load.
 *
 * Wraps at both ends, and Home/End jump to the extremes.
 */
function onKeydown(e: KeyboardEvent): void {
  const keys = props.tabs.map((t) => t.key)
  const i = keys.indexOf(model.value)
  if (i < 0) return

  let next: number | null = null
  if (e.key === 'ArrowRight' || e.key === 'ArrowDown') next = (i + 1) % keys.length
  else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') next = (i - 1 + keys.length) % keys.length
  else if (e.key === 'Home') next = 0
  else if (e.key === 'End') next = keys.length - 1
  if (next === null) return

  e.preventDefault()
  model.value = keys[next]
  // Focus has to move with the selection, or the roving tabindex leaves the user's focus on a tab
  // that is no longer the selected one — the next arrow press then starts from the wrong place.
  // BOTH roles. This queried `[role="tab"]` only, so in the four radio strips (search scope, the
  // card's corpus scope, the trend window, the Your Week layout) the arrows moved the SELECTION
  // while focus stayed put — stranded on a button whose tabindex had just become -1, which is
  // exactly the failure the note above says this line prevents. Silent: selection-follows-focus
  // made it look like it worked unless you were actually holding a keyboard.
  const buttons = listEl.value?.querySelectorAll<HTMLElement>('[role="tab"], [role="radio"]')
  buttons?.[next]?.focus()
}
</script>

<template>
  <div
    ref="listEl"
    :role="pattern === 'radio' ? 'radiogroup' : 'tablist'"
    :aria-label="label"
    :class="listClass"
    @keydown="onKeydown"
  >
    <button
      v-for="t in tabs"
      :key="t.key"
      type="button"
      :role="pattern === 'radio' ? 'radio' : 'tab'"
      :id="tabId(idPrefix, t.key)"
      :aria-selected="pattern === 'radio' ? undefined : model === t.key"
      :aria-checked="pattern === 'radio' ? model === t.key : undefined"
      :aria-label="t.ariaLabel"
      :aria-controls="pattern === 'radio' ? undefined : panelId(idPrefix, t.key)"
      :tabindex="model === t.key ? 0 : -1"
      :data-testid="t.testid"
      :class="tabClass(t.key)"
      @click="model = t.key"
    >
      {{ t.label }}
    </button>
  </div>
</template>
