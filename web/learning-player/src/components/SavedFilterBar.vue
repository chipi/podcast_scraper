<script setup lang="ts">
/**
 * The Saved tab's filter bar (RFC-121 ph. 3 / #2042) — lifted to the top of Saved so it governs
 * every section at once, instead of a colour filter buried inside the Highlights list.
 *
 * Three controls: **type** chips (which saved kinds to show — multi-select, none selected = all),
 * an always-on **colour** swatch strip (the whole palette, not only the colours in use), and
 * a **sort** select. Type chips render only for kinds present and colour swatches only for colours
 * in use, so the bar never offers a filter that would empty the list — same presence rule as the
 * sections themselves (#1962 single-empty-state).
 *
 * Presentational — two-way binds via v-model; the parent owns the state and applies it.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { HIGHLIGHT_COLORS } from '../utils/highlightColors'
import BellOffIcon from './BellOffIcon.vue'
import TypeFilterBar from './TypeFilterBar.vue'

const types = defineModel<string[]>('types', { default: () => [] })
const color = defineModel<string | null>('color', { default: null })
const sort = defineModel<string>('sort', { default: 'recent' })
const search = defineModel<string>('search', { default: '' })
/** Muted filter: `false` shows everything (default), `true` narrows to retired captures only. */
const mutedOnly = defineModel<boolean>('mutedOnly', { default: false })

const props = withDefaults(
  defineProps<{
    /** The saved kinds that actually have items, in display order. */
    availableTypes: { key: string; label: string }[]
    /**
     * Whether the colour strip renders at all.
     *
     * Following uses this same bar but nothing there CAN carry a colour — colour is a property of a
     * saved item. It used to opt out by passing an empty `colorsPresent`, which worked only while
     * the strip was data-driven; now that the palette is always shown, opting out has to be said
     * rather than implied.
     */
    showColors?: boolean
    /**
     * Whether the muted toggle renders. Opt-IN because only highlights can be muted — Following has
     * none, so the control would be inert there.
     *
     * On Saved it is ALWAYS on. It was additionally gated on the user already having muted
     * something, which I added unasked and then described as "by design" when the operator could
     * not find it (2026-09-18). A filter nobody can see is a feature nobody can use, and one that
     * appears only after you have used it elsewhere is worse than one that is simply there.
     */
    showMuted?: boolean
    /** Sort options for the select; defaults to the shared Recent / A–Z set used by both tabs. */
    sortOptions?: { value: string; label: string }[]
    /** Search-box placeholder (a type-to-filter over every section). */
    searchPlaceholder?: string
  }>(),
  { showColors: true, showMuted: false }
)

const { t } = useI18n()

/**
 * The WHOLE palette, always — not only the colours currently in use (operator 2026-09-17).
 *
 * Rendering just the present colours meant the control's size changed with the data: one saved amber
 * highlight produced a single lone dot, which reads as a broken or disabled control rather than a
 * colour filter. The full strip shows what the feature IS, and a colour with nothing behind it
 * simply filters to empty — which is a legible answer, not a dead end.
 */
const colorOptions = HIGHLIGHT_COLORS
// One sort model across Following AND Saved (#2042): Recent (default) or A–Z. Per-episode grouping
// of highlights is structural and unaffected — sort only orders the groups + the flat lists.
const resolvedSortOptions = computed(
  () =>
    props.sortOptions ?? [
      { value: 'recent', label: t('library.sortRecent') },
      { value: 'title', label: t('library.sortAz') },
    ],
)

function pickColor(token: string): void {
  color.value = color.value === token ? null : token
}
const hasFilters = computed(
  () =>
    types.value.length > 0 ||
    color.value !== null ||
    mutedOnly.value ||
    search.value.trim() !== '',
)
function clearAll(): void {
  types.value = []
  color.value = null
  mutedOnly.value = false
  search.value = ''
}
</script>

<template>
  <div v-if="availableTypes.length" class="mb-5 flex flex-col gap-3" data-testid="saved-filter-bar">
    <!-- Search + sort share ONE row (operator 2026-09-17). The search was full-width with sort
         stranded on the row below, which spent a whole line on a control two words wide. The field
         only needs enough room to read a query back, so it flexes and the select takes its natural
         width beside it.
         Type-to-filter search across every section — the primary find tool at 100+ items (#2042).
         A match is never hidden behind a section cap: the parent lifts caps while this is non-empty. -->
    <div class="flex items-center gap-2">
      <input
        v-model="search"
        type="search"
        :placeholder="searchPlaceholder ?? t('library.searchSaved')"
        :aria-label="searchPlaceholder ?? t('library.searchSaved')"
        data-testid="saved-search"
        class="lp-search min-w-0 flex-1 rounded-full border border-border bg-surface px-4 py-2 text-sm text-canvas-foreground outline-none focus:border-accent"
      />
      <select
        v-model="sort"
        :aria-label="t('library.savedSort')"
        data-testid="saved-sort"
        class="shrink-0 rounded-full border border-border bg-surface px-3 py-1.5 text-xs font-semibold text-canvas-foreground outline-none focus:border-accent"
      >
        <option v-for="o in resolvedSortOptions" :key="o.value" :value="o.value">{{ o.label }}</option>
      </select>
    </div>
    <!-- Type chips: none selected = All (shared TypeFilterBar — same strip as Search and Boards). -->
    <TypeFilterBar v-model="types" :options="availableTypes" testid-prefix="saved-type" />

    <!-- Colour swatches + sort on ONE row (operator): the "Colour"/"Sort" text labels are dropped —
         the swatches read as colours and the select shows its value — so the swatches (44px targets)
         and the sort select fit a single line instead of wrapping to two. Sort is pushed right.

         GENUINELY one row now (operator 2026-09-19). It said "one row" and then wrapped anyway:
         `flex-wrap` on both this container and the swatch group meant the six 44px targets filled
         the width on a phone and the mute bell dropped onto a line of its own, reading as a second,
         unrelated control. The swatches scroll horizontally instead; mute and clear are pinned
         right behind a hairline separator, which is what marks them as a different question from
         "which colour" rather than position alone doing that job.

         AND THEN IT FIT ON NO PHONE (operator 2026-09-27: "the colours are too spaced and there's
         like some invisible scroll"). One row that scrolls is not one row — it is a row with a
         hidden half, and the scrollbar is suppressed here, so nothing on screen says the sixth
         colour exists. Measured off the device screenshot at 393pt: the content box is 324.6pt, the
         strip was clipped at 263.6pt, and six `w-11` targets with `gap-1` need 284pt — 20.4pt over,
         which is why violet was cut exactly in half.

         The gaps were the whole overspend, and removing them is also what the operator asked for
         visually: a 16pt dot centred in a 44pt target already shows 28pt of space, so the `gap-1`
         was padding padding. The budget now reads 6x44 + 4 + 1 + 4 + 44 = 317pt in 324.6pt. `ml-auto`
         below absorbs the 7.6pt of slack into the one place it belongs — between the colours and the
         thing that is not a colour.

         `overflow-x-auto` STAYS, on purpose. It is inert at this width and every wider one; it is
         the fallback for a phone narrower than the operator's (a 375pt one is ~10pt short), where
         scrolling to a swatch beats clipping it with no way to reach it at all. If the strip ever
         needs a seventh colour, this arithmetic is what has to be redone — see the guard in
         `__checks__/touch-affordances.test.ts`, which fails on a re-added gap. -->
    <div class="flex items-center gap-1">
      <div
        v-if="showColors"
        class="flex min-w-0 items-center overflow-x-auto pb-1 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
        role="group"
        :aria-label="t('library.savedFilterColor')"
      >
        <!-- "Any colour" leads the strip as an EMPTY ring (operator 2026-09-17), so clearing a
             colour is the same gesture in the same place as choosing one. Previously the only way
             back was the separate "Clear" link, which also dropped the type and search filters —
             one control undoing three things the user did not ask to undo. Mirrors the "All" chip
             that leads the type row. -->
        <button
          type="button"
          data-testid="saved-filter-swatch-any"
          class="relative flex h-11 w-11 shrink-0 items-center justify-center rounded-full transition"
          :aria-pressed="color === null"
          :aria-label="t('library.savedFilterColorAny')"
          :title="t('library.savedFilterColorAny')"
          @click="color = null"
        >
          <span
            class="h-4 w-4 rounded-full border border-border ring-offset-1 ring-offset-canvas transition"
            :class="color === null ? 'ring-2 ring-accent' : 'opacity-70'"
          />
          <!-- FOURTH instance, and the one nobody had noticed: found by the class-level guard in
               `__checks__/accessible-names.test.ts` rather than by a device run (2026-09-26). Same
               shape as its siblings — `aria-label` over a subtree with no text — so on Android this
               reset control is announced as an unnamed "Button", and a TalkBack user has no way to
               know it clears the colour filter. -->
          <span class="sr-only left-0 top-0">{{ t('library.savedFilterColorAny') }}</span>
        </button>
        <button
          v-for="c in colorOptions"
          :key="c.token"
          type="button"
          data-testid="saved-filter-swatch"
          class="relative flex h-11 w-11 shrink-0 items-center justify-center rounded-full transition"
          :aria-pressed="color === c.token"
          :aria-label="t('library.savedFilterColorOnly', { color: t(c.labelKey) })"
          :title="t(c.labelKey)"
          @click="pickColor(c.token)"
        >
          <span
            class="h-4 w-4 rounded-full ring-offset-1 ring-offset-canvas transition"
            :class="[c.swatch, color === c.token ? 'ring-2 ring-accent' : 'opacity-70']"
          />
          <!-- THIRD instance of one defect (2026-09-26): an `aria-label` on a button whose subtree
               carries no text is dropped by Android System WebView, so the control is announced as
               an unnamed Button. Already fixed on `SavedColorControl`'s trigger (2026-09-24) and on
               its swatches (same day as this); these filter swatches were the one left.
               It cost a wrong diagnosis: `AppJourneyTests.test07` failed with "the Saved colour
               filter offers 0 colour(s) — the seed coloured too few items", blaming the seed. The
               seed was fine and the colours WERE applied — these buttons simply had no name to
               find.
               The text matches `aria-label` exactly: Android reads `getText()` before
               `getContentDescription()`, so a shorter string here would shadow the label and the
               two would drift. Anchored, hence `relative` on the button. -->
          <span class="sr-only left-0 top-0">{{
            t('library.savedFilterColorOnly', { color: t(c.labelKey) })
          }}</span>
        </button>
      </div>

      <!-- Muted: one toggle, sitting with the colour swatches because it reads the same way — off
           shows everything, on narrows. The icon is the bell-with-slash the row itself uses when a
           capture is muted, so the filter and the thing it filters carry one glyph.

           A toggle rather than an any/muted/active triple like the colours: "everything except the
           muted" is the default minus a handful, and the operator asked for one icon. If reviewing
           the ACTIVE set alone turns out to matter, this becomes a three-state control. -->
      <div class="ml-auto flex shrink-0 items-center gap-1">
        <!-- A hairline, not a gap: "which colour" and "muted or not" are different questions, and
             spacing alone did not say so once they shared a row. It is doing more work at `gap-1`
             than it was at `gap-2`, which is the point of keeping it while the gaps came down. -->
        <span
          v-if="showColors && showMuted"
          aria-hidden="true"
          class="h-6 w-px shrink-0 bg-border"
        />
      <button
        v-if="showMuted"
        type="button"
        class="relative flex h-11 w-11 shrink-0 items-center justify-center rounded-full transition"
        data-testid="saved-filter-muted"
        :aria-pressed="mutedOnly"
        :aria-label="t('library.savedFilterMuted')"
        :title="t('library.savedFilterMuted')"
        @click="mutedOnly = !mutedOnly"
      >
        <span
          class="flex h-7 w-7 items-center justify-center rounded-full border transition"
          :class="mutedOnly ? 'border-accent text-accent' : 'border-border text-muted'"
        ><BellOffIcon :size="14" /></span>
        <!-- FIFTH instance, also found by the guard rather than by a device run: an icon component
             is not a text node, so `aria-label` over `<BellOffIcon>` alone leaves this announced as
             an unnamed "Button" on Android. -->
        <span class="sr-only left-0 top-0">{{ t('library.savedFilterMuted') }}</span>
      </button>

      <button
        v-if="hasFilters"
        type="button"
        class="shrink-0 text-xs font-semibold text-accent"
        data-testid="saved-filter-clear"
        @click="clearAll"
      >{{ t('library.savedFilterClear') }}</button>
      </div>
    </div>
  </div>
</template>
