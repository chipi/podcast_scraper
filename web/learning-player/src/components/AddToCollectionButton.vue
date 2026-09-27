<script setup lang="ts">
/**
 * Add-to-collection control (RFC-119) — a compact icon button that pins ANY typed item (episode /
 * show / search / topic / person / link / highlight) into one of the user's collections. Opens a
 * small menu of collections (loaded on first open) with an inline "new collection" create. Sign-in
 * gated, like the queue / favourite controls. Reusable across every surface that pins.
 */
import { ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import {
  addToCollection,
  createCollection,
  getCollections,
  getCollectionsContaining,
} from '../services/api'
import { enqueue, isPermanent } from '../services/outbox'
import type { Collection, CollectionItemRef } from '../services/types'
import { useSignInGate } from '../composables/useSignInGate'
import { useAnchoredMenu } from '../composables/useAnchoredMenu'

const props = withDefaults(
  defineProps<{
    item: CollectionItemRef
    /**
     * `icon` — compact round icon, for dense cards/rails (default). `pill` — a labelled pill
     * (`+ Collection`) for roomy detail/player surfaces, matching the Follow pill idiom (CO.1).
     * `menuitem` — a full-width row inside a ⋯ overflow (the list/grid card collapses download +
     * collect behind ⋯ so four controls don't wrap the artwork-width column, operator 2026-09-13).
     */
    variant?: 'icon' | 'pill' | 'menuitem'
  }>(),
  { variant: 'icon' },
)
const { t } = useI18n()
const { isGated, gated } = useSignInGate()

const collections = ref<Collection[]>([])
/** Collection ids holding `props.item`. Empty AND `membershipKnown=false` means "we could not look". */
const holdingIds = ref<Set<string>>(new Set())
const membershipKnown = ref(false)
const holds = (id: string): boolean => membershipKnown.value && holdingIds.value.has(id)
const loaded = ref(false)
const newName = ref('')
const addedTo = ref<string | null>(null)

/**
 * The last failure, shown in the panel (#2004 item 13).
 *
 * Every call here used to end in `.catch(() => null)` with an `if (result)` guard, so a failed add
 * changed NOTHING on screen: no message, no spinner, no closed panel. A user tapping a failing
 * backend sees an unchanged screen and concludes the tap missed — the same failure mode as the
 * capture bug in #1592, where the only signal of failure was the absence of a change.
 *
 * A write that did not happen must say so.
 */
const error = ref<string | null>(null)

/**
 * Positioning + dismissal come from the shared popover shell now — teleported, `position: fixed`,
 * clamped on screen by `anchorPanel`. This replaces the bespoke `right-0`/`left-0` flip: the panel
 * is 224px wide and hard-anchoring it `right-0` ran it off the LEFT edge wherever the trigger sat
 * near the left margin (the entity card, a card control under the artwork). One rule for every menu
 * now (operator 2026-09-13). Teleporting also dissolves the sibling-card z-index fight the old
 * `absolute` panel had — no wrapper z-index hack needed.
 */
const triggerEl = ref<HTMLElement | null>(null)
const panelEl = ref<HTMLElement | null>(null)
const { open, toggle, close, teleportTarget } = useAnchoredMenu(triggerEl, panelEl, { align: 'end' })

// Load collections on first open; clear the transient "added" receipt whenever it closes.
watch(open, async (isOpen) => {
  if (!isOpen) {
    addedTo.value = null
    return
  }
  // Refetch on EVERY open, keeping the current list visible while it runs. The old
  // `if (loaded.value) return` early-out LATCHED whatever the first open produced: a single
  // transient empty (a flaky native fetch) then stayed empty on every reopen until the card
  // remounted — the "second time I open collections it's empty, I have to go to another topic to
  // reset" bug. Now a good list survives a later transient failure, and an empty one self-heals on
  // the next open.
  error.value = null
  try {
    // Two calls, deliberately: the boards, and — separately — which of them already hold this
    // item. Membership is a question about the ITEM, so it is not a field on a collection.
    const [rows, membership] = await Promise.all([
      getCollections(),
      getCollectionsContaining(props.item).catch(() => ({ ids: [] as string[], checked: false })),
    ])
    collections.value = rows
    // `checked` false = we could not look. Keep the set EMPTY and remember we did not know, so the
    // rows stay unmarked rather than confidently claiming the item is saved nowhere.
    holdingIds.value = membership.checked ? new Set(membership.ids) : new Set()
    membershipKnown.value = membership.checked
    loaded.value = true
  } catch {
    // Only surface empty + error when we have NOTHING to show; never blank a list we already have.
    if (!loaded.value) {
      collections.value = []
      error.value = t('collections.loadFailed')
    }
  }
})

const onClick = gated(toggle)

async function pick(id: string): Promise<void> {
  error.value = null
  let updated: Collection
  try {
    updated = await addToCollection(id, props.item)
  } catch (err) {
    /**
     * Queue a TRANSIENT failure instead of losing it (#2004 item 13).
     *
     * Every other per-user write already did this — favourites, queue, highlights, notes, follows.
     * Collections was the only one that dropped the write on the floor, which is why a flaky moment
     * was invisible everywhere else and permanent here. Only a REFUSAL discards, same rule as
     * `stores/capture.ts`: a 502 or a dead socket is not an answer.
     */
    if (isPermanent(err)) {
      error.value = t('collections.addFailed')
      return
    }
    enqueue({ op: 'collection.addItem', collectionId: id, item: props.item })
    addedTo.value = id
    window.setTimeout(() => close(false), 800)
    return
  }
  holdingIds.value = new Set(holdingIds.value).add(id)
  membershipKnown.value = true
  const i = collections.value.findIndex((c) => c.id === updated.id)
  if (i >= 0) collections.value[i] = updated
  addedTo.value = id
  window.setTimeout(() => {
    open.value = false
    addedTo.value = null
  }, 800)
}

async function createAndAdd(): Promise<void> {
  const name = newName.value.trim()
  if (!name) return
  error.value = null
  let created: Collection
  try {
    created = await createCollection(name)
  } catch (err) {
    // The name stays in the input on a REFUSAL — retyping it would be the app's mistake, not theirs.
    if (isPermanent(err)) {
      error.value = t('collections.createFailed')
      return
    }
    // Transient: queue the create AND the pin that follows it, so the item the user was adding is
    // not lost offline (#2004 #5). The create carries a client-minted id; the server honours it on
    // replay, so the queued addItem targeting that same id lands (it no longer 404s).
    const clientId = `col_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`
    enqueue({ op: 'collection.create', name, clientId })
    enqueue({ op: 'collection.addItem', collectionId: clientId, item: props.item })
    collections.value = [{ id: clientId, name, created_at: Date.now() / 1000, count: 1 }, ...collections.value]
    newName.value = ''
    addedTo.value = clientId
    window.setTimeout(() => close(false), 800)
    return
  }
  collections.value = [created, ...collections.value]
  newName.value = ''
  await pick(created.id)
}
</script>

<template>
  <!-- The menu is teleported to <body> (shared shell), so it no longer overflows into the card below
       and there is no sibling-card stacking fight to out-rank — the wrapper needs no z-index hack. -->
  <div class="relative" :class="variant === 'menuitem' ? 'block' : 'inline-flex'">
    <button
      ref="triggerEl"
      type="button"
      :class="
        variant === 'pill'
          ? 'lp-tap inline-flex items-center gap-1 rounded-full bg-overlay px-3 py-1 text-xs font-bold text-canvas-foreground transition hover:bg-elevated'
          : variant === 'menuitem'
            ? 'flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm text-canvas-foreground transition hover:bg-overlay'
            : 'lp-tap flex h-8 w-8 items-center justify-center rounded-full border border-border text-muted transition hover:text-canvas-foreground'
      "
      :data-menuitem="variant === 'menuitem' ? '' : undefined"
      :role="variant === 'menuitem' ? 'menuitem' : undefined"
      :aria-label="isGated ? t('auth.signInToSave') : t('collections.addTo')"
      :title="variant === 'menuitem' ? undefined : t('collections.addTo')"
      aria-haspopup="true"
      :aria-expanded="open"
      data-testid="add-to-collection"
      @click.stop.prevent="onClick"
    >
      <template v-if="variant === 'pill'">
        <span aria-hidden="true">+</span>
        {{ t('collections.pill') }}
      </template>
      <!-- A BOARD — four cells, no plus (operator 2026-09-27, picked from a rendered comparison).
           THE GLYPH IT REPLACED. It drew `M6 3v18l6-4 6 4V3z`, the bookmark — which is this app's
           HIGHLIGHT mark: `CaptureMoment` in the player transport draws it for mark-a-moment, and
           the transcript's line save draws it too. So a collection control and a capture control
           were the same shape, and the operator read the transport's capture button as a stray
           copy of this one and asked for it to be deleted. It is not a copy, and on a phone it is
           the only way to mark a moment: a glyph collision came one instruction from removing a
           feature. One glyph per concept now — see UXS-014.
           WHY A GRID, NOT A FOLDER. A folder was tried first and rejected on sight: it is the
           filesystem's metaphor, and these are `Boards` (RFC-119 calls them pinboards) — the wrong
           idea before it is the wrong drawing. It also carried a `+`, which is what made it the
           busiest mark in the set at 16px.
           WHY NO PLUS. The pill variant already says "+ Collection" in words and the icon variant
           carries `collections.addTo` as its accessible name, so the `+` was a third stroke paying
           for nothing. Dropping it is the same call made against the folded-corner-plus-plus glyph
           on 2026-09-13, for the same reason.
           WHY FOUR CELLS RATHER THAN OFFSET CARDS. Stacked cards say "a set kept together" more
           precisely, and were the first recommendation — but their whole meaning is the OVERLAP,
           and at 16px, the size that actually ships in a card row, the overlap mushes into a thick
           square. Four separated cells keep air between the strokes and stay crisp. Rendered at
           16/24/64 side by side before choosing; legibility at the shipping size won over the
           better metaphor.
           Nothing else in the app draws a grid — checked against the compass (Browse) and the
           4-bar (Library) at 16px in the muted state they share. The "grid means app launcher"
           worry is a convention imported from other software, not a collision here. -->
      <svg v-else viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4 shrink-0" aria-hidden="true">
        <rect x="3" y="3" width="7" height="7" rx="1.5" /><rect x="14" y="3" width="7" height="7" rx="1.5" />
        <rect x="3" y="14" width="7" height="7" rx="1.5" /><rect x="14" y="14" width="7" height="7" rx="1.5" />
      </svg>
      <!-- A non-hidden accessible name for the ICON-ONLY variant (2026-09-24, Android device tier).
           `aria-haspopup` PLUS a fully hidden subtree leaves the button UNNAMED on Android System
           WebView 150; the pill and menuitem variants already carry readable text, this one did
           not. See OverflowMenu.vue for the measurement. -->
      <span v-if="variant !== 'pill' && variant !== 'menuitem'" class="sr-only">{{
        t('collections.addTo')
      }}</span>
      <span v-if="variant === 'menuitem'">{{ t('collections.addTo') }}</span>
    </button>

    <Teleport :to="teleportTarget">
      <div
        v-if="open"
        ref="panelEl"
        class="invisible fixed left-0 top-0 z-50 w-56 max-w-[calc(100vw-1rem)] rounded-xl border border-border bg-surface p-2 shadow-lg"
        data-testid="add-to-collection-menu"
        @click.stop
      >
      <p class="px-2 pb-1 text-xs font-bold uppercase tracking-wide text-muted">
        {{ t('collections.addTo') }}
      </p>
      <!-- A board that ALREADY holds this item says so (operator 2026-09-19). Every row used to
           look identical, so the only way to find out where something already lived was to add it
           again and watch nothing happen — the add is idempotent, so that tap is silent.

           `holds()` is false when the lookup did not happen, so a failed membership read renders
           nothing rather than a confident "not in this one". The word "Added" carries the state,
           not the tick alone — a bare ✓ beside a name reads as "selected", which is the opposite
           of what it means here. -->
      <ul class="max-h-48 overflow-y-auto">
        <li v-for="c in collections" :key="c.id">
          <!--
            The row states its own colour and its own width (operator 2026-09-27: the board NAMES
            did not render on device — "✓ Added" was there, the name beside it was not).

            NOT REPRODUCED, so this is the two ways it could happen removed, not a diagnosis. What
            was ruled out, each by measurement rather than by reading: the stored rows on prod carry
            their names (`AI`, `Investments`, `Tech`); the Vue DOM renders all three with the right
            classes; the compiled CSS paints them at full width and full contrast in Chromium, both
            standalone and underneath the sheet this was opened from; and prod runs this exact file.
            What is left is the iOS WKWebView the screenshot came from, which is not reachable here.

            So both remaining candidates are closed off by construction:

            1. COLOUR was inherited. The name was the ONLY text in this teleported panel with no
               colour of its own — the header, the "✓ Added", the input and Create all state theirs,
               which is why they survived and it did not. The panel teleports to `<body>` or into an
               open `<dialog>`, and a `<dialog>`'s UA style sets `color: CanvasText`, so a control
               relying on inheritance can land black-on-black through no fault of the theme. Both
               branches are explicit now.
            2. WIDTH was `flex-basis: auto` plus `min-w-0`, which lets this span — and only this
               span, since its sibling is `shrink-0` — be shrunk to zero by the flex algorithm.
               At zero width `truncate` (`overflow: hidden`) renders nothing at all: no text, not
               even an ellipsis. Exactly the symptom. `flex-1` gives it a definite basis and the
               remaining space instead.
          -->
          <button
            type="button"
            class="flex w-full items-center justify-between gap-2 rounded-lg px-2 py-1.5 text-left text-sm transition hover:bg-overlay"
            :class="holds(c.id) ? 'text-grounded' : 'text-canvas-foreground'"
            data-testid="add-to-collection-pick"
            :data-contains="holds(c.id) ? 'true' : undefined"
            @click="pick(c.id)"
          >
            <!--
              NO TRUNCATION. The board name wraps rather than being clipped (operator 2026-09-27,
              after three wrong fixes and a diagnostic build).

              The bug: on the SECOND open of this menu the names vanished, leaving only "✓ Added".
              It read as a colour problem for three attempts and it was a LAYOUT problem.

              Proven on device by tinting this span's box. First open: full-width box, names
              visible. Second open: the box collapsed to a sliver, wide enough only for the
              diagnostic's character count, with the name clipped away. The discriminator is the
              `shrink-0` "✓ Added" sibling, which only exists once the item is in that board — i.e.
              from the second open onwards. With no sibling the name is the row's only child and
              gets full width whatever the flex maths says; with one, the distribution matters, and
              it was being computed against the wrong container width — `place()` forces a
              synchronous layout while the panel is still `visibility:hidden` at `left:0;top:0`,
              and nothing invalidates it after the reveal.

              `flex-1` did not save it: `flex: 1 1 0%` distributes FREE SPACE, and there was none to
              distribute. `truncate`'s `overflow:hidden` then hid the text instead of letting it
              spill, which is precisely what made a layout fault look like an invisible colour.

              So the fix removes the need for the measurement to be right rather than trying to fix
              the measurement: no `overflow:hidden`, no `nowrap`, no `flex-1`. A 224px panel with
              short board names has room to wrap, and a wrapped name is legible where a clipped one
              is nothing. `break-words` keeps a pathological name from widening the panel.

              Deliberately NOT touched: `useAnchoredMenu`'s invisible-measure-reveal flow. Every
              menu in the app shares it and it exists to prevent a focus-blur and an off-position
              flash. If it needs fixing it should be fixed in `place()`, for all of them, not worked
              around here.
            -->
            <span class="min-w-0 flex-1 break-words">{{ c.name }}</span>
            <span
              v-if="addedTo === c.id || holds(c.id)"
              class="shrink-0 whitespace-nowrap text-xs text-grounded"
              >✓ {{ t('collections.alreadyIn') }}</span
            >
          </button>
        </li>
      </ul>
      <p
        v-if="error"
        data-testid="collection-error"
        class="px-2 pb-1 text-xs font-semibold text-danger"
        role="alert"
      >{{ error }}</p>
      <form class="mt-1 flex gap-1 border-t border-border pt-2" @submit.prevent="createAndAdd">
        <input
          v-model="newName"
          type="text"
          :placeholder="t('collections.namePlaceholder')"
          class="min-w-0 flex-1 rounded-lg border border-border bg-canvas px-2 py-1 text-sm outline-none focus:border-accent"
        />
        <button type="submit" class="shrink-0 rounded-lg bg-accent px-2 py-1 text-sm font-bold text-accent-foreground">
          {{ t('collections.create') }}
        </button>
      </form>
      </div>
    </Teleport>
  </div>
</template>
