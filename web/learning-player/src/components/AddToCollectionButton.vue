<script setup lang="ts">
/**
 * Add-to-board control (RFC-119) — a compact icon button that pins ANY typed item (episode / show /
 * search / topic / person / link / highlight) into one of the user's boards. Opens the "Save to
 * board" sheet: every board with its picture and count, ⊕ / ✓ per row, and a "New board" create.
 * Sign-in gated, like the queue / favourite controls. Reusable across every surface that pins.
 */
import { nextTick, ref, watch } from 'vue'
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
import { useSheetDrag } from '../composables/useSheetDrag'
import { useCollectionsStore } from '../stores/collections'

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
/**
 * The SHARED list, which Library's Boards tab renders and which stays mounted between tab switches.
 * Every board this sheet creates or adds to goes there too: the sheet used to update only its own
 * `collections`, so a new board — or a new count — never reached Library until a full reload
 * (operator on device, 2026-10-09).
 */
const store = useCollectionsStore()
/** Collection ids holding `props.item`. Empty AND `membershipKnown=false` means "we could not look". */
const holdingIds = ref<Set<string>>(new Set())
const membershipKnown = ref(false)
const holds = (id: string): boolean => membershipKnown.value && holdingIds.value.has(id)
const loaded = ref(false)
const newName = ref('')
const addedTo = ref<string | null>(null)
/** The "New board" field is folded until asked for: the sheet leads with the boards themselves. */
const creating = ref(false)
const nameEl = ref<HTMLInputElement | null>(null)
async function startCreate(): Promise<void> {
  creating.value = !creating.value
  if (creating.value) {
    await nextTick()
    nameEl.value?.focus()
  }
}
/** Boards whose cover failed to load fall back to the board glyph rather than a broken image. */
const brokenCovers = ref<Set<string>>(new Set())

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
const { open, toggle, close, teleportTarget } = useAnchoredMenu(
  triggerEl,
  panelEl,
  { align: 'end' },
  { anchored: false },
)
const handleDrag = useSheetDrag(panelEl, () => close(false))

// Load collections on first open; clear the transient "added" receipt whenever it closes.
watch(open, async (isOpen) => {
  if (!isOpen) {
    addedTo.value = null
    creating.value = false
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
  store.upsert(updated)
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
    const pending: Collection = { id: clientId, name, created_at: Date.now() / 1000, count: 1 }
    collections.value = [pending, ...collections.value]
    store.upsert(pending)
    newName.value = ''
    addedTo.value = clientId
    window.setTimeout(() => close(false), 800)
    return
  }
  collections.value = [created, ...collections.value]
  store.upsert(created)
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

    <!-- A SHEET, not a 224px menu (operator 2026-10-07, after Instagram's save-to-collection sheet):
         bigger rows, each board's picture, its count, and one clear state per row — ⊕ to add, a
         filled ✓ where the item already is. A bottom sheet on a phone, a centred panel from `sm`.
         Same open/close shell (`useAnchoredMenu`) for Escape, outside-tap and the teleport target;
         its anchor placement does not apply to a sheet, so the panel is positioned by its classes. -->
    <Teleport :to="teleportTarget">
      <!-- The app's shared sheet scrim: dims the page, bottom-anchors the sheet on a phone and
           centres it from `sm`. A tap on the scrim itself closes, like every other sheet. -->
      <div
        v-if="open"
        class="lp-sheet-scrim"
        data-testid="add-to-collection-backdrop"
        @click.self.stop="close(false)"
      >
      <div
        ref="panelEl"
        role="dialog"
        :aria-label="t('collections.sheetTitle')"
        class="lp-sheet lp-sheet--half w-full max-w-lg overflow-y-auto rounded-t-2xl border border-border bg-surface text-canvas-foreground shadow-xl sm:rounded-2xl"
        data-testid="add-to-collection-menu"
        @click.stop
      >
      <!-- Padding on an inner box: `.lp-sheet` owns height and the bottom safe-area inset, and a
           padding utility on the sheet itself would override that inset. -->
      <div class="px-4 pb-4 pt-2">
        <!-- Pull the handle down to close (operator 2026-10-10). -->
        <div
          class="-mx-4 -mt-2 mb-1 flex h-7 touch-none items-center justify-center sm:hidden"
          aria-hidden="true"
          data-testid="add-to-collection-handle"
          v-bind="handleDrag"
        >
          <span class="h-1 w-10 rounded-full bg-border" />
        </div>
        <div class="mb-2 flex items-center justify-between gap-3">
          <h2 class="font-display text-lg font-bold text-canvas-foreground">{{ t('collections.sheetTitle') }}</h2>
          <button
            type="button"
            class="shrink-0 text-sm font-bold text-accent"
            data-testid="add-to-collection-new"
            :aria-expanded="creating"
            @click="startCreate"
          >+ {{ t('collections.newBoard') }}</button>
        </div>

        <form
          v-if="creating"
          class="mb-3 flex gap-2"
          data-testid="add-to-collection-create"
          @submit.prevent="createAndAdd"
        >
          <input
            ref="nameEl"
            v-model="newName"
            type="text"
            :placeholder="t('collections.namePlaceholder')"
            :aria-label="t('collections.namePlaceholder')"
            class="min-w-0 flex-1 rounded-xl border border-border bg-canvas px-3 py-2 text-sm text-canvas-foreground outline-none focus:border-accent"
            data-testid="add-to-collection-name"
          />
          <button type="submit" class="shrink-0 rounded-xl bg-accent px-4 py-2 text-sm font-bold text-accent-foreground">
            {{ t('collections.create') }}
          </button>
        </form>

        <p
          v-if="error"
          data-testid="collection-error"
          class="mb-2 text-sm font-semibold text-danger"
          role="alert"
        >{{ error }}</p>

        <!-- Rows state their own colour and wrap their names rather than clip them: the two
             fixes the old menu needed on device (2026-09-27) — an inherited colour inside a
             teleported `<dialog>` went black-on-black, and `truncate` hid a name the flex maths
             had squeezed to nothing. Neither can recur here: explicit colour, no clipping. -->
        <ul class="flex flex-col">
          <li v-for="c in collections" :key="c.id">
            <button
              type="button"
              class="flex w-full items-center gap-3 rounded-xl px-1 py-2 text-left transition hover:bg-overlay"
              :class="holds(c.id) ? 'text-grounded' : 'text-canvas-foreground'"
              data-testid="add-to-collection-pick"
              :data-contains="holds(c.id) ? 'true' : undefined"
              :aria-pressed="addedTo === c.id || holds(c.id)"
              @click="pick(c.id)"
            >
              <img
                v-if="c.cover_url && !brokenCovers.has(c.id)"
                :src="c.cover_url"
                alt=""
                loading="lazy"
                class="h-14 w-14 shrink-0 rounded-xl bg-elevated object-cover"
                data-testid="add-to-collection-thumb"
                @error="brokenCovers = new Set(brokenCovers).add(c.id)"
              />
              <!-- No picture yet: the board's initial, so picture-less boards are told apart at a glance
                   (a repeated icon would make them identical), and the trigger keeps the one board
                   glyph in this file (save-affordances.test). -->
              <span
                v-else
                class="flex h-14 w-14 shrink-0 items-center justify-center rounded-xl bg-elevated font-display text-xl font-bold text-muted"
                aria-hidden="true"
                data-testid="add-to-collection-thumb"
              >{{ (c.name.trim()[0] || '#').toUpperCase() }}</span>
              <span class="min-w-0 flex-1">
                <span class="block break-words font-semibold" data-testid="add-to-collection-board-name">{{ c.name }}</span>
                <span class="block text-sm text-muted">{{
                  addedTo === c.id || holds(c.id) ? t('collections.savedHere') : t('collections.count', c.count, { named: { count: c.count } })
                }}</span>
              </span>
              <!-- The state, drawn: a filled ✓ where the item already is, an outlined ⊕ where it
                   could go. "Added" in words beside the name carries it for a screen reader. -->
              <span
                v-if="addedTo === c.id || holds(c.id)"
                class="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-canvas-foreground text-canvas"
                aria-hidden="true"
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4"><path d="M5 12.5l4.5 4.5L19 7.5" /></svg>
              </span>
              <span
                v-else
                class="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border-2 border-canvas-foreground/70 text-canvas-foreground"
                aria-hidden="true"
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" class="h-4 w-4"><path d="M12 6v12M6 12h12" /></svg>
              </span>
              <span v-if="addedTo === c.id || holds(c.id)" class="sr-only">{{ t('collections.alreadyIn') }}</span>
            </button>
          </li>
        </ul>
        <p v-if="loaded && !collections.length && !creating" class="py-2 text-sm text-muted" data-testid="add-to-collection-none">
          {{ t('collections.empty') }}
        </p>
      </div>
      </div>
      </div>
    </Teleport>
  </div>
</template>
