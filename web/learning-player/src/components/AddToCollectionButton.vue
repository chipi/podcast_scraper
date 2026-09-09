<script setup lang="ts">
/**
 * Add-to-collection control (RFC-119) — a compact icon button that pins ANY typed item (episode /
 * show / search / topic / person / link / highlight) into one of the user's collections. Opens a
 * small menu of collections (loaded on first open) with an inline "new collection" create. Sign-in
 * gated, like the queue / favourite controls. Reusable across every surface that pins.
 */
import { nextTick, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { addToCollection, createCollection, getCollections } from '../services/api'
import { enqueue, isPermanent } from '../services/outbox'
import type { Collection, CollectionItemRef } from '../services/types'
import { useSignInGate } from '../composables/useSignInGate'

const props = withDefaults(
  defineProps<{
    item: CollectionItemRef
    /**
     * `icon` — compact round icon, for dense cards/rails (default). `pill` — a labelled pill
     * (`＋ Collection`) for roomy detail/player surfaces, matching the Follow pill idiom (CO.1).
     */
    variant?: 'icon' | 'pill'
  }>(),
  { variant: 'icon' },
)
const { t } = useI18n()
const { isGated, gated } = useSignInGate()

const open = ref(false)
const collections = ref<Collection[]>([])
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
 * Which edge the menu hangs from.
 *
 * It was always `right-0`: the panel is 224px wide and grows LEFTWARD from the button. That is
 * right for a control at the right edge of a card — where this button usually lives — and wrong
 * wherever it does not. On the entity card the button sits near the LEFT margin, so the menu ran
 * straight off the side of the phone and most of it was unreachable.
 *
 * Measured rather than guessed from the layout: the same component is used on browse rows, the show
 * page, search results, the entity card and the player masthead, and hard-coding a side per call
 * site is how this drifts back.
 */
const align = ref<'left' | 'right'>('right')
const menuEl = ref<HTMLElement | null>(null)

/** Keep the panel fully on screen, flipping to whichever edge has room. */
async function placeMenu(): Promise<void> {
  align.value = 'right'
  await nextTick()
  const el = menuEl.value
  if (!el) return
  const box = el.getBoundingClientRect()
  const MARGIN = 8
  // Overflowing the LEFT edge is the reported bug; check the right too, since flipping blindly
  // would just move the problem for a button near the right margin on a narrow screen.
  if (box.left < MARGIN && box.right + box.width <= window.innerWidth - MARGIN) {
    align.value = 'left'
  }
}

async function toggle(): Promise<void> {
  open.value = !open.value
  if (open.value) void placeMenu()
  if (open.value && !loaded.value) {
    error.value = null
    try {
      collections.value = await getCollections()
      loaded.value = true
    } catch {
      // NOT `loaded = true`: a failed load must retry on the next open rather than latch an empty
      // list that looks like "you have no collections".
      collections.value = []
      error.value = t('collections.loadFailed')
    }
  }
}
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
    window.setTimeout(() => {
      open.value = false
      addedTo.value = null
    }, 800)
    return
  }
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
    // Transient: queue the create and show it locally, like every other offline-capable write.
    const clientId = `col_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`
    enqueue({ op: 'collection.create', name, clientId })
    collections.value = [{ id: clientId, name, created_at: Date.now() / 1000, count: 0 }, ...collections.value]
    newName.value = ''
    return
  }
  collections.value = [created, ...collections.value]
  newName.value = ''
  await pick(created.id)
}
</script>

<template>
  <!--
    While the menu is OPEN this wrapper outranks every row control on the page, not just its own.

    The menu is absolutely positioned and taller than the card, so it overflows into the card BELOW
    it. That card's action buttons carry the same `z-30` and come later in document order, so at
    equal z-index they paint on top of the open menu and swallow clicks meant for it — the create
    button was unreachable in `collections.spec.ts`. The menu's own `z-40` cannot fix this: it is
    scoped to the stacking context this wrapper creates, so it orders siblings INSIDE the menu, not
    the wrapper against other cards. Raising the wrapper is what moves the whole context.
  -->
  <div class="relative inline-flex" :class="open ? 'z-50' : 'z-30'">
    <button
      type="button"
      :class="
        variant === 'pill'
          ? 'lp-tap inline-flex items-center gap-1 rounded-full bg-overlay px-3 py-1 text-xs font-bold text-canvas-foreground transition hover:bg-elevated'
          : 'lp-tap flex h-8 w-8 items-center justify-center rounded-full border border-border text-muted transition hover:text-canvas-foreground'
      "
      :aria-label="isGated ? t('auth.signInToSave') : t('collections.addTo')"
      :title="t('collections.addTo')"
      data-testid="add-to-collection"
      @click="onClick"
    >
      <template v-if="variant === 'pill'">
        <span aria-hidden="true">＋</span>
        {{ t('collections.pill') }}
      </template>
      <svg v-else viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" aria-hidden="true">
        <path d="M4 4h11l3 3v13l-6-3-6 3V4z" /><path d="M9 8h4M11 6v4" />
      </svg>
    </button>

    <div
      v-if="open"
      ref="menuEl"
      class="absolute top-9 z-40 w-56 max-w-[calc(100vw-1rem)] rounded-xl border border-border bg-surface p-2 shadow-lg"
      :class="align === 'left' ? 'left-0' : 'right-0'"
      data-testid="add-to-collection-menu"
    >
      <p class="px-2 pb-1 text-xs font-bold uppercase tracking-wide text-muted">
        {{ t('collections.addTo') }}
      </p>
      <ul class="max-h-48 overflow-y-auto">
        <li v-for="c in collections" :key="c.id">
          <button
            type="button"
            class="flex w-full items-center justify-between gap-2 rounded-lg px-2 py-1.5 text-left text-sm transition hover:bg-overlay"
            data-testid="add-to-collection-pick"
            @click="pick(c.id)"
          >
            <span class="min-w-0 truncate">{{ c.name }}</span>
            <span v-if="addedTo === c.id" class="shrink-0 text-xs text-grounded">✓</span>
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
  </div>
</template>
