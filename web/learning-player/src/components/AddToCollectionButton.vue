<script setup lang="ts">
/**
 * Add-to-collection control (RFC-119) — a compact icon button that pins ANY typed item (episode /
 * show / search / topic / person / link / highlight) into one of the user's collections. Opens a
 * small menu of collections (loaded on first open) with an inline "new collection" create. Sign-in
 * gated, like the queue / favourite controls. Reusable across every surface that pins.
 */
import { ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { addToCollection, createCollection, getCollections } from '../services/api'
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
const { open, toggle, close } = useAnchoredMenu(triggerEl, panelEl, { align: 'end' })

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
    collections.value = await getCollections()
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
      <!-- A plain bookmark — the folded-corner-plus-plus glyph was too busy at 16px (operator
           2026-09-13). "Add to collection" is carried by the aria-label / menu, not by icon detail. -->
      <svg v-else viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4 shrink-0" aria-hidden="true">
        <path d="M6 3v18l6-4 6 4V3z" />
      </svg>
      <span v-if="variant === 'menuitem'">{{ t('collections.addTo') }}</span>
    </button>

    <Teleport to="body">
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
    </Teleport>
  </div>
</template>
