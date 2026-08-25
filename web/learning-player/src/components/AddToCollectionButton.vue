<script setup lang="ts">
/**
 * Add-to-collection control (RFC-119) — a compact icon button that pins ANY typed item (episode /
 * show / search / topic / person / link / highlight) into one of the user's collections. Opens a
 * small menu of collections (loaded on first open) with an inline "new collection" create. Sign-in
 * gated, like the queue / favourite controls. Reusable across every surface that pins.
 */
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { addToCollection, createCollection, getCollections } from '../services/api'
import type { Collection, CollectionItemRef } from '../services/types'
import { useSignInGate } from '../composables/useSignInGate'

const props = defineProps<{ item: CollectionItemRef }>()
const { t } = useI18n()
const { isGated, gated } = useSignInGate()

const open = ref(false)
const collections = ref<Collection[]>([])
const loaded = ref(false)
const newName = ref('')
const addedTo = ref<string | null>(null)

async function toggle(): Promise<void> {
  open.value = !open.value
  if (open.value && !loaded.value) {
    collections.value = await getCollections().catch(() => [])
    loaded.value = true
  }
}
const onClick = gated(toggle)

async function pick(id: string): Promise<void> {
  const updated = await addToCollection(id, props.item).catch(() => null)
  if (updated) {
    const i = collections.value.findIndex((c) => c.id === updated.id)
    if (i >= 0) collections.value[i] = updated
    addedTo.value = id
    window.setTimeout(() => {
      open.value = false
      addedTo.value = null
    }, 800)
  }
}

async function createAndAdd(): Promise<void> {
  const name = newName.value.trim()
  if (!name) return
  const created = await createCollection(name).catch(() => null)
  if (!created) return
  collections.value = [created, ...collections.value]
  newName.value = ''
  await pick(created.id)
}
</script>

<template>
  <div class="relative z-30 inline-flex">
    <button
      type="button"
      class="flex h-8 w-8 items-center justify-center rounded-full border border-border text-muted transition hover:text-canvas-foreground"
      :aria-label="isGated ? t('auth.signInToSave') : t('collections.addTo')"
      :title="t('collections.addTo')"
      data-testid="add-to-collection"
      @click="onClick"
    >
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" aria-hidden="true">
        <path d="M4 4h11l3 3v13l-6-3-6 3V4z" /><path d="M9 8h4M11 6v4" />
      </svg>
    </button>

    <div
      v-if="open"
      class="absolute right-0 top-9 z-40 w-56 rounded-xl border border-border bg-surface p-2 shadow-lg"
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
