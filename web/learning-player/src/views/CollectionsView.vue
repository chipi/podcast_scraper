<script setup lang="ts">
/**
 * Collections / boards (PRD-046 FR4 / #1417) — the curation layer: named sets of highlights that
 * span episodes. Create a board, open it to see its highlights (with jump-to-moment), delete it.
 * Embedded in the Library "Collections" tab. Auth-gated (empty when signed out).
 */
import { onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import {
  createCollection,
  deleteCollection,
  getCollection,
  getCollections,
  removeFromCollection,
} from '../services/api'
import type { Collection, CollectionDetail, CollectionItem } from '../services/types'

const { t } = useI18n()

const collections = ref<Collection[]>([])
const open = ref<CollectionDetail | null>(null)
const newName = ref('')
const loaded = ref(false)

async function load(): Promise<void> {
  collections.value = await getCollections().catch(() => [])
  loaded.value = true
}

async function create(): Promise<void> {
  const name = newName.value.trim()
  if (!name) return
  const created = await createCollection(name)
  collections.value = [created, ...collections.value]
  newName.value = ''
}

async function openCollection(id: string): Promise<void> {
  open.value = await getCollection(id)
}

async function remove(id: string): Promise<void> {
  collections.value = await deleteCollection(id)
  if (open.value?.collection.id === id) open.value = null
}

async function removeItem(it: CollectionItem): Promise<void> {
  if (!open.value) return
  const cid = open.value.collection.id
  await removeFromCollection(cid, it.kind, it.ref)
  open.value = await getCollection(cid) // re-resolve so the list + count stay honest
}

onMounted(load)
</script>

<template>
  <div>
    <!-- create -->
    <form class="mb-4 flex items-center gap-2" @submit.prevent="create">
      <input
        v-model="newName"
        type="text"
        :placeholder="t('collections.namePlaceholder')"
        class="min-w-0 flex-1 rounded-lg border border-border bg-overlay px-3 py-2 text-sm"
        maxlength="120"
      />
      <button
        type="submit"
        class="rounded-full bg-accent px-4 py-2 text-sm font-bold text-canvas disabled:opacity-50"
        :disabled="!newName.trim()"
      >{{ t('collections.create') }}</button>
    </form>

    <p v-if="loaded && !collections.length" class="text-sm text-muted">{{ t('collections.empty') }}</p>

    <!-- detail view of an open collection -->
    <section v-if="open" class="mb-4 rounded-2xl border border-border p-4">
      <div class="mb-3 flex items-center justify-between gap-2">
        <h3 class="font-display text-lg font-bold">{{ open.collection.name }}</h3>
        <button type="button" class="text-sm text-accent" @click="open = null">{{ t('collections.back') }}</button>
      </div>
      <p v-if="!open.items.length" class="text-sm text-muted">{{ t('collections.emptyBoard') }}</p>
      <ul v-else class="flex flex-col gap-2" data-testid="collection-items">
        <li
          v-for="it in open.items"
          :key="it.kind + '|' + it.ref"
          class="flex items-center gap-2 rounded-xl border border-border p-3"
        >
          <span class="shrink-0 rounded-full bg-overlay px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide text-muted">
            {{ t('collections.kind.' + it.kind) }}
          </span>
          <a
            v-if="it.kind === 'link'"
            :href="it.deep_link ?? it.ref"
            target="_blank"
            rel="noopener"
            class="min-w-0 flex-1 truncate text-sm font-semibold text-canvas-foreground no-underline"
          >{{ it.title ?? it.ref }}</a>
          <RouterLink
            v-else-if="it.deep_link"
            :to="it.deep_link"
            class="min-w-0 flex-1 truncate text-sm font-semibold text-canvas-foreground no-underline"
          >{{ it.title ?? it.ref }}</RouterLink>
          <span v-else class="min-w-0 flex-1 truncate text-sm font-semibold">{{ it.title ?? it.ref }}</span>
          <button
            type="button"
            class="shrink-0 rounded-full px-1.5 text-xs text-muted transition hover:text-danger"
            :aria-label="t('collections.removeItem')"
            data-testid="collection-item-remove"
            @click="removeItem(it)"
          >✕</button>
        </li>
      </ul>
    </section>

    <!-- collection list -->
    <ul v-if="collections.length" class="flex flex-col gap-2">
      <li
        v-for="c in collections"
        :key="c.id"
        class="flex items-center justify-between gap-2 rounded-xl border border-border p-3"
      >
        <button type="button" class="min-w-0 flex-1 text-left" @click="openCollection(c.id)">
          <span class="font-semibold">{{ c.name }}</span>
          <span class="ml-2 text-xs text-muted">{{ t('collections.count', c.count, { named: { count: c.count } }) }}</span>
        </button>
        <button
          type="button"
          class="rounded-full p-1 text-muted transition hover:text-danger"
          :aria-label="t('collections.remove')"
          @click="remove(c.id)"
        >✕</button>
      </li>
    </ul>
  </div>
</template>
