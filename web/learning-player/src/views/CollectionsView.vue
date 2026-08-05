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
} from '../services/api'
import type { Collection, CollectionDetail } from '../services/types'
import { formatTime } from '../player/transcriptSync'

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

function jumpQuery(startMs: number | null): Record<string, string> {
  return startMs != null ? { t: String(Math.floor(startMs / 1000)) } : {}
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
      <p v-if="!open.highlights.length" class="text-sm text-muted">{{ t('collections.emptyBoard') }}</p>
      <ul v-else class="flex flex-col gap-2">
        <li v-for="h in open.highlights" :key="h.id" class="rounded-xl border border-border p-3">
          <p class="text-sm font-semibold leading-snug">{{ h.quote_text ?? t('collections.moment') }}</p>
          <div class="mt-1 flex items-center gap-2">
            <RouterLink
              v-if="h.start_ms != null"
              :to="{ name: 'player', params: { slug: h.episode_slug }, query: jumpQuery(h.start_ms) }"
              class="font-mono text-xs text-accent no-underline"
            >▶ {{ formatTime(h.start_ms / 1000) }}</RouterLink>
            <span
              v-for="r in h.graph_refs ?? []"
              :key="r.id"
              class="rounded-full bg-overlay px-2 py-0.5 text-xs"
              :class="r.kind === 'person' ? 'text-person' : 'text-topic'"
            >{{ r.label }}</span>
          </div>
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
