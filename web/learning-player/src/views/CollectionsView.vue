<script setup lang="ts">
/**
 * Collections / boards (PRD-046 FR4 / #1417) — the curation layer: named sets of highlights that
 * span episodes. Create a board, open it to see its highlights (with jump-to-moment), delete it.
 * Embedded in the Library "Collections" tab. Auth-gated (empty when signed out).
 */
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink, useRouter } from 'vue-router'
import {
  addToCollection,
  createCollection,
  deleteCollection,
  getCollection,
  getCollections,
  getEpisode,
  removeFromCollection,
} from '../services/api'
import type { Collection, CollectionDetail, CollectionItem } from '../services/types'
import { useQueueStore } from '../stores/queue'
import { useSignInGate } from '../composables/useSignInGate'

const { t } = useI18n()
const router = useRouter()
const queue = useQueueStore()
const { gated } = useSignInGate()

const collections = ref<Collection[]>([])
const open = ref<CollectionDetail | null>(null)
const newName = ref('')
const newLink = ref('')
const loaded = ref(false)

const episodeItems = computed(() => open.value?.items.filter((i) => i.kind === 'episode') ?? [])

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
  void hydrateEpisodeTitles()
}

// Episode items come back with the slug as their ref and no title (the client hydrates display).
// Fill titles in place so a board reads as episode names, not slugs.
async function hydrateEpisodeTitles(): Promise<void> {
  const items = open.value?.items ?? []
  await Promise.all(
    items
      .filter((i) => i.kind === 'episode' && !i.title)
      .map(async (i) => {
        const ep = await getEpisode(i.ref).catch(() => null)
        if (ep) {
          i.title = ep.title
          i.subtitle = ep.podcast_title
        }
      }),
  )
}

/** Queue every episode in this collection, oldest-pinned first, and open the first (#1839 P4). */
const playAll = gated(async () => {
  const eps = episodeItems.value
  if (!eps.length) return
  for (const it of eps) await queue.add(it.ref)
  void router.push({ name: 'player', params: { slug: eps[0].ref } })
})

async function addLink(): Promise<void> {
  if (!open.value) return
  const url = newLink.value.trim()
  if (!url) return
  const cid = open.value.collection.id
  await addToCollection(cid, { kind: 'link', ref: url }).catch(() => null)
  newLink.value = ''
  open.value = await getCollection(cid)
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
        <h3 class="min-w-0 truncate font-display text-lg font-bold">{{ open.collection.name }}</h3>
        <div class="flex shrink-0 items-center gap-2">
          <button
            v-if="episodeItems.length"
            type="button"
            class="rounded-full bg-accent px-3 py-1 text-sm font-bold text-accent-foreground"
            data-testid="collection-play-all"
            @click="playAll"
          >▶ {{ t('collections.playAll') }}</button>
          <button type="button" class="text-sm text-accent" @click="open = null">{{ t('collections.back') }}</button>
        </div>
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

      <!-- Pin an external link (an article / blog post found while researching) — URL only (RFC-119). -->
      <form class="mt-3 flex gap-1 border-t border-border pt-3" @submit.prevent="addLink">
        <input
          v-model="newLink"
          type="url"
          :placeholder="t('collections.addLinkPlaceholder')"
          class="min-w-0 flex-1 rounded-lg border border-border bg-canvas px-3 py-1.5 text-sm outline-none focus:border-accent"
          data-testid="collection-add-link"
        />
        <button
          type="submit"
          class="shrink-0 rounded-lg bg-overlay px-3 py-1.5 text-sm font-bold text-accent"
          :disabled="!newLink.trim()"
        >{{ t('collections.addLink') }}</button>
      </form>
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
