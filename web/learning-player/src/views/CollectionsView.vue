<script setup lang="ts">
/**
 * Collections / boards (PRD-046 FR4 / #1417) — the curation layer: named sets of highlights that
 * span episodes. Create a board, open it to see its highlights (with jump-to-moment), delete it.
 * Embedded in the Library "Collections" tab. Auth-gated (empty when signed out).
 */
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import ConfirmDialog from '../components/ConfirmDialog.vue'
import SectionStatus from '../components/SectionStatus.vue'
import { useCollectionsStore } from '../stores/collections'
import { RouterLink, useRouter } from 'vue-router'
import { addToCollection, createCollection, deleteCollection, getCollection, getEpisode, getPodcasts, removeFromCollection } from '../services/api'
import type { Collection, CollectionDetail, CollectionItem } from '../services/types'
import { useQueueStore } from '../stores/queue'
import { useSignInGate } from '../composables/useSignInGate'

const { t } = useI18n()
const router = useRouter()
const queue = useQueueStore()
const { gated } = useSignInGate()

const collections = ref<Collection[]>([])
const open = ref<CollectionDetail | null>(null)

/**
 * Display data for the items the server does not resolve.
 *
 * The route's own docstring says it returns `{kind, ref, deep_link}` for everything except
 * highlights and "the client hydrates display through its existing endpoints" — and the client
 * never did. So an episode or a show rendered `title ?? ref`, and `ref` for a show is a content
 * hash: the board showed `sha256:68377a5abb…` where a person expects the show.
 *
 * Keyed by `kind|ref`, the same key the list renders by.
 */
const shown = ref<Record<string, { title: string; subtitle?: string; artwork?: string }>>({})

/** Items whose kind carries artwork; the rest are text rows with a kind chip. */
const HYDRATES = new Set(['episode', 'show'])

function itemKey(it: CollectionItem): string {
  return `${it.kind}|${it.ref}`
}

/**
 * Fill in titles and artwork for the open board, best-effort and in parallel.
 *
 * Failures are deliberately silent per item: one unresolvable show must not blank the rest of the
 * board, and the row falls back to its kind chip plus a shortened ref rather than to a raw hash.
 * Shows resolve from the podcasts list — one request for the whole board rather than one per row,
 * since a board of eight shows from one feed would otherwise be eight identical lookups.
 */
async function hydrate(detail: CollectionDetail): Promise<void> {
  const items = detail.items.filter((i) => HYDRATES.has(i.kind))
  if (!items.length) return

  const wantShows = items.some((i) => i.kind === 'show')
  const podcasts = wantShows ? await getPodcasts().catch(() => []) : []
  const byFeed = new Map(podcasts.map((p) => [p.feed_id, p]))

  await Promise.all(
    items.map(async (it) => {
      // A late response for a board the user already closed (or swapped) must not paint over the
      // one they are looking at.
      const stillOpen = () => open.value?.collection.id === detail.collection.id
      if (it.kind === 'show') {
        const p = byFeed.get(it.ref)
        if (p && stillOpen()) {
          shown.value[itemKey(it)] = {
            title: p.title ?? it.ref,
            subtitle: t('collections.episodeCount', p.episode_count, { named: { count: p.episode_count } }),
            artwork: p.artwork_url ?? p.image_url ?? undefined,
          }
        }
        return
      }
      try {
        const ep = await getEpisode(it.ref)
        if (!stillOpen()) return
        shown.value[itemKey(it)] = {
          title: ep.title,
          subtitle: ep.podcast_title ?? undefined,
          artwork: ep.artwork_url ?? ep.episode_image_url ?? ep.feed_image_url ?? undefined,
        }
      } catch {
        /* leave the row on its fallback — see the docstring */
      }
    }),
  )
}

/**
 * What a row shows when nothing resolved it.
 *
 * Never the bare ref: a `sha256:…` tells a person nothing and reads like a bug. The kind chip beside
 * it already says what sort of thing this is, so the text says the item could not be loaded and the
 * shortened ref stays only as a breadcrumb for whoever has to debug it.
 */
function fallbackTitle(it: CollectionItem): string {
  if (it.title) return it.title
  if (!HYDRATES.has(it.kind)) return it.ref
  const short = it.ref.length > 14 ? `${it.ref.slice(0, 14)}…` : it.ref
  return t('collections.unresolved', { ref: short })
}
const newName = ref('')
const newLink = ref('')
const loaded = ref(false)

const episodeItems = computed(() => open.value?.items.filter((i) => i.kind === 'episode') ?? [])

/**
 * A failed load is NOT an empty library (#2004 item 13).
 *
 * `getCollections().catch(() => [])` plus a `loaded` latch turned any failure — including the 401 the
 * API layer used to manufacture — into the "you have no collections yet" empty state. That is the
 * screen a user sees after creating a collection elsewhere and being told, in effect, that it never
 * existed. `loaded` now only latches on success, so the empty state means empty.
 */
const loadError = ref(false)
const linkError = ref(false)

async function load(): Promise<void> {
  // Through the STORE (#2013): a failed read falls back to the cached copy instead of rendering
  // "you have no collections yet", which is the specific lie that made the data look lost. The
  // error state is reserved for "no answer AND no cache".
  const store = useCollectionsStore()
  await store.load()
  collections.value = store.items
  loaded.value = store.loaded
  loadError.value = store.unavailable
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
  void hydrate(open.value)
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
  try {
    await addToCollection(cid, { kind: 'link', ref: url })
  } catch {
    // Keep the URL in the box: a link the user pasted must not vanish because the save failed.
    linkError.value = true
    return
  }
  linkError.value = false
  newLink.value = ''
  open.value = await getCollection(cid)
  void hydrate(open.value)
}

/**
 * Deleting a collection is confirmed (#1594) — it destroys a board the user built, and it cannot
 * be undone: `createCollection` mints a new id, so a restore would be a different collection with
 * the same name and every reference to the old one still broken.
 *
 * The pending id doubles as the open/closed flag, so there is no second boolean to keep in sync.
 */
const pendingDelete = ref<string | null>(null)

async function confirmDelete(): Promise<void> {
  const id = pendingDelete.value
  pendingDelete.value = null
  if (!id) return
  collections.value = await deleteCollection(id)
  if (open.value?.collection.id === id) open.value = null
}

async function removeItem(it: CollectionItem): Promise<void> {
  if (!open.value) return
  const cid = open.value.collection.id
  await removeFromCollection(cid, it.kind, it.ref)
  open.value = await getCollection(cid) // re-resolve so the list + count stay honest
  void hydrate(open.value)
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

    <SectionStatus v-if="loadError" phase="error" data-testid="collections-load-error" @retry="load" />
    <p v-else-if="loaded && !collections.length" class="text-sm text-muted">{{ t('collections.empty') }}</p>

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
        <!--
          A board row reads like every other row in the app: artwork, title, one line of context.
          It used to be a kind chip beside `title ?? ref`, so a show — whose ref is a content hash —
          rendered as `sha256:68377a5abb…`. The chip stays because a board is MIXED: it is the only
          thing telling a topic from a search from a link at a glance.
        -->
        <li
          v-for="it in open.items"
          :key="itemKey(it)"
          class="flex items-center gap-3 rounded-xl border border-border p-3"
          data-testid="collection-item"
        >
          <img
            v-if="shown[itemKey(it)]?.artwork"
            :src="shown[itemKey(it)]!.artwork"
            alt=""
            loading="lazy"
            class="h-12 w-12 shrink-0 rounded-lg object-cover"
          />
          <!-- Same 48px footprint whether or not artwork resolved, so rows do not jump as they
               hydrate and a kind without artwork still lines up with one that has it. -->
          <div
            v-else-if="HYDRATES.has(it.kind)"
            class="h-12 w-12 shrink-0 rounded-lg bg-elevated"
            aria-hidden="true"
          />

          <div class="min-w-0 flex-1">
            <component
              :is="it.kind === 'link' ? 'a' : it.deep_link ? RouterLink : 'span'"
              v-bind="
                it.kind === 'link'
                  ? { href: it.deep_link ?? it.ref, target: '_blank', rel: 'noopener' }
                  : it.deep_link
                    ? { to: it.deep_link }
                    : {}
              "
              class="block truncate text-sm font-semibold text-canvas-foreground no-underline"
              data-testid="collection-item-title"
            >{{ shown[itemKey(it)]?.title ?? fallbackTitle(it) }}</component>
            <div class="mt-0.5 flex items-center gap-2">
              <span class="lp-kicker shrink-0">{{ t('collections.kind.' + it.kind) }}</span>
              <span
                v-if="shown[itemKey(it)]?.subtitle ?? it.subtitle"
                class="min-w-0 truncate text-xs text-muted"
              >{{ shown[itemKey(it)]?.subtitle ?? it.subtitle }}</span>
            </div>
          </div>
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
      <!-- The link failure was set but never rendered — a failed pin changed nothing on screen
           except keeping the URL, which is the silent-write class this was meant to end. -->
      <p
        v-if="linkError"
        data-testid="collection-link-error"
        class="mt-1 text-xs font-semibold text-danger"
        role="alert"
      >{{ t('collections.addFailed') }}</p>
    </section>

    <!-- collection list -->
    <ul v-if="collections.length" class="flex flex-col gap-2">
      <li
        v-for="c in collections"
        :key="c.id"
        class="flex items-center justify-between gap-2 rounded-xl border border-border p-3"
      >
        <button
          type="button"
          class="min-w-0 flex-1 text-left"
          data-testid="collection-open"
          @click="openCollection(c.id)"
        >
          <span class="font-semibold">{{ c.name }}</span>
          <span class="ml-2 text-xs text-muted">{{ t('collections.count', c.count, { named: { count: c.count } }) }}</span>
        </button>
        <button
          type="button"
          class="lp-tap rounded-full p-1 text-muted transition hover:text-danger"
          :aria-label="t('collections.remove')"
          data-testid="collection-delete"
          @click="pendingDelete = c.id"
        >✕</button>
      </li>
    </ul>

    <ConfirmDialog
      :open="pendingDelete !== null"
      :title="t('collections.confirmDeleteTitle')"
      :body="t('collections.confirmDeleteBody')"
      :confirm-label="t('collections.confirmDelete')"
      data-testid="collection-delete-confirm"
      @confirm="confirmDelete"
      @cancel="pendingDelete = null"
    />
  </div>
</template>
