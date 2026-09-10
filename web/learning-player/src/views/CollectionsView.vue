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
import { useCaptureStore } from '../stores/capture'
import { RouterLink, useRouter } from 'vue-router'
import { addToCollection, createCollection, deleteCollection, getCollection, getEpisode, getPodcasts, removeFromCollection } from '../services/api'
import type { Collection, CollectionDetail, CollectionItem } from '../services/types'
import { useQueueStore } from '../stores/queue'
import { useSignInGate } from '../composables/useSignInGate'
import { formatPublishDate } from '../utils/format'

const { t, locale } = useI18n()
const router = useRouter()

/** "Last modified" date for a collection (CO.2), or null when unknown. */
function modifiedLabel(c: Collection): string | null {
  if (!c.updated_at) return null
  return formatPublishDate(new Date(c.updated_at * 1000).toISOString(), locale.value)
}
const queue = useQueueStore()
const { gated } = useSignInGate()
const capture = useCaptureStore()

// Search + sort across boards and notes (CO.5).
const search = ref('')
const sortBy = ref<'updated' | 'name' | 'count'>('updated')
function noteDate(unixSeconds: number): string {
  return formatPublishDate(new Date(unixSeconds * 1000).toISOString(), locale.value) ?? ''
}
/** A route to the note's target when it has a page; null otherwise (highlight/insight/storyline). */
function noteRoute(target: string, id: string): { name: string; params: Record<string, string> } | null {
  if (target === 'episode') return { name: 'player', params: { slug: id } }
  if (target === 'topic') return { name: 'topic', params: { id } }
  if (target === 'person') return { name: 'person', params: { id } }
  if (target === 'show') return { name: 'podcast', params: { feedId: id } }
  return null
}

const collections = ref<Collection[]>([])
const open = ref<CollectionDetail | null>(null)

// Boards filtered by the search box + sorted by the chosen key (CO.5).
const visibleCollections = computed(() => {
  const q = search.value.trim().toLowerCase()
  const list = q ? collections.value.filter((c) => c.name.toLowerCase().includes(q)) : [...collections.value]
  if (sortBy.value === 'name') return list.sort((a, b) => a.name.localeCompare(b.name))
  if (sortBy.value === 'count') return list.sort((a, b) => b.count - a.count)
  return list.sort((a, b) => (b.updated_at ?? b.created_at) - (a.updated_at ?? a.created_at))
})
// Notes newest-first, filtered by the same search box (NT.4 + CO.5).
const visibleNotes = computed(() => {
  const q = search.value.trim().toLowerCase()
  const base = [...capture.notes].sort((a, b) => b.created_at - a.created_at)
  return q ? base.filter((n) => n.text.toLowerCase().includes(q)) : base
})

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

/**
 * Accordion: one board open at a time, and tapping the open one closes it.
 *
 * The detail used to render in a SEPARATE panel above the list, with a "Back" link — so the open
 * board appeared twice on screen (once as a panel, once as a row below), and the only way to reach
 * another board was to close this one first. `open` already held a single board; the list simply
 * did not reflect it.
 */
async function openCollection(id: string): Promise<void> {
  if (open.value?.collection.id === id) {
    open.value = null
    return
  }
  // Clear immediately so a slow fetch cannot leave the previous board expanded under a different
  // row's header.
  open.value = null
  const detail = await getCollection(id)
  open.value = detail
  void hydrate(detail)
}

/** Queue every episode in this collection, oldest-pinned first, and open the first (#1839 P4). */
const playAll = gated(async () => {
  const eps = episodeItems.value
  if (!eps.length) return
  for (const it of eps) await queue.add(it.ref)
  void router.push({ name: 'player', params: { slug: eps[0].ref } })
})

/** Deleting a note is a per-user write — gate it like every other (#1590). Per-item id, so wrap
 * and invoke a zero-arg gated closure rather than passing an arg `gated()` does not accept. */
function removeNote(id: string): void {
  void gated(() => capture.removeNote(id))()
}

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

onMounted(() => {
  void load()
  void capture.ensureLoaded().catch(() => {})
})
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

    <!-- Search + sort across boards and notes (CO.5). Only when there's something to filter. -->
    <div v-if="collections.length || capture.notes.length" class="mb-4 flex flex-wrap items-center gap-2">
      <input
        v-model="search"
        type="search"
        :placeholder="t('collections.searchPlaceholder')"
        class="min-w-0 flex-1 rounded-full border border-border bg-surface px-4 py-2 text-sm outline-none focus:border-accent"
        data-testid="collections-search"
      />
      <select
        v-model="sortBy"
        :aria-label="t('list.sort')"
        class="shrink-0 rounded-full border border-border bg-surface px-3 py-2 text-sm font-semibold outline-none focus:border-accent"
        data-testid="collections-sort"
      >
        <option value="updated">{{ t('collections.sortUpdated') }}</option>
        <option value="name">{{ t('collections.sortName') }}</option>
        <option value="count">{{ t('collections.sortCount') }}</option>
      </select>
    </div>

    <!--
      An ACCORDION: one board open at a time, opened and closed in place.

      The open board used to render in a separate panel ABOVE this list, with a "Back" link — so it
      appeared twice on screen (as the panel, and again as a row here), and reaching another board
      meant closing this one first. Now the row IS the board: tapping it expands beneath its own
      header, tapping it again collapses it, and tapping a different one moves the expansion there.
    -->
    <ul v-if="visibleCollections.length" class="flex flex-col gap-2">
      <li
        v-for="c in visibleCollections"
        :key="c.id"
        class="rounded-xl border border-border"
        :class="open?.collection.id === c.id ? 'bg-overlay/40' : ''"
      >
        <div class="flex items-center justify-between gap-2 p-3">
          <button
            type="button"
            class="flex min-w-0 flex-1 items-center gap-2 text-left"
            data-testid="collection-open"
            :aria-expanded="open?.collection.id === c.id"
            @click="openCollection(c.id)"
          >
            <!-- The chevron is the affordance that says this opens in place rather than navigating
                 somewhere — which is what the old "Back" link implied it had done. -->
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2.5"
              stroke-linecap="round"
              stroke-linejoin="round"
              class="h-3 w-3 shrink-0 text-muted transition-transform"
              :class="open?.collection.id === c.id ? 'rotate-90' : ''"
              aria-hidden="true"
            >
              <path d="m9 6 6 6-6 6" />
            </svg>
            <span class="min-w-0">
              <span class="font-semibold">{{ c.name }}</span>
              <span class="ml-2 text-xs text-muted">{{ t('collections.count', c.count, { named: { count: c.count } }) }}</span>
              <span v-if="modifiedLabel(c)" class="block text-xs text-muted">
                {{ t('collections.updated', { date: modifiedLabel(c) }) }}
              </span>
            </span>
          </button>
          <button
            v-if="open?.collection.id === c.id && episodeItems.length"
            type="button"
            class="shrink-0 rounded-full bg-accent px-3 py-1 text-sm font-bold text-accent-foreground"
            data-testid="collection-play-all"
            @click="playAll"
          >▶ {{ t('collections.playAll') }}</button>
          <button
            type="button"
            class="lp-tap rounded-full p-1 text-muted transition hover:text-danger"
            :aria-label="t('collections.remove')"
            data-testid="collection-delete"
            @click="pendingDelete = c.id"
          >✕</button>
        </div>

        <div v-if="open?.collection.id === c.id" class="border-t border-border p-3">
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
        </div>
      </li>
    </ul>

    <!-- Notes (NT.4) — every note the user has taken, beside their boards in this tab. -->
    <section v-if="visibleNotes.length" class="mt-8" data-testid="collections-notes">
      <h2 class="lp-section mb-2">{{ t('notes.title') }}</h2>
      <ul class="flex flex-col gap-2">
        <li
          v-for="n in visibleNotes"
          :key="n.id"
          class="rounded-xl border border-border p-3"
          data-testid="collections-note"
        >
          <p class="whitespace-pre-wrap text-sm leading-relaxed text-canvas-foreground">{{ n.text }}</p>
          <div class="mt-1.5 flex items-center gap-2 text-xs">
            <span class="lp-kicker">{{ n.target }} · {{ noteDate(n.created_at) }}</span>
            <RouterLink
              v-if="noteRoute(n.target, n.target_id)"
              :to="noteRoute(n.target, n.target_id)!"
              class="font-semibold text-accent no-underline"
            >
              {{ t('notes.open') }}
            </RouterLink>
            <button
              type="button"
              class="ml-auto font-semibold text-muted transition hover:text-danger"
              :aria-label="t('notes.remove')"
              @click="removeNote(n.id)"
            >
              {{ t('notes.remove') }}
            </button>
          </div>
        </li>
      </ul>
    </section>

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
