<script setup lang="ts">
/**
 * Highlights review (P2 Capture, PRD-040) — the user's captured moments / spans / saved insights,
 * grouped by episode, each with jump-to-moment, inline notes, and delete. A Markdown export link
 * sits in the header (the single export format, REMEMBER-half-scope §4). Embedded in the Library
 * "Highlights" tab. Auth-gated (the store no-ops + stays empty when signed out).
 */
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import {
  addToCollection,
  exportObsidian,
  fetchHighlightsExport,
  getCollections,
  getEpisode,
  highlightsExportUrl,
} from '../services/api'
import type { Collection } from '../services/types'
import { isNative, saveAndShareText } from '../services/native'
import type { Highlight } from '../services/types'
import ConfirmDialog from '../components/ConfirmDialog.vue'
import SavedColorControl from '../components/SavedColorControl.vue'
import ShowAllToggle from '../components/ShowAllToggle.vue'
import { useCaptureStore } from '../stores/capture'
import { formatTime } from '../player/transcriptSync'
import { HIGHLIGHT_COLORS, borderClass } from '../utils/highlightColors'
import { matchesQuery } from '../utils/textFilter'
import { useCappedSections } from '../composables/useCappedSections'
import { shareHighlightCard } from '../composables/useShareCard'

const { t } = useI18n()
const capture = useCaptureStore()

/**
 * The colour filter and sort are owned by the Saved tab's filter bar now (RFC-121 ph. 3) and passed
 * in, so one bar governs every Saved section rather than a strip buried in this list. Defaults keep
 * the pre-lift behaviour (all colours, grouped by episode) for any standalone mount.
 */
const props = defineProps<{ filterColor?: string | null; sort?: string; search?: string }>()

// Episode groups are capped like every other Library section (#2042 follow-up); a search lifts it.
const groupCaps = useCappedSections()
const searchActive = computed(() => (props.search ?? '').trim() !== '')

// Palette order → a stable rank for the "by colour" sort (unknown/none sort last).
const COLOR_RANK = new Map(HIGHLIGHT_COLORS.map((c, i) => [c.token, i]))
function colorRank(token: string | null | undefined): number {
  return token != null && COLOR_RANK.has(token) ? (COLOR_RANK.get(token) as number) : Number.MAX_SAFE_INTEGER
}

// Episode titles for the group headings (slug → title), hydrated lazily; slug is the fallback.
const titles = ref<Record<string, string>>({})

interface Group {
  slug: string
  title: string
  highlights: Highlight[]
}

function sortWithin(list: Highlight[], sort: string): Highlight[] {
  const byRecent = (a: Highlight, b: Highlight): number => (b.created_at ?? 0) - (a.created_at ?? 0)
  if (sort === 'color') {
    return [...list].sort((a, b) => colorRank(a.color) - colorRank(b.color) || byRecent(a, b))
  }
  return [...list].sort(byRecent)
}

const groups = computed<Group[]>(() => {
  const sort = props.sort ?? 'episode'
  const query = props.search ?? ''
  const bySlug = new Map<string, Highlight[]>()
  for (const h of capture.highlights) {
    if (props.filterColor && h.color !== props.filterColor) continue
    // Search matches a highlight's own text (quote / speaker); episode titles are findable through
    // the Episodes section. Mirrors LibraryView's count predicate so the two agree.
    if (!(matchesQuery(h.quote_text, query) || matchesQuery(h.speaker, query))) continue
    const list = bySlug.get(h.episode_slug) ?? []
    list.push(h)
    bySlug.set(h.episode_slug, list)
  }
  const out = [...bySlug.entries()].map(([slug, highlights]) => ({
    slug,
    title: titles.value[slug] ?? slug,
    highlights: sortWithin(highlights, sort),
  }))
  // Group ORDER: A–Z by title when sorting by episode; otherwise most-recent group first (also the
  // natural order for the flat "recent"/"colour" reads over a grouped list — the newest work leads).
  if (sort === 'episode') {
    out.sort((a, b) => a.title.localeCompare(b.title))
  } else {
    const latest = (g: Group): number => Math.max(...g.highlights.map((h) => h.created_at ?? 0), 0)
    out.sort((a, b) => latest(b) - latest(a))
  }
  return out
})

const visibleGroups = computed<Group[]>(() =>
  groupCaps.visible('groups', groups.value, searchActive.value),
)

function jumpQuery(h: Highlight): Record<string, string> {
  return h.start_ms != null ? { t: String(Math.floor(h.start_ms / 1000)) } : {}
}

function label(h: Highlight): string {
  if (h.kind === 'moment') return t('highlights.moment')
  return h.quote_text ?? t('highlights.span')
}

// --- notes (inline add / edit) ---
/**
 * Pending destructive actions (#1594). Both deletes destroy something the user WROTE — a captured
 * moment and a note about it — and neither can be restored: the create endpoints mint new ids and
 * the deleted row leaves a tombstone the client cannot reason about. So the app asks first rather
 * than promising an undo it cannot honour.
 *
 * Two separate refs rather than one tagged union: a highlight and a note need different wording,
 * and collapsing them would mean a `kind` discriminator threaded through the template for no gain.
 */
const pendingHighlight = ref<string | null>(null)
const pendingNote = ref<string | null>(null)

function confirmRemoveHighlight(): void {
  const id = pendingHighlight.value
  pendingHighlight.value = null
  if (id) void capture.remove(id)
}

function confirmRemoveNote(): void {
  const id = pendingNote.value
  pendingNote.value = null
  if (id) void capture.removeNote(id)
}

const editing = ref<string | null>(null) // note id being edited, or `new:<highlightId>`
const draft = ref('')

function startAdd(highlightId: string): void {
  editing.value = `new:${highlightId}`
  draft.value = ''
}
function startEdit(noteId: string, text: string): void {
  editing.value = noteId
  draft.value = text
}
function cancel(): void {
  editing.value = null
  draft.value = ''
}
async function save(): Promise<void> {
  const text = draft.value.trim()
  const key = editing.value
  if (!key || !text) {
    cancel()
    return
  }
  if (key.startsWith('new:')) {
    await capture.addNote('highlight', key.slice(4), text)
  } else {
    await capture.editNote(key, text)
  }
  cancel()
}

// Native export: `<a download>` can't save in the iOS/Android WebView, so fetch the Markdown and
// hand it to the OS share sheet instead (#1310). Web keeps the plain download link.
const exporting = ref(false)
async function exportHighlightsNative(): Promise<void> {
  if (exporting.value) return
  exporting.value = true
  try {
    const md = await fetchHighlightsExport(props.filterColor)
    await saveAndShareText('my-highlights.md', md)
  } finally {
    exporting.value = false
  }
}

// Collections a highlight can be filed into (#1417). Loaded lazily; the per-highlight
// "Add to…" select adds on change then resets to its placeholder.
const collections = ref<Collection[]>([])
/** A failed collections load must not read as 'you have none' (#2004 item 13). */
const collectionsError = ref(false)

async function addHighlightTo(highlightId: string, collectionId: string): Promise<void> {
  if (!collectionId) return
  const updated = await addToCollection(collectionId, { kind: 'highlight', ref: highlightId })
  const i = collections.value.findIndex((c) => c.id === updated.id)
  if (i >= 0) collections.value[i] = updated
}

// Share a highlight as a text/quote card (#1418) — no audio (bridge-only).
async function share(h: Highlight): Promise<void> {
  await shareHighlightCard(h, titles.value[h.episode_slug] ?? h.episode_slug)
}

// Graph-aware Obsidian export (#1472). Incremental: the last-applied revision is remembered in
// localStorage so a repeat click only pulls what changed. First click (or cleared storage) is full.
const OBSIDIAN_CURSOR_KEY = 'obsidian_export_cursor'
const OBSIDIAN_EPOCH_KEY = 'obsidian_export_epoch'
const exportingObsidian = ref(false)
const obsidianMsg = ref('')
/** True after a SUCCESSFUL export — gates the "what to do with the zip" line (not shown on error). */
const obsidianDone = ref(false)

async function doObsidianExport(): Promise<void> {
  exportingObsidian.value = true
  obsidianMsg.value = ''
  obsidianDone.value = false
  try {
    // ALWAYS a full export from the web. The server's incremental protocol is correct, but it
    // assumes a client that APPLIES the manifest — writes `written`, deletes `removed`, honours
    // `replace_namespace`. This button does none of that: it downloads a zip and the user moves
    // the folder into their vault by hand. Hand-applying a delta is destructive either way —
    // Finder's default "Replace" drops every unchanged note, and "Merge" silently keeps the
    // orphans the tombstones exist to remove. A human will not execute a tombstone list.
    //
    // Restore the cursor here only alongside a programmatic applier (an Obsidian plugin, or the
    // native shell writing files itself). Until then `since=0` is the only safe request.
    const r = await exportObsidian(0)
    localStorage.setItem(OBSIDIAN_CURSOR_KEY, String(r.revision))
    // Stored beside the cursor, not instead of it. A revision only identifies a snapshot within
    // one server epoch (#41); persisting the number alone would leave whatever applier arrives
    // next holding an integer it cannot validate — exactly the collision the epoch exists to
    // catch. Unused while we always request since=0.
    localStorage.setItem(OBSIDIAN_EPOCH_KEY, r.epoch)
    obsidianMsg.value = t('highlights.obsidianFull', { written: r.written })
    obsidianDone.value = true
  } catch {
    obsidianMsg.value = t('highlights.obsidianError')
  } finally {
    exportingObsidian.value = false
  }
}

onMounted(async () => {
  // Tolerated, not awaited blindly: this view's whole job is to show captures, so a load failure
  // renders the empty state rather than tearing down the rest of the mount (collections, titles).
  await capture.ensureLoaded().catch(() => {})
  // Third caller of getCollections (#2004 item 13). Swallowing here reproduced the same lie the
  // other two stopped telling: a failed load renders as "you have no collections".
  try {
    collections.value = await getCollections()
  } catch {
    collections.value = []
    collectionsError.value = true
  }
  const slugs = [...new Set(capture.highlights.map((h) => h.episode_slug))]
  await Promise.all(
    slugs.map(async (slug) => {
      const d = await getEpisode(slug).catch(() => null)
      if (d) titles.value[slug] = d.title
    }),
  )
})
</script>

<template>
  <div>
    <div v-if="capture.count" class="mb-4 flex items-center justify-between gap-3">
      <p class="text-sm text-muted">{{ t('highlights.count', capture.count, { named: { count: capture.count } }) }}</p>
      <!-- Compact export cluster: a muted "Export" kicker + short format chips on ONE line. The
           full "Export Markdown" / "Export to Obsidian" survives as the aria-label (accessible name
           + e2e selector); the visible chips are `whitespace-nowrap text-xs` so they never wrap to
           two lines the way "Export to Obsidian" did on a phone. -->
      <div class="flex shrink-0 items-center gap-2">
        <span class="text-xs text-muted">{{ t('highlights.exportKicker') }}</span>
        <!-- Native shell: write+share (WKWebView can't `<a download>`); web: plain download link (#1310). -->
        <button
          v-if="isNative()"
          type="button"
          :disabled="exporting"
          :aria-label="t('highlights.export')"
          class="whitespace-nowrap rounded-full border border-border px-2.5 py-1 text-xs font-bold text-accent transition hover:bg-overlay disabled:opacity-50"
          @click="exportHighlightsNative"
        >{{ t('highlights.exportMarkdownShort') }}</button>
        <a
          v-else
          :href="highlightsExportUrl(filterColor)"
          download="my-highlights.md"
          :aria-label="t('highlights.export')"
          class="whitespace-nowrap rounded-full border border-border px-2.5 py-1 text-xs font-bold text-accent no-underline transition hover:bg-overlay"
        >{{ t('highlights.exportMarkdownShort') }}</a>
        <!-- Graph-aware Obsidian export (#1472) — web only (native zip handling is a follow). -->
        <button
          v-if="!isNative()"
          type="button"
          :disabled="exportingObsidian"
          :aria-label="t('highlights.exportObsidian')"
          class="whitespace-nowrap rounded-full border border-border px-2.5 py-1 text-xs font-bold text-accent transition hover:bg-overlay disabled:opacity-50"
          @click="doObsidianExport"
        >{{ t('highlights.exportObsidianShort') }}</button>
      </div>
    </div>
    <p v-if="obsidianMsg" class="mb-1 text-xs text-muted">{{ obsidianMsg }}</p>
    <!--
      Obsidian has no import format to target — a vault IS a folder of Markdown files, so the only
      way in is to put the folder there. Without saying so, the export ends at "here is a zip" and
      the user has to go and find out what to do with it, which is exactly what happened in review.
    -->
    <p v-if="obsidianDone" class="mb-3 text-xs text-muted">{{ t('highlights.obsidianNext') }}</p>

    <!-- The colour filter used to live here as an always-on swatch strip; it is lifted to the Saved
         tab's filter bar (RFC-121 ph. 3) and arrives as `filterColor`, so one bar governs every
         section. `sort` arrives the same way. -->

    <!-- An empty state with no action is a dead end (#1967). This one occupied ~85% of the
         viewport with a heading, one sentence, and nothing to do — the joint-lowest-scoring
         surface in the app.
         The ghost card shows the SHAPE of what will live here, at low opacity so it cannot be
         mistaken for real content, and the action underneath is the only thing a person can
         actually do about it: go and listen to something. `aria-hidden` on the ghost so a screen
         reader gets the sentence and the link, not a description of a fake highlight. -->
    <div v-if="!capture.count">
      <p class="text-muted">{{ t('highlights.empty') }}</p>
      <div
        class="mt-4 rounded-2xl border border-border p-4 opacity-40"
        aria-hidden="true"
      >
        <span class="lp-kicker block">{{ t('library.highlights') }}</span>
        <span class="mt-2 block h-3 w-3/4 rounded bg-overlay"></span>
        <span class="mt-2 block h-3 w-1/2 rounded bg-overlay"></span>
      </div>
      <RouterLink
        :to="{ name: 'catalog' }"
        class="mt-4 inline-block text-sm font-bold text-accent no-underline"
      >
        {{ t('highlights.emptyCta') }}
      </RouterLink>
    </div>

    <section v-for="g in visibleGroups" :key="g.slug" class="mb-6">
      <RouterLink
        :to="{ name: 'player', params: { slug: g.slug } }"
        class="lp-section mb-2 block no-underline hover:text-accent"
      >{{ g.title }}</RouterLink>
      <ul class="flex flex-col gap-3">
        <li
          v-for="h in g.highlights"
          :key="h.id"
          class="rounded-xl border border-l-4 border-border p-3"
          :class="borderClass(h.color)"
        >
          <!-- Content is full-width; the controls sit in their own row BELOW it, not in a
               shrink-0 column beside it that squeezed the quote to ~half the row. -->
          <div class="min-w-0">
              <span
                v-if="h.kind !== 'moment'"
                class="lp-kicker"
              >{{ h.kind === 'insight' ? t('highlights.insight') : t('highlights.span') }}</span>
              <p class="text-sm font-semibold leading-snug">{{ label(h) }}</p>
              <p v-if="h.speaker" class="lp-speaker mt-0.5 text-xs">{{ h.speaker }}</p>
              <!-- Graph refs (#1419): the highlight as a node — person/topic it's linked to. -->
              <div v-if="h.graph_refs?.length" class="mt-1 flex flex-wrap gap-1">
                <span
                  v-for="r in h.graph_refs"
                  :key="r.id"
                  class="rounded-full bg-overlay px-2 py-0.5 text-xs"
                  :class="r.kind === 'person' ? 'text-person' : 'text-topic'"
                >{{ r.label }}</span>
              </div>
              <span
                v-if="h.anchor_status === 'drifted'"
                class="mt-1 inline-block rounded-full bg-overlay px-2 py-0.5 text-xs text-danger"
                :title="t('highlights.driftedHint')"
              >⚠ {{ t('highlights.drifted') }}</span>
            </div>
            <div class="mt-2 flex flex-wrap items-center gap-2">
              <RouterLink
                v-if="h.start_ms != null"
                :to="{ name: 'player', params: { slug: h.episode_slug }, query: jumpQuery(h) }"
                class="font-mono text-xs text-accent no-underline"
              >▶ {{ formatTime(h.start_ms / 1000) }}</RouterLink>
              <select
                v-if="collections.length"
                class="max-w-[9rem] rounded-lg border border-border bg-overlay px-1.5 py-1 text-xs"
                :aria-label="t('collections.addTo')"
                @change="addHighlightTo(h.id, ($event.target as HTMLSelectElement).value); ($event.target as HTMLSelectElement).value = ''"
              >
                <option value="">{{ t('collections.addTo') }}</option>
                <option v-for="c in collections" :key="c.id" :value="c.id">{{ c.name }}</option>
              </select>
              <!-- Add a note — same control weight + placement as Add-to-collection (opens the
                   inline editor below), so the two "annotate this" actions read as a pair. -->
              <button
                type="button"
                class="rounded-lg border border-border bg-overlay px-1.5 py-1 text-xs text-muted transition hover:text-accent"
                data-testid="highlight-add-note"
                @click="startAdd(h.id)"
              >+ {{ t('highlights.addNote') }}</button>
              <!-- Colour: the shared collapsed control (one current-colour dot that expands the
                   palette on tap) — identical on every saved surface (#2042). -->
              <SavedColorControl :color="h.color" @pick="capture.setColor(h.id, $event)" />
              <button
                type="button"
                class="rounded-full p-1 text-muted transition hover:text-accent"
                :aria-label="t('highlights.share')"
                :title="t('highlights.share')"
                @click="share(h)"
              >↗</button>
              <button
                type="button"
                class="lp-tap rounded-full p-1 text-muted transition hover:text-danger"
                :aria-label="t('highlights.remove')"
                :title="t('highlights.remove')"
                data-testid="highlight-delete"
                @click="pendingHighlight = h.id"
              >✕</button>
            </div>

          <!-- Notes attached to this highlight -->
          <ul v-if="capture.notesFor('highlight', h.id).length" class="mt-2 flex flex-col gap-1">
            <li
              v-for="n in capture.notesFor('highlight', h.id)"
              :key="n.id"
              class="border-l-2 border-border pl-2 text-sm text-muted"
            >
              <div v-if="editing === n.id">
                <textarea
                  v-model="draft"
                  rows="2"
                  class="w-full rounded border border-border bg-canvas px-2 py-1 text-sm"
                  :aria-label="t('highlights.noteLabel')"
                />
                <div class="mt-1 flex gap-2">
                  <button type="button" class="text-xs font-bold text-accent" @click="save">{{ t('highlights.saveNote') }}</button>
                  <button type="button" class="text-xs text-muted" @click="cancel">{{ t('highlights.cancel') }}</button>
                </div>
              </div>
              <div v-else class="flex items-start justify-between gap-2">
                <span class="min-w-0 flex-1 whitespace-pre-line">{{ n.text }}</span>
                <span class="flex shrink-0 gap-1">
                  <button type="button" class="text-xs text-accent" @click="startEdit(n.id, n.text)">{{ t('highlights.editNote') }}</button>
                  <button
                    type="button"
                    class="text-xs text-muted hover:text-danger"
                    :aria-label="t('highlights.removeNote')"
                    data-testid="note-delete"
                    @click="pendingNote = n.id"
                  >✕</button>
                </span>
              </div>
            </li>
          </ul>

          <!-- Add a new note -->
          <div v-if="editing === `new:${h.id}`" class="mt-2">
            <textarea
              v-model="draft"
              rows="2"
              class="w-full rounded border border-border bg-canvas px-2 py-1 text-sm"
              :aria-label="t('highlights.noteLabel')"
              :placeholder="t('highlights.notePlaceholder')"
            />
            <div class="mt-1 flex gap-2">
              <button type="button" class="text-xs font-bold text-accent" @click="save">{{ t('highlights.saveNote') }}</button>
              <button type="button" class="text-xs text-muted" @click="cancel">{{ t('highlights.cancel') }}</button>
            </div>
          </div>
        </li>
      </ul>
    </section>

    <!-- Cap the number of episode groups shown; a search lifts it (#2042 follow-up). -->
    <ShowAllToggle
      v-if="groupCaps.overflows(groups.length, searchActive)"
      :expanded="groupCaps.expanded.has('groups')"
      :count="groups.length"
      @toggle="groupCaps.toggle('groups')"
    />

    <ConfirmDialog
      :open="pendingHighlight !== null"
      :title="t('highlights.confirmDeleteTitle')"
      :body="t('highlights.confirmDeleteBody')"
      :confirm-label="t('highlights.confirmDelete')"
      data-testid="highlight-delete-confirm"
      @confirm="confirmRemoveHighlight"
      @cancel="pendingHighlight = null"
    />

    <ConfirmDialog
      :open="pendingNote !== null"
      :title="t('highlights.confirmDeleteNoteTitle')"
      :body="t('highlights.confirmDeleteNoteBody')"
      :confirm-label="t('highlights.confirmDeleteNote')"
      data-testid="note-delete-confirm"
      @confirm="confirmRemoveNote"
      @cancel="pendingNote = null"
    />
  </div>

</template>
