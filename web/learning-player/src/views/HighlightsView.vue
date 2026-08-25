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
import { useCaptureStore } from '../stores/capture'
import { formatTime } from '../player/transcriptSync'
import { HIGHLIGHT_COLORS, borderClass } from '../utils/highlightColors'
import { shareHighlightCard } from '../composables/useShareCard'

const { t } = useI18n()
const capture = useCaptureStore()

// Colour filter (PRD-040 FR4.2): null = show all; otherwise only highlights of that colour.
const activeColor = ref<string | null>(null)
function toggleFilter(token: string): void {
  activeColor.value = activeColor.value === token ? null : token
}

// Episode titles for the group headings (slug → title), hydrated lazily; slug is the fallback.
const titles = ref<Record<string, string>>({})

interface Group {
  slug: string
  title: string
  highlights: Highlight[]
}

const groups = computed<Group[]>(() => {
  const bySlug = new Map<string, Highlight[]>()
  for (const h of capture.highlights) {
    if (activeColor.value && h.color !== activeColor.value) continue
    const list = bySlug.get(h.episode_slug) ?? []
    list.push(h)
    bySlug.set(h.episode_slug, list)
  }
  return [...bySlug.entries()].map(([slug, highlights]) => ({
    slug,
    title: titles.value[slug] ?? slug,
    highlights,
  }))
})

function jumpQuery(h: Highlight): Record<string, string> {
  return h.start_ms != null ? { t: String(Math.floor(h.start_ms / 1000)) } : {}
}

function label(h: Highlight): string {
  if (h.kind === 'moment') return t('highlights.moment')
  return h.quote_text ?? t('highlights.span')
}

// --- notes (inline add / edit) ---
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
    const md = await fetchHighlightsExport()
    await saveAndShareText('my-highlights.md', md)
  } finally {
    exporting.value = false
  }
}

// Collections a highlight can be filed into (#1417). Loaded lazily; the per-highlight
// "Add to…" select adds on change then resets to its placeholder.
const collections = ref<Collection[]>([])

async function addHighlightTo(highlightId: string, collectionId: string): Promise<void> {
  if (!collectionId) return
  const updated = await addToCollection(collectionId, highlightId)
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
  collections.value = await getCollections().catch(() => [])
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
      <!-- Native shell: write+share (WKWebView can't `<a download>`); web: plain download link (#1310). -->
      <button
        v-if="isNative()"
        type="button"
        :disabled="exporting"
        class="rounded-full border border-border px-3 py-1 text-sm font-bold text-accent transition hover:bg-overlay disabled:opacity-50"
        @click="exportHighlightsNative"
      >{{ t('highlights.export') }}</button>
      <a
        v-else
        :href="highlightsExportUrl()"
        download="my-highlights.md"
        class="rounded-full border border-border px-3 py-1 text-sm font-bold text-accent no-underline transition hover:bg-overlay"
      >{{ t('highlights.export') }}</a>
      <!-- Graph-aware Obsidian export (#1472) — web only (native zip handling is a follow). -->
      <button
        v-if="!isNative()"
        type="button"
        :disabled="exportingObsidian"
        class="rounded-full border border-border px-3 py-1 text-sm font-bold text-accent transition hover:bg-overlay disabled:opacity-50"
        @click="doObsidianExport"
      >{{ t('highlights.exportObsidian') }}</button>
    </div>
    <p v-if="obsidianMsg" class="mb-1 text-xs text-muted">{{ obsidianMsg }}</p>
    <!--
      Obsidian has no import format to target — a vault IS a folder of Markdown files, so the only
      way in is to put the folder there. Without saying so, the export ends at "here is a zip" and
      the user has to go and find out what to do with it, which is exactly what happened in review.
    -->
    <p v-if="obsidianDone" class="mb-3 text-xs text-muted">{{ t('highlights.obsidianNext') }}</p>

    <!-- Colour filter (FR4.2): tap a swatch to show only that colour; tap again to clear. -->
    <div v-if="capture.count" class="mb-4 flex items-center gap-2">
      <span class="text-xs text-muted">{{ t('highlights.filterByColor') }}</span>
      <button
        v-for="c in HIGHLIGHT_COLORS"
        :key="c.token"
        type="button"
        class="h-4 w-4 rounded-full ring-offset-1 ring-offset-canvas transition"
        :class="[c.swatch, activeColor === c.token ? 'ring-2 ring-accent' : 'hover:ring-1 hover:ring-border']"
        :aria-pressed="activeColor === c.token"
        :aria-label="t('highlights.filterColor', { color: t(c.labelKey) })"
        :title="t(c.labelKey)"
        @click="toggleFilter(c.token)"
      />
      <button
        v-if="activeColor"
        type="button"
        class="text-xs text-accent"
        @click="activeColor = null"
      >{{ t('highlights.clearFilter') }}</button>
    </div>

    <p v-if="!capture.count" class="text-muted">{{ t('highlights.empty') }}</p>

    <section v-for="g in groups" :key="g.slug" class="mb-6">
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
              <button
                type="button"
                class="rounded-full p-1 text-muted transition hover:text-accent"
                :aria-label="t('highlights.share')"
                :title="t('highlights.share')"
                @click="share(h)"
              >↗</button>
              <button
                type="button"
                class="rounded-full p-1 text-muted transition hover:text-danger"
                :aria-label="t('highlights.remove')"
                :title="t('highlights.remove')"
                @click="capture.remove(h.id)"
              >✕</button>
            </div>

          <!-- Colour swatches (FR1.4): tap to set; tap the active one to clear. -->
          <div class="mt-2 flex items-center gap-1.5">
            <button
              v-for="c in HIGHLIGHT_COLORS"
              :key="c.token"
              type="button"
              class="h-3.5 w-3.5 rounded-full ring-offset-1 ring-offset-surface transition"
              :class="[c.swatch, h.color === c.token ? 'ring-2 ring-accent' : 'opacity-60 hover:opacity-100']"
              :aria-pressed="h.color === c.token"
              :aria-label="t('highlights.setColor', { color: t(c.labelKey) })"
              :title="t(c.labelKey)"
              @click="capture.setColor(h.id, h.color === c.token ? null : c.token)"
            />
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
                  <button type="button" class="text-xs text-muted hover:text-danger" :aria-label="t('highlights.removeNote')" @click="capture.removeNote(n.id)">✕</button>
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
          <button
            v-else
            type="button"
            class="mt-2 text-xs font-bold text-accent"
            @click="startAdd(h.id)"
          >+ {{ t('highlights.addNote') }}</button>
        </li>
      </ul>
    </section>
  </div>
</template>
