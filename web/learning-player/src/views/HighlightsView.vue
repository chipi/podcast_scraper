<script setup lang="ts">
/**
 * Highlights review (P2 Capture, PRD-040) — the user's captured moments / spans / saved insights,
 * grouped by episode, each with jump-to-moment, inline notes, and delete. A Markdown export link
 * sits in the header (the single export format, REMEMBER-half-scope §4). Embedded in the Library
 * "Highlights" tab. Auth-gated (the store no-ops + stays empty when signed out).
 */
import { computed, onMounted, ref } from 'vue'
import BellOffIcon from "../components/BellOffIcon.vue"
import BookmarkIcon from "../components/BookmarkIcon.vue"
import CloseIcon from "../components/CloseIcon.vue"
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import {
  addToCollection,
  exportObsidian,
  fetchHighlightsExport,
  getCollections,
  getEpisode,
  highlightsExportUrl,
  highlightsPrintUrl,
} from '../services/api'
import type { Collection } from '../services/types'
import { isNative, openExternal, saveAndShareBinary, saveAndShareText } from '../services/native'
import type { EpisodeDetail, EpisodeSummary, Highlight } from '../services/types'
import ConfirmDialog from '../components/ConfirmDialog.vue'
import SavedColorControl from '../components/SavedColorControl.vue'
import EpisodeRow from '../components/EpisodeRow.vue'
import ShowAllToggle from '../components/ShowAllToggle.vue'
import { useCaptureStore } from '../stores/capture'
import { formatTime } from '../player/transcriptSync'
import { borderClass } from '../utils/highlightColors'
import { summaryFromDetail } from '../utils/episode'
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
const props = defineProps<{
  filterColor?: string | null
  sort?: string
  search?: string
  /** `true` = only captures the user stopped resurfacing; default/false = everything. */
  mutedOnly?: boolean
}>()

// Episode groups are capped like every other Library section (#2042 follow-up); a search lifts it.
// Episode groups AND the captures inside each one page 10 at a time (operator 2026-09-18). A heavy
// listener has dozens of captures on a single episode, and "Show all" on that is not a page — it is
// a scroll with no landmarks. Separate instances so walking one episode does not move the others.
const groupCaps = useCappedSections(10, 10)
const itemCaps = useCappedSections(10, 10)

/**
 * Episode groups the user has folded away (operator 2026-09-18).
 *
 * Collapsed-by-exception rather than open-by-exception: the captures are the content, so hiding
 * them has to be a choice. Presentation-only and per-view — a fold is tidying, not a preference
 * worth persisting across sessions.
 */
const collapsed = ref<Set<string>>(new Set())
function toggleGroup(slug: string): void {
  const next = new Set(collapsed.value)
  if (!next.delete(slug)) next.add(slug)
  collapsed.value = next
}
const searchActive = computed(() => (props.search ?? '').trim() !== '')

/**
 * The episode behind each group heading (slug → detail), hydrated lazily.
 *
 * Whole detail rather than just the title: the heading carries the episode's ARTWORK now (operator
 * 2026-09-18, matching Revisit and the downloads rows), and the detail this view already fetched
 * for the title carries it — so the thumbnail costs no extra request.
 */
const details = ref<Record<string, EpisodeDetail>>({})
/** Title with the slug as fallback, so a group still reads sensibly before its episode lands. */
const titleFor = (slug: string): string => details.value[slug]?.title ?? slug

/**
 * The group heading's episode, in the shape `EpisodeRow` takes.
 *
 * Through `summaryFromDetail` — the one adapter Queue, Recent and Revisit use — so the heading
 * cannot drift from the rows it is modelled on. Unresolved episodes still render a row, titled by
 * slug: a heading that vanished until a fetch landed would make groups appear to pop into
 * existence.
 */
function headingEpisode(slug: string): EpisodeSummary {
  const d = details.value[slug]
  if (d) return summaryFromDetail(d)
  return {
    slug,
    title: slug,
    podcast_title: null,
    feed_id: null,
    publish_date: null,
    duration_seconds: null,
    artwork_url: null,
    episode_image_url: null,
    feed_image_url: null,
    status: 'ready',
  } as unknown as EpisodeSummary
}

interface Group {
  slug: string
  title: string
  highlights: Highlight[]
}

const byRecent = (a: Highlight, b: Highlight): number => (b.created_at ?? 0) - (a.created_at ?? 0)

const groups = computed<Group[]>(() => {
  const sort = props.sort ?? 'recent'
  const query = props.search ?? ''
  const bySlug = new Map<string, Highlight[]>()
  for (const h of capture.highlights) {
    if (props.filterColor && h.color !== props.filterColor) continue
    // Muted filter (operator 2026-09-18): sits beside the colour filter and reads the same way
    // — unset shows everything, set narrows. `retired` is optional on the type, so the coerce
    // keeps an older cached payload (no field) out of the muted bucket rather than in it.
    if (props.mutedOnly && !h.retired) continue
    // Search matches a highlight's own text (quote / speaker); episode titles are findable through
    // the Episodes section. Mirrors LibraryView's count predicate so the two agree.
    if (!(matchesQuery(h.quote_text, query) || matchesQuery(h.speaker, query))) continue
    const list = bySlug.get(h.episode_slug) ?? []
    list.push(h)
    bySlug.set(h.episode_slug, list)
  }
  // Highlights stay grouped by episode (structural); the shared sort only orders things. Within a
  // group, newest first. Group ORDER: A–Z by episode title for 'title', else most-recent group first.
  const out = [...bySlug.entries()].map(([slug, highlights]) => ({
    slug,
    title: titleFor(slug),
    highlights: [...highlights].sort(byRecent),
  }))
  if (sort === 'title') {
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

// The count beside the header reflects what the active colour/search filter actually shows, not the
// raw total — else "12 highlights" sat above a list of 2 under a filter (Fable-5 review nit).
const shownCount = computed(() => groups.value.reduce((n, g) => n + g.highlights.length, 0))

function jumpQuery(h: Highlight): Record<string, string> {
  return h.start_ms != null ? { t: String(Math.floor(h.start_ms / 1000)) } : {}
}

/** The captured words, or '' when a moment saved before this stored only a timestamp. */
function quoteOf(h: Highlight): string {
  return h.quote_text?.trim() ?? ''
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
    const md = await fetchHighlightsExport(props.filterColor, {
      mutedOnly: props.mutedOnly,
      q: props.search,
    })
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
/** Undo a retire. Only reachable from a row that IS retired, so there is no toggle to reason about. */
/**
 * Open the print-styled export so the browser can save it as PDF.
 *
 * Carries the SAME filters as the other formats — it is the same document, one route along.
 */
async function openPrintable(): Promise<void> {
  await openExternal(
    highlightsPrintUrl(props.filterColor, { mutedOnly: props.mutedOnly, q: props.search }),
  )
}

async function resume(id: string): Promise<void> {
  await capture.unretire(id)
}

async function share(h: Highlight): Promise<void> {
  await shareHighlightCard(h, titleFor(h.episode_slug))
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
    const r = await exportObsidian(0, undefined, isNative()
      ? (blob) => saveAndShareBinary('closelistening-obsidian.zip', blob)
      : undefined)
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
      if (d) details.value[slug] = d
    }),
  )
})
</script>

<template>
  <div>
    <div v-if="capture.count" class="mb-4 flex items-center justify-between gap-3">
      <p class="text-sm text-muted">{{ t('highlights.count', shownCount, { named: { count: shownCount } }) }}</p>
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
          :href="highlightsExportUrl(filterColor, { mutedOnly, q: search })"
          download="my-highlights.md"
          :aria-label="t('highlights.export')"
          class="whitespace-nowrap rounded-full border border-border px-2.5 py-1 text-xs font-bold text-accent no-underline transition hover:bg-overlay"
        >{{ t('highlights.exportMarkdownShort') }}</a>
        <!-- PDF, via the browser's own print-to-PDF (operator 2026-09-18). No PDF library: the
             renderer is already in the browser, and "Print -> Save as PDF" is native everywhere we
             ship, including the iOS share sheet. The server returns the SAME export document with a
             print stylesheet (`export.html`) and the browser converts it.

             Opened in a new tab rather than printed from a hidden iframe: the user needs to SEE
             what they are about to print, and a print dialog fired from an invisible frame with no
             preview is indistinguishable from the app having hijacked the printer. -->
        <button
          type="button"
          :aria-label="t('highlights.exportPdf')"
          data-testid="export-pdf"
          class="whitespace-nowrap rounded-full border border-border px-2.5 py-1 text-xs font-bold text-accent transition hover:bg-overlay"
          @click="openPrintable"
        >{{ t('highlights.exportPdfShort') }}</button>
        <!-- Graph-aware Obsidian export (#1472). Was `v-if="!isNative()"` — the zip reached the
             device through `<a download>`, which WKWebView ignores, so rather than fix the delivery
             the button was hidden and the feature just disappeared on the phone (operator
             2026-09-18). Native now takes the same bytes through the share sheet. -->
        <button
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
    <p v-if="obsidianDone" class="mb-3 text-xs text-muted">
      {{ isNative() ? t('highlights.obsidianNextNative') : t('highlights.obsidianNext') }}
    </p>

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
      <!-- The episode heads its own group as the SHARED compact row (operator 2026-09-18) — the
           same `EpisodeRow` Home's What's New 2-5, the entity cards and the storyline sheet use.
           It already is the standard small view: 40px artwork, title, and the show name under it,
           which is the part a text-only heading was missing. Hand-rolling an img + title here was
           a fourth near-copy of a row that already exists.

           Before the episode resolves, the row still renders with the slug as its title rather
           than the group disappearing or reserving a grey box for something that may not arrive. -->
      <div class="mb-2" data-testid="highlight-group-heading">
        <EpisodeRow :episode="headingEpisode(g.slug)">
          <!-- Collapse the episode (operator 2026-09-18), in the row's own `#trailing` slot so the
               control is a SIBLING of the link rather than nested inside it — an interactive inside
               an interactive is the thing EpisodeRow's slot exists to avoid. Groups start open:
               collapsing is for tidying a long Saved list, not a default that hides your captures. -->
          <template #trailing>
            <!-- `self-center`: EpisodeRow's row is `items-start` (correct for artwork beside two
                 lines of text), which pinned this control to the top corner. It acts on the whole
                 row, so it centres against it. -->
            <button
              type="button"
              class="lp-tap shrink-0 self-center rounded-full px-2 py-1 text-xs font-bold text-accent"
              :aria-expanded="!collapsed.has(g.slug)"
              :aria-label="
                collapsed.has(g.slug)
                  ? t('highlights.expandGroup', { title: g.title })
                  : t('highlights.collapseGroup', { title: g.title })
              "
              data-testid="highlight-group-collapse"
              @click="toggleGroup(g.slug)"
            >{{ collapsed.has(g.slug) ? '▼' : '▲' }}</button>
          </template>
        </EpisodeRow>
      </div>
      <ul v-show="!collapsed.has(g.slug)" class="flex flex-col gap-3">
        <li
          v-for="h in itemCaps.visible(g.slug, g.highlights, searchActive)"
          :key="h.id"
          class="rounded-xl border border-l-4 border-border p-3"
          :class="borderClass(h.color)"
        >
          <!-- Content is full-width; the controls sit in their own row BELOW it, not in a
               shrink-0 column beside it that squeezed the quote to ~half the row. -->
          <div class="min-w-0">
              <!-- TITLE — what kind of capture this is. A moment says so too now: it used to be
                   the only kind with no kicker, because the words "Marked moment" were standing in
                   as the body text (operator 2026-09-17). -->
              <!-- The kicker and the per-card icon actions share ONE line, actions hard right
                   (operator 2026-09-18). They used to wrap onto a second row beneath the controls,
                   giving every card a trailing strip of three lonely glyphs and making a two-line
                   capture three lines tall. The kicker line was half empty; this is space the card
                   already had. -->
              <div class="flex items-start justify-between gap-2">
                <span class="flex min-w-0 flex-wrap items-center gap-2">
                  <span class="lp-kicker">{{
                    h.kind === 'insight'
                      ? t('highlights.insight')
                      : h.kind === 'span'
                        ? t('highlights.span')
                        : t('highlights.moment')
                  }}</span>
                  <!-- Drift sits BESIDE the kind (operator 2026-09-18) — both are facts about what
                       this capture IS, so they belong on the same line. Under the speaker it read
                       as a comment on the quote instead, and pushed the card a line taller. -->
                  <span
                    v-if="h.anchor_status === 'drifted'"
                    class="rounded-full bg-overlay px-2 py-0.5 text-xs text-danger"
                    :title="t('highlights.driftedHint')"
                    data-testid="highlight-drifted"
                  >⚠ {{ t('highlights.drifted') }}</span>
                </span>
                <div class="-mt-1 flex shrink-0 items-center gap-1">
                  <!-- ORDER: colour first, share last (operator 2026-09-18). Colour is what
                       this capture IS, so it leads; share sends it somewhere else, so it
                       trails. The state and unsave controls sit between, bell before
                       bookmark — the same order as the Revisit card. -->
                  <!-- Colour: the shared collapsed control (one current-colour dot that expands the
                       palette on tap) — identical on every saved surface (#2042). -->
                  <SavedColorControl :color="h.color" @pick="capture.setColor(h.id, $event)" />
                  <!-- RETIRED: shown ONLY when it is (operator 2026-09-18) — "by default, things
                       are not quiet", so a not-retired capture carries no badge and the row is
                       unchanged for the overwhelming majority.

                       This exists because retiring was otherwise a one-way door. Stopping a
                       capture resurfacing removes it from Revisit, which makes Revisit the one
                       place the undo CANNOT live; Saved is the only surface listing every capture,
                       so it is where the state has to be visible and reversible. The icon is the
                       same bell-with-slash pressed on the Revisit card — pressing it again undoes
                       exactly what that press did. -->
                  <button
                    v-if="h.retired"
                    type="button"
                    class="lp-tap flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-accent text-accent transition hover:bg-accent/10"
                    :aria-label="t('highlights.resumeResurfacing')"
                    :title="t('highlights.resumeResurfacing')"
                    data-testid="highlight-retired"
                    @click="resume(h.id)"
                  ><BellOffIcon /></button>
                  <!-- The FILLED bookmark, not a ✕ (operator 2026-09-18) — the same control, and the
                       same reasoning, as the Revisit card's third outcome. This action UNSAVES, so
                       it shows the glyph that did the saving, filled: tapping it reads as undoing
                       the save rather than as a generic destroy.

                       Identical here and on Revisit deliberately. These are the two surfaces that
                       list the same objects, so an unsave that looked like ✕ on one and a bookmark
                       on the other would be two controls for one action. Accent at rest (the saved
                       state it shows), danger on hover (what pressing it does), and still
                       confirm-gated (#1594) — the capture and its notes do not come back. -->
                  <button
                    type="button"
                    class="lp-tap rounded-full p-1 text-accent transition hover:text-danger"
                    :aria-label="t('highlights.unsave')"
                    :title="t('highlights.unsave')"
                    data-testid="highlight-delete"
                    @click="pendingHighlight = h.id"
                  ><BookmarkIcon filled /></button>
                  <button
                    type="button"
                    class="rounded-full p-1 text-muted transition hover:text-accent"
                    :aria-label="t('highlights.share')"
                    :title="t('highlights.share')"
                    @click="share(h)"
                  >↗</button>
                </div>
              </div>
              <!-- The QUOTE — the spoken line that was captured, under the title and above the
                   speaker who said it. Set as a quotation rather than a heading: these are somebody
                   else's words and the card is the record of them. A moment saved before the text
                   was captured has none, and shows nothing here rather than a placeholder
                   pretending to be a quote. -->
              <blockquote
                v-if="quoteOf(h)"
                class="mt-1 border-l-2 border-border pl-2 text-sm italic leading-snug text-canvas-foreground"
                data-testid="highlight-quote"
              >
                {{ quoteOf(h) }}
              </blockquote>
              <p v-if="h.speaker" class="lp-speaker mt-1 text-xs">{{ h.speaker }}</p>
              <!-- Graph refs (#1419): the highlight as a node — person/topic it's linked to. -->
              <!-- The person/topic pills that sat here are gone (operator 2026-09-17): the card is a
                   captured moment, and a row of entity chips under it repeated what the transcript
                   already says while pushing the text itself down. -->
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
                  ><CloseIcon /></button>
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
      <!-- Captures WITHIN this episode page 10 at a time, keyed by slug so each episode is walked
           independently. A search lifts it, same rule as everywhere else. -->
      <ShowAllToggle
        v-if="!collapsed.has(g.slug) && itemCaps.overflows(g.highlights.length, searchActive, g.slug)"
        :expanded="itemCaps.remaining(g.slug, g.highlights.length) === 0"
        :count="g.highlights.length"
        :remaining="itemCaps.remaining(g.slug, g.highlights.length)"
        @toggle="itemCaps.toggle(g.slug, g.highlights.length)"
      />
    </section>

    <!-- Episode groups page 10 at a time; a search lifts it (#2042 follow-up). -->
    <ShowAllToggle
      v-if="groupCaps.overflows(groups.length, searchActive, 'groups')"
      :expanded="groupCaps.remaining('groups', groups.length) === 0"
      :count="groups.length"
      :remaining="groupCaps.remaining('groups', groups.length)"
      @toggle="groupCaps.toggle('groups', groups.length)"
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
