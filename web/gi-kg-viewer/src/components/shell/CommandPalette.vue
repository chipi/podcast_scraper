<script setup lang="ts">
/**
 * Command palette — Search v3 §S3 (RFC-107 §4). Shell-wide overlay
 * summonable from any tab via Cmd-K / Ctrl-K / `/`. Live-queries
 * ``/api/search`` and offers 3 actions per hit:
 *
 *   * Open in Workspace — switches main tab to 'search' and runs the query
 *     via the shared useSearchStore.
 *   * Pin to rail — routes the hit to the appropriate subject.focus* helper
 *     via useSearchFocus utilities (episode / person / topic).
 *   * Show on graph — reuses the same graph-focus path Search results use
 *     today (graph node id from the hit's source_id).
 *
 * Empty state (query cleared): shows Recent (USERPREFS-1
 * ``search.recentQueries``) and Saved (USERPREFS-1 ``search.savedQueries`` —
 * populated in slice S7 #1237).
 *
 * Not in this slice: the ``palette=1`` server-side hint on /api/search
 * (would cap response cost) — S4/S5 can add it if palette latency is a
 * concern. The current shipped endpoint returns the same shape.
 */
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'
import { searchCorpus, type SearchHit } from '../../api/searchApi'
import { fetchCorpusEpisodes, type CorpusEpisodeListItem } from '../../api/corpusLibraryApi'
import { useShellStore } from '../../stores/shell'
import { useSearchStore } from '../../stores/search'
import { useSubjectStore } from '../../stores/subject'
import { useUserPreferencesStore } from '../../stores/userPreferences'
import { useSavedQueriesStore } from '../../stores/savedQueries'
import { useThemeStore } from '../../stores/theme'
import { useAuthStore } from '../../stores/auth'
import {
  episodeFallbackForSearchHit,
  graphNodeIdFromSearchHit,
} from '../../utils/searchFocus'
import { copyTextToClipboard } from '../../utils/clipboard'
import {
  buildPaletteCommands,
  matchCommands,
  stripCommandPrefix,
  type MainTabId,
  type PaletteCommand,
} from '../../utils/paletteCommands'

interface RecentEntry {
  q: string
  ts?: number
}

const emit = defineEmits<{
  'open-in-workspace': [q: string]
  'show-on-graph': [cyId: string]
  /** #1259-1 command mode — switch main tab from a palette command row. */
  'go-tab': [tab: MainTabId]
  /** #1259-1 command mode — parent opens the shell's Configuration dialog. */
  'open-configuration': []
  /** #1259-1 command mode — parent opens the shell's Health dialog. */
  'open-health': []
  /** #1259-1 command mode — parent triggers a corpus re-index (admin only). */
  'rebuild-index': []
}>()

const shell = useShellStore()
const search = useSearchStore()
const subject = useSubjectStore()
const userPrefs = useUserPreferencesStore()
const savedQueries = useSavedQueriesStore()
const theme = useThemeStore()
const auth = useAuthStore()

const open = ref(false)
const query = ref('')
const results = ref<SearchHit[]>([])
const episodeSuggestions = ref<CorpusEpisodeListItem[]>([])
const loading = ref(false)
const error = ref<string | null>(null)
const inputRef = ref<HTMLInputElement | null>(null)
const previousFocus = ref<HTMLElement | null>(null)

const recentQueries = computed<RecentEntry[]>(() => {
  const raw = userPrefs.get<RecentEntry[]>('search.recentQueries')
  return Array.isArray(raw) ? raw.slice(0, 8) : []
})

const showEmptyState = computed(() => open.value && !query.value.trim() && !loading.value)

// ---------- #1259-1: command mode (nav + session actions) ---------------

async function runCopyCorpusPath(): Promise<void> {
  const path = shell.corpusPath.trim()
  if (!path) return
  await copyTextToClipboard(path)
}

function runSaveCurrentQuery(): void {
  const q = search.query.trim()
  if (!q) return
  void savedQueries.saveQuery(q)
}

function runClearSearch(): void {
  search.query = ''
  search.clearResults()
}

function runResetFilters(): void {
  search.filters.topK = 10
  search.filters.groundedOnly = false
  search.filters.feed = ''
  search.filters.since = ''
  search.filters.speaker = ''
  search.filters.types = []
  search.filters.embeddingModel = ''
  search.filters.dedupeKgSurfaces = true
  search.filters.topic = ''
  search.filters.minConfidence = ''
  search.filters.episodeId = ''
  search.filters.enrichResults = null
}

const paletteCommands = computed<PaletteCommand[]>(() =>
  buildPaletteCommands({
    goTab: (tab) => emit('go-tab', tab),
    saveCurrentQuery: runSaveCurrentQuery,
    clearSearch: runClearSearch,
    resetFilters: runResetFilters,
    cycleTheme: () => theme.cycle(),
    copyCorpusPath: runCopyCorpusPath,
    openConfiguration: () => emit('open-configuration'),
    openHealth: () => emit('open-health'),
    rebuildIndex: () => emit('rebuild-index'),
    isAdmin: auth.isAdmin,
  }),
)

/**
 * When the input starts with ``>`` the palette is in COMMAND MODE — only
 * commands render, corpus-search is skipped. Otherwise the palette runs
 * a normal live-fetch AND shows top-N command matches ABOVE the hit list
 * when the query fuzz-matches a command. Selecting a command replaces
 * the query.
 */
const parsedInput = computed(() => stripCommandPrefix(query.value))
const commandModeActive = computed(() => parsedInput.value.isCommandMode)
const matchedCommands = computed<PaletteCommand[]>(() => {
  if (!open.value) return []
  const { isCommandMode, query: sub } = parsedInput.value
  if (isCommandMode) return matchCommands(paletteCommands.value, sub, { limit: 20 })
  if (!sub) return []
  return matchCommands(paletteCommands.value, sub, { limit: 3 })
})

async function runCommand(cmd: PaletteCommand): Promise<void> {
  await cmd.run()
  closePalette()
}

// ---------- #1259-2: subject jump ---------------------------------------

/**
 * Split the /api/search hit page into "subject" rows (kg_topic /
 * kg_entity — one-click open the subject panel) and "content" rows
 * (insight / quote / transcript / summary_* — keep the 3-action row so
 * the user can Open in Workspace or Show on graph).
 *
 * Rendered as a dedicated section ABOVE content hits so the "find a
 * thing by name" workflow gets first priority when the query looks
 * like an entity label. Episodes come from a separate corpus-library
 * fetch (title / summary / bullets substring, server-side).
 */
const subjectHits = computed<SearchHit[]>(() => {
  const out: SearchHit[] = []
  const seen = new Set<string>()
  for (const hit of results.value) {
    const dt = String(hit.metadata?.doc_type ?? '')
    const src = String(hit.metadata?.source_id ?? '')
    if ((dt !== 'kg_topic' && dt !== 'kg_entity') || !src) continue
    const key = `${dt}:${src}`
    if (seen.has(key)) continue
    seen.add(key)
    out.push(hit)
    if (out.length >= 5) break
  }
  return out
})

const contentHits = computed<SearchHit[]>(() =>
  results.value.filter((h) => {
    const dt = String(h.metadata?.doc_type ?? '')
    return dt !== 'kg_topic' && dt !== 'kg_entity'
  }),
)

function subjectLabelFor(hit: SearchHit): string {
  const md = hit.metadata ?? {}
  const explicit = md.topic_label ?? md.entity_name ?? md.label
  if (typeof explicit === 'string' && explicit.trim()) return explicit.trim()
  return hit.text?.trim() || String(md.source_id ?? '(unnamed)')
}

function subjectKindLabelFor(hit: SearchHit): string {
  return hit.metadata?.doc_type === 'kg_topic' ? 'Topic' : 'Person'
}

function onSubjectHitClick(hit: SearchHit): void {
  const md = hit.metadata ?? {}
  const src = typeof md.source_id === 'string' ? md.source_id.trim() : ''
  if (!src) return
  if (md.doc_type === 'kg_topic') subject.focusTopic(src)
  else if (md.doc_type === 'kg_entity') subject.focusPerson(src)
  closePalette()
}

function onEpisodeSuggestionClick(item: CorpusEpisodeListItem): void {
  const rel = item.metadata_relative_path
  if (!rel) return
  subject.focusEpisode(rel)
  closePalette()
}

function episodeSuggestionLabel(item: CorpusEpisodeListItem): string {
  return (item.episode_title || item.summary_title || 'Untitled episode').trim()
}

function episodeSuggestionSubtitle(item: CorpusEpisodeListItem): string {
  const feed = item.feed_display_title?.trim() || ''
  const date = item.publish_date?.trim() || ''
  return [feed, date].filter(Boolean).join(' · ')
}

// Debounce timer for live search.
let debounceHandle: ReturnType<typeof setTimeout> | null = null
const DEBOUNCE_MS = 200

function schedule(): void {
  if (debounceHandle !== null) clearTimeout(debounceHandle)
  const q = query.value.trim()
  if (!q) {
    results.value = []
    error.value = null
    return
  }
  // #1259-1: command mode short-circuits the corpus fetch.
  if (parsedInput.value.isCommandMode) {
    results.value = []
    episodeSuggestions.value = []
    error.value = null
    loading.value = false
    return
  }
  const root = shell.corpusPath.trim()
  if (!root) {
    error.value = 'Set a corpus path first (status bar).'
    results.value = []
    return
  }
  loading.value = true
  debounceHandle = setTimeout(async () => {
    try {
      const [body, episodesEnvelope] = await Promise.all([
        searchCorpus(q, { path: root, topK: 8 }),
        // #1259-2: episode subject-jump — server-side substring match on
        // summary title / bullets. 5 is enough to fill the section; the
        // list is already sort=newest and the palette is a preview
        // surface, not a browser.
        fetchCorpusEpisodes(root, { q, limit: 5 }).catch(() => null),
      ])
      if (body.error) {
        error.value = body.error
        results.value = []
      } else {
        error.value = null
        results.value = body.results
      }
      episodeSuggestions.value = episodesEnvelope?.items ?? []
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
      results.value = []
      episodeSuggestions.value = []
    } finally {
      loading.value = false
    }
  }, DEBOUNCE_MS)
}

watch(query, () => schedule())

function openPalette(): void {
  if (open.value) return
  previousFocus.value = (document.activeElement as HTMLElement | null) ?? null
  open.value = true
  void nextTick(() => inputRef.value?.focus())
}

function closePalette(): void {
  if (!open.value) return
  open.value = false
  query.value = ''
  results.value = []
  episodeSuggestions.value = []
  error.value = null
  if (debounceHandle !== null) {
    clearTimeout(debounceHandle)
    debounceHandle = null
  }
  // Restore focus to the element that opened the palette (a11y).
  const prev = previousFocus.value
  if (prev && typeof prev.focus === 'function') {
    void nextTick(() => prev.focus())
  }
}

function onBackdropClick(ev: MouseEvent): void {
  // Only close when clicking the backdrop itself, not children.
  if (ev.target === ev.currentTarget) closePalette()
}

function onKeydown(ev: KeyboardEvent): void {
  if (ev.key === 'Escape') {
    ev.preventDefault()
    closePalette()
  }
}

function openInWorkspace(q: string): void {
  const term = q.trim()
  if (!term) return
  search.query = term
  emit('open-in-workspace', term)
  closePalette()
  // The parent switches main tab; runSearch is triggered there once the
  // shell path is confirmed.
}

function pinToRail(hit: SearchHit): void {
  // Route hit to the appropriate subject.focus* helper.
  const md = hit.metadata ?? {}
  const docType = md.doc_type as string | undefined
  if (docType === 'kg_topic' && typeof md.source_id === 'string') {
    subject.focusTopic(md.source_id)
  } else if (docType === 'kg_entity' && typeof md.source_id === 'string') {
    // Person / entity — focusPerson accepts the person: id.
    subject.focusPerson(md.source_id)
  } else {
    // Fall back to episode focus via metadata_relative_path.
    const rel = episodeFallbackForSearchHit(hit)
    if (rel) subject.focusEpisode(rel)
  }
  closePalette()
}

function showOnGraph(hit: SearchHit): void {
  const cy = graphNodeIdFromSearchHit(hit)
  if (!cy) return
  emit('show-on-graph', cy)
  closePalette()
}

function useRecent(q: string): void {
  query.value = q
  void nextTick(() => inputRef.value?.focus())
}

defineExpose({ open: openPalette, close: closePalette })

// Cleanup on unmount so a lingering timeout doesn't leak into a subsequent
// mount (e.g. during hot-reload).
onBeforeUnmount(() => {
  if (debounceHandle !== null) {
    clearTimeout(debounceHandle)
    debounceHandle = null
  }
})
</script>

<template>
  <Teleport to="body">
    <div
      v-if="open"
      class="fixed inset-0 z-[60] flex items-start justify-center bg-canvas/70 pt-24 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-label="Command palette"
      data-testid="command-palette"
      @click="onBackdropClick"
      @keydown="onKeydown"
    >
      <div
        class="w-full max-w-xl overflow-hidden rounded-lg border border-border bg-elevated shadow-lg"
      >
        <input
          ref="inputRef"
          v-model="query"
          type="text"
          placeholder="Search anywhere — Cmd-K, /, or type > for commands"
          class="w-full border-b border-border bg-transparent px-4 py-3 text-sm text-elevated-foreground outline-none placeholder:text-muted"
          data-testid="command-palette-input"
          aria-label="Search query"
        />

        <div class="max-h-[24rem] overflow-y-auto p-2">
          <!-- #1259-1: Commands (nav + session actions) — visible in command mode
               and as a compact top strip when the user's plain query matches. -->
          <section
            v-if="matchedCommands.length"
            class="mb-2"
            aria-labelledby="palette-commands-heading"
            data-testid="command-palette-commands"
          >
            <h3
              id="palette-commands-heading"
              class="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted"
            >
              Commands
            </h3>
            <ul class="flex flex-col gap-0.5">
              <li v-for="cmd in matchedCommands" :key="cmd.id">
                <button
                  type="button"
                  class="flex w-full items-center justify-between rounded px-2 py-1 text-left text-xs text-elevated-foreground hover:bg-overlay"
                  :data-testid="`command-palette-command-${cmd.id}`"
                  :data-command-id="cmd.id"
                  @click="runCommand(cmd)"
                >
                  <span class="flex items-center gap-2">
                    <span
                      class="inline-flex h-4 min-w-[1rem] items-center justify-center rounded bg-primary/15 px-1 text-[9px] font-medium uppercase tracking-wide text-primary"
                    >{{ cmd.category === 'nav' ? 'Go' : cmd.category === 'admin' ? 'Adm' : cmd.category === 'modal' ? 'Open' : 'Act' }}</span>
                    <span class="truncate">{{ cmd.label }}</span>
                  </span>
                  <span
                    v-if="cmd.shortcut"
                    class="ml-2 shrink-0 rounded border border-border px-1 font-mono text-[10px] text-muted"
                  >{{ cmd.shortcut }}</span>
                </button>
              </li>
            </ul>
          </section>

          <!-- Empty state — Recent + Saved (only when NOT command mode) -->
          <div v-if="showEmptyState && !commandModeActive" class="space-y-3 p-2">
            <section aria-labelledby="palette-recent-heading">
              <h3
                id="palette-recent-heading"
                class="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted"
              >
                Recent
              </h3>
              <ul
                v-if="recentQueries.length"
                data-testid="command-palette-recent-list"
                class="flex flex-col gap-0.5"
              >
                <li v-for="(r, i) in recentQueries" :key="`${r.q}-${r.ts ?? i}`">
                  <button
                    type="button"
                    class="w-full truncate rounded px-2 py-1 text-left text-xs text-elevated-foreground hover:bg-overlay"
                    :title="r.q"
                    @click="useRecent(r.q)"
                  >
                    {{ r.q }}
                  </button>
                </li>
              </ul>
              <p
                v-else
                class="text-xs text-muted"
                data-testid="command-palette-recent-empty"
              >
                No recent queries yet.
              </p>
            </section>

            <section aria-labelledby="palette-saved-heading">
              <h3
                id="palette-saved-heading"
                class="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted"
              >
                Saved
              </h3>
              <p
                class="text-xs text-muted"
                data-testid="command-palette-saved-empty"
              >
                Saved queries land in slice S7 (USERPREFS-1 ``search.savedQueries``).
              </p>
            </section>
          </div>

          <!-- Loading / error / results (suppressed in command mode) -->
          <p
            v-if="loading && !commandModeActive"
            class="p-2 text-xs text-muted"
            data-testid="command-palette-loading"
          >
            Searching…
          </p>
          <p
            v-else-if="error && !commandModeActive"
            class="p-2 text-xs text-danger"
            data-testid="command-palette-error"
          >
            {{ error }}
          </p>
          <!-- #1259-2: Subject jump — episodes (from corpus-library) + topics/persons (from search) -->
          <section
            v-if="!commandModeActive && episodeSuggestions.length"
            class="mb-2"
            aria-labelledby="palette-episodes-heading"
            data-testid="command-palette-episodes"
          >
            <h3
              id="palette-episodes-heading"
              class="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted"
            >
              Episodes
            </h3>
            <ul class="flex flex-col gap-0.5">
              <li v-for="ep in episodeSuggestions" :key="ep.metadata_relative_path">
                <button
                  type="button"
                  class="flex w-full items-center justify-between gap-2 rounded px-2 py-1 text-left text-xs text-elevated-foreground hover:bg-overlay"
                  data-testid="command-palette-episode-suggestion"
                  @click="onEpisodeSuggestionClick(ep)"
                >
                  <span class="min-w-0 flex-1">
                    <span class="block truncate">{{ episodeSuggestionLabel(ep) }}</span>
                    <span
                      v-if="episodeSuggestionSubtitle(ep)"
                      class="block truncate text-[10px] text-muted"
                    >{{ episodeSuggestionSubtitle(ep) }}</span>
                  </span>
                  <span
                    class="inline-flex h-4 shrink-0 items-center justify-center rounded bg-primary/15 px-1 text-[9px] font-medium uppercase tracking-wide text-primary"
                  >Episode</span>
                </button>
              </li>
            </ul>
          </section>

          <section
            v-if="!commandModeActive && subjectHits.length"
            class="mb-2"
            aria-labelledby="palette-subjects-heading"
            data-testid="command-palette-subjects"
          >
            <h3
              id="palette-subjects-heading"
              class="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted"
            >
              Topics &amp; people
            </h3>
            <ul class="flex flex-col gap-0.5">
              <li v-for="hit in subjectHits" :key="hit.doc_id">
                <button
                  type="button"
                  class="flex w-full items-center justify-between gap-2 rounded px-2 py-1 text-left text-xs text-elevated-foreground hover:bg-overlay"
                  data-testid="command-palette-subject-hit"
                  @click="onSubjectHitClick(hit)"
                >
                  <span class="min-w-0 flex-1 truncate">{{ subjectLabelFor(hit) }}</span>
                  <span
                    class="inline-flex h-4 shrink-0 items-center justify-center rounded bg-primary/15 px-1 text-[9px] font-medium uppercase tracking-wide text-primary"
                  >{{ subjectKindLabelFor(hit) }}</span>
                </button>
              </li>
            </ul>
          </section>

          <ul
            v-if="contentHits.length && !commandModeActive"
            class="flex flex-col gap-1"
            data-testid="command-palette-results"
          >
            <li
              v-for="hit in contentHits"
              :key="hit.doc_id"
              class="rounded border border-border/60 p-2"
            >
              <p class="mb-1 line-clamp-2 text-xs text-elevated-foreground">
                {{ hit.text }}
              </p>
              <div class="flex flex-wrap gap-1">
                <button
                  type="button"
                  class="rounded border border-border px-1.5 py-0.5 text-[10px] text-primary hover:bg-overlay"
                  data-testid="command-palette-action-open-workspace"
                  @click="openInWorkspace(query)"
                >
                  Open in Workspace
                </button>
                <button
                  type="button"
                  class="rounded border border-border px-1.5 py-0.5 text-[10px] text-primary hover:bg-overlay"
                  data-testid="command-palette-action-pin-rail"
                  @click="pinToRail(hit)"
                >
                  Pin to rail
                </button>
                <button
                  type="button"
                  class="rounded border border-border px-1.5 py-0.5 text-[10px] text-primary hover:bg-overlay disabled:opacity-40"
                  :disabled="!graphNodeIdFromSearchHit(hit)"
                  data-testid="command-palette-action-show-graph"
                  @click="showOnGraph(hit)"
                >
                  Show on graph
                </button>
              </div>
            </li>
          </ul>
          <p
            v-else-if="query.trim() && !loading && !commandModeActive && !matchedCommands.length && !subjectHits.length && !episodeSuggestions.length && !contentHits.length"
            class="p-2 text-xs text-muted"
            data-testid="command-palette-no-results"
          >
            No results for “{{ query }}”.
          </p>
          <p
            v-else-if="commandModeActive && !matchedCommands.length"
            class="p-2 text-xs text-muted"
            data-testid="command-palette-commands-no-results"
          >
            No matching command.
          </p>
        </div>
      </div>
    </div>
  </Teleport>
</template>
