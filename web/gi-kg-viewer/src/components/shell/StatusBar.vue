<script setup lang="ts">
import { computed, ref, useTemplateRef } from 'vue'
import { getFeeds, putFeeds, type FeedApiEntry } from '../../api/feedsApi'
import { getOperatorConfig, putOperatorConfig } from '../../api/operatorConfigApi'
import { mergeOperatorYamlProfile, splitOperatorYamlProfile } from '../../utils/operatorYamlProfile'
import { useArtifactsStore } from '../../stores/artifacts'
import { useIndexStatsStore } from '../../stores/indexStats'
import { useShellStore } from '../../stores/shell'

const shell = useShellStore()
const artifacts = useArtifactsStore()
const indexStats = useIndexStatsStore()

const localFileInputRef = useTemplateRef<HTMLInputElement>('localFileInputRef')
const indexDialogRef = useTemplateRef<HTMLDialogElement>('indexDialogRef')
const artifactListDialogRef = useTemplateRef<HTMLDialogElement>('artifactListDialogRef')
const sourcesDialogRef = useTemplateRef<HTMLDialogElement>('sourcesDialogRef')

type SourcesDialogTab = 'feeds' | 'profile' | 'operator' | 'health'

const sourcesTab = ref<SourcesDialogTab>('feeds')
/** In-memory feed list (mirrors ``GET/PUT /api/feeds``); last write wins. */
const feedsCrudList = ref<FeedApiEntry[]>([])
const feedsSpecRelPath = ref('feeds.spec.yaml')
const feedsNewUrl = ref('')
const feedsEditingIndex = ref<number | null>(null)
const feedsEditingDraft = ref('')
const feedsEditorText = ref('')
type FeedsPanelTab = 'list' | 'json'
const feedsPanelTab = ref<FeedsPanelTab>('list')
const operatorYamlBody = ref('')
const operatorProfileSelected = ref('')
const availableProfiles = ref<string[]>([])
const operatorFileHint = ref('')
/** Corpus path (trimmed) we last GET for operator YAML; skip re-fetch when switching Job Profile ↔ Job Configuration. */
const operatorSourcesLoadedForPath = ref('')
const sourcesBusy = ref(false)
const sourcesError = ref<string | null>(null)

/** Advertised in GET /api/health but often off in minimal server builds. */
const optionalHealthCapsLimited = computed(
  () =>
    !shell.feedsApiAvailable ||
    !shell.operatorConfigApiAvailable ||
    !shell.jobsApiAvailable,
)

const healthDotClass = computed(() => {
  if (shell.healthError) {
    return 'bg-danger'
  }
  if (!shell.healthStatus) {
    return 'bg-muted'
  }
  const st = String(shell.healthStatus).toLowerCase()
  if (st !== 'ok') {
    return 'bg-warning'
  }
  /** Reachable + OK, but viewer cannot drive corpus APIs until path is set. */
  if (!shell.hasCorpusPath) {
    return 'bg-warning'
  }
  if (optionalHealthCapsLimited.value) {
    return 'bg-warning'
  }
  return 'bg-success'
})

/** Single summary line: server status + viewer context (path / optional routes). */
const healthDialogSummary = computed(() => {
  if (shell.healthError) {
    return { text: shell.healthError, textClass: 'text-danger' }
  }
  if (!shell.healthStatus) {
    return { text: 'Checking…', textClass: 'text-muted' }
  }
  const st = String(shell.healthStatus).toLowerCase()
  if (st !== 'ok') {
    return { text: shell.healthStatusDisplay, textClass: 'text-warning' }
  }
  if (!shell.hasCorpusPath) {
    return {
      text: 'OK — corpus path not set',
      textClass: 'text-warning',
    }
  }
  if (optionalHealthCapsLimited.value) {
    return {
      text: 'OK — limited (optional APIs off)',
      textClass: 'text-warning',
    }
  }
  return { text: shell.healthStatusDisplay, textClass: 'text-success' }
})

const showRebuildBolt = computed(
  () => Boolean(indexStats.indexEnvelope?.reindex_recommended),
)

/** Style for health-tab links to sample GET endpoints (new tab). */
const healthDocLinkClass =
  'text-primary underline decoration-dotted underline-offset-2 hover:decoration-solid'

function viewerApiAbsoluteUrl(pathWithQuery: string): string {
  if (typeof window === 'undefined') {
    return pathWithQuery
  }
  try {
    return new URL(pathWithQuery, window.location.origin).href
  } catch {
    return pathWithQuery
  }
}

function corpusPathQuery(): string {
  const p = shell.corpusPath.trim()
  return p ? `?${new URLSearchParams({ path: p }).toString()}` : ''
}

/**
 * Sample GET URL the viewer uses for each health row (same origin; opens in new tab).
 * When corpus path is set, includes ``path=`` where the API expects it.
 */
function healthApiProbeUrl(
  key:
    | 'health'
    | 'artifacts'
    | 'search'
    | 'explore'
    | 'index'
    | 'library'
    | 'digest'
    | 'feeds'
    | 'operator'
    | 'jobs',
): string {
  const cq = corpusPathQuery()
  const root = shell.corpusPath.trim()
  switch (key) {
    case 'health':
      return viewerApiAbsoluteUrl('/api/health')
    case 'artifacts':
      return viewerApiAbsoluteUrl(`/api/artifacts${cq}`)
    case 'search': {
      if (!root) {
        return viewerApiAbsoluteUrl('/api/search?q=_&top_k=5')
      }
      const sp = new URLSearchParams({ path: root, q: '_', top_k: '5' })
      return viewerApiAbsoluteUrl(`/api/search?${sp.toString()}`)
    }
    case 'explore': {
      if (!root) {
        return viewerApiAbsoluteUrl('/api/explore')
      }
      const ep = new URLSearchParams({ path: root, limit: '20' })
      return viewerApiAbsoluteUrl(`/api/explore?${ep.toString()}`)
    }
    case 'index':
      return viewerApiAbsoluteUrl(`/api/index/stats${cq}`)
    case 'library':
      return viewerApiAbsoluteUrl(`/api/corpus/feeds${cq}`)
    case 'digest':
      return viewerApiAbsoluteUrl(`/api/corpus/digest${cq}`)
    case 'feeds':
      return viewerApiAbsoluteUrl(`/api/feeds${cq}`)
    case 'operator':
      return viewerApiAbsoluteUrl(`/api/operator-config${cq}`)
    case 'jobs':
      return viewerApiAbsoluteUrl(`/api/jobs${cq}`)
    default:
      return viewerApiAbsoluteUrl('/api/health')
  }
}

function openSourcesDialogHealth(): void {
  sourcesTab.value = 'health'
  sourcesDialogRef.value?.showModal()
}

function openIndexDialog(): void {
  indexDialogRef.value?.showModal()
}

function triggerLocalFilePick(): void {
  localFileInputRef.value?.click()
}

const emit = defineEmits<{
  localArtifactsLoaded: [loaded: boolean]
  'go-graph': []
}>()

async function onListArtifactsClick(): Promise<void> {
  await shell.fetchArtifactList()
  artifactListDialogRef.value?.showModal()
}

async function onLoadIntoGraphFromDialog(): Promise<void> {
  await artifacts.loadSelected()
  if (artifacts.displayArtifact) {
    emit('go-graph')
    artifactListDialogRef.value?.close()
  }
}

async function onLocalFilesChange(ev: Event): Promise<void> {
  const el = ev.target as HTMLInputElement
  await artifacts.loadFromLocalFiles(el.files)
  el.value = ''
  emit('localArtifactsLoaded', Boolean(artifacts.displayArtifact))
}

/** Load only the active tab so a broken operator file does not block the Feeds editor. */
async function loadSourcesTab(tab: SourcesDialogTab): Promise<void> {
  if (tab === 'health') {
    return
  }
  const p = shell.corpusPath.trim()
  sourcesBusy.value = true
  sourcesError.value = null
  try {
    if (tab === 'feeds' && shell.feedsApiAvailable) {
      const f = await getFeeds(p)
      feedsCrudList.value = [...f.feeds]
      feedsSpecRelPath.value = f.file_relpath
      syncFeedsEditorFromCrud()
      feedsNewUrl.value = ''
      cancelFeedEdit()
      feedsPanelTab.value = 'list'
    } else if (
      (tab === 'profile' || tab === 'operator') &&
      shell.operatorConfigApiAvailable
    ) {
      if (operatorSourcesLoadedForPath.value === p) {
        return
      }
      const o = await getOperatorConfig(p)
      operatorFileHint.value = o.operator_config_path
      availableProfiles.value = o.available_profiles ?? []
      const sp = splitOperatorYamlProfile(o.content)
      operatorProfileSelected.value = sp.profile
      operatorYamlBody.value = sp.body
      operatorSourcesLoadedForPath.value = p
    }
  } catch (e) {
    sourcesError.value = e instanceof Error ? e.message : String(e)
  } finally {
    sourcesBusy.value = false
  }
}

async function openSourcesDialog(tab: SourcesDialogTab): Promise<void> {
  sourcesTab.value = tab
  if (tab !== 'health') {
    await loadSourcesTab(tab)
  }
  sourcesDialogRef.value?.showModal()
}

/** Open configuration dialog: Feeds tab if feeds API is on, else job configuration (YAML) tab. */
async function openSourcesDialogDefault(): Promise<void> {
  const feedsOn = shell.feedsApiAvailable
  const tab: SourcesDialogTab = feedsOn ? 'feeds' : 'operator'
  await openSourcesDialog(tab)
}

function closeSourcesDialog(): void {
  operatorSourcesLoadedForPath.value = ''
  sourcesDialogRef.value?.close()
}

async function selectSourcesTab(tab: SourcesDialogTab): Promise<void> {
  if (sourcesTab.value === tab) {
    return
  }
  sourcesTab.value = tab
  if (tab !== 'health') {
    await loadSourcesTab(tab)
  }
}

async function saveFeedsFromDialog(): Promise<void> {
  sourcesBusy.value = true
  sourcesError.value = null
  const p = shell.corpusPath.trim()
  try {
    let parsed: unknown
    try {
      parsed = JSON.parse(feedsEditorText.value) as unknown
    } catch {
      sourcesError.value = 'Invalid JSON — fix syntax before saving.'
      return
    }
    const root = parsed as { feeds?: unknown }
    if (!root || typeof root !== 'object' || !Array.isArray(root.feeds)) {
      sourcesError.value = 'Root must be an object with a "feeds" array.'
      return
    }
    const out = await putFeeds(p, root.feeds as FeedApiEntry[])
    feedsCrudList.value = [...out.feeds]
    feedsSpecRelPath.value = out.file_relpath
    feedsEditorText.value = JSON.stringify({ feeds: out.feeds }, null, 2)
    cancelFeedEdit()
  } catch (e) {
    sourcesError.value = e instanceof Error ? e.message : String(e)
  } finally {
    sourcesBusy.value = false
  }
}

async function saveOperatorFromDialog(): Promise<void> {
  sourcesBusy.value = true
  sourcesError.value = null
  const p = shell.corpusPath.trim()
  try {
    // Dropdown is the only source of truth for `profile:` (None = omit line; ignore pasted profile: in textarea).
    const inner = splitOperatorYamlProfile(operatorYamlBody.value)
    const prof = operatorProfileSelected.value.trim()
    const merged = mergeOperatorYamlProfile(prof, inner.body)
    const out = await putOperatorConfig(p, merged)
    availableProfiles.value = out.available_profiles ?? availableProfiles.value
    const sp = splitOperatorYamlProfile(out.content)
    operatorProfileSelected.value = sp.profile
    operatorYamlBody.value = sp.body
    operatorSourcesLoadedForPath.value = p
  } catch (e) {
    sourcesError.value = e instanceof Error ? e.message : String(e)
  } finally {
    sourcesBusy.value = false
  }
}

const corpusPathModel = computed({
  get: () => shell.corpusPath,
  set: (v: string) => {
    shell.corpusPath = v
  },
})

const feedsApiOpenHref = computed(() => {
  const p = shell.corpusPath.trim()
  if (!p) {
    return ''
  }
  return viewerApiAbsoluteUrl(`/api/feeds?${new URLSearchParams({ path: p }).toString()}`)
})

function feedEntryDisplayUrl(e: FeedApiEntry): string {
  if (typeof e === 'string') {
    return e.trim()
  }
  if (e && typeof e === 'object') {
    const o = e as Record<string, unknown>
    const u = o.url ?? o.rss
    if (typeof u === 'string') {
      return u.trim()
    }
  }
  return '(invalid entry)'
}

function feedEntryHasExtraKeys(e: FeedApiEntry): boolean {
  if (typeof e === 'string' || !e || typeof e !== 'object') {
    return false
  }
  const keys = Object.keys(e as Record<string, unknown>).filter((k) => k !== 'url' && k !== 'rss')
  return keys.length > 0
}

function syncFeedsEditorFromCrud(): void {
  feedsEditorText.value = JSON.stringify({ feeds: feedsCrudList.value }, null, 2)
}

async function persistFeedsFromCrud(): Promise<void> {
  const p = shell.corpusPath.trim()
  sourcesBusy.value = true
  sourcesError.value = null
  try {
    const out = await putFeeds(p, feedsCrudList.value)
    feedsCrudList.value = [...out.feeds]
    feedsSpecRelPath.value = out.file_relpath
    syncFeedsEditorFromCrud()
    feedsEditingIndex.value = null
    feedsEditingDraft.value = ''
  } catch (e) {
    sourcesError.value = e instanceof Error ? e.message : String(e)
    try {
      const f = await getFeeds(p)
      feedsCrudList.value = [...f.feeds]
      feedsSpecRelPath.value = f.file_relpath
      syncFeedsEditorFromCrud()
    } catch {
      /* ignore */
    }
  } finally {
    sourcesBusy.value = false
  }
}

async function reloadFeedsCrud(): Promise<void> {
  await loadSourcesTab('feeds')
}

function startFeedEdit(index: number): void {
  feedsEditingIndex.value = index
  feedsEditingDraft.value = feedEntryDisplayUrl(feedsCrudList.value[index]!)
}

function cancelFeedEdit(): void {
  feedsEditingIndex.value = null
  feedsEditingDraft.value = ''
}

async function saveFeedEdit(): Promise<void> {
  const idx = feedsEditingIndex.value
  if (idx == null || idx < 0 || idx >= feedsCrudList.value.length) {
    return
  }
  const u = feedsEditingDraft.value.trim()
  if (!u) {
    sourcesError.value = 'URL cannot be empty.'
    return
  }
  const next = feedsCrudList.value.map((e) =>
    typeof e === 'string' ? e : ({ ...(e as Record<string, unknown>) } as FeedApiEntry),
  )
  const otherUrls = next
    .map((e, i) => (i === idx ? null : feedEntryDisplayUrl(e)))
    .filter((x): x is string => x != null && x !== '(invalid entry)')
  if (otherUrls.includes(u)) {
    sourcesError.value = 'Another feed already uses this URL.'
    return
  }
  sourcesError.value = null
  const cur = next[idx]!
  if (typeof cur === 'string') {
    next[idx] = u
  } else {
    next[idx] = { ...(cur as Record<string, unknown>), url: u } as FeedApiEntry
  }
  feedsCrudList.value = next
  await persistFeedsFromCrud()
}

async function deleteFeedAt(index: number): Promise<void> {
  if (index < 0 || index >= feedsCrudList.value.length) {
    return
  }
  const next = feedsCrudList.value.filter((_, i) => i !== index)
  feedsCrudList.value = next
  if (feedsEditingIndex.value === index) {
    cancelFeedEdit()
  } else if (feedsEditingIndex.value != null && feedsEditingIndex.value > index) {
    feedsEditingIndex.value = feedsEditingIndex.value - 1
  }
  await persistFeedsFromCrud()
}

async function addFeedFromInput(): Promise<void> {
  const u = feedsNewUrl.value.trim()
  if (!u) {
    sourcesError.value = 'Enter a feed URL.'
    return
  }
  sourcesError.value = null
  const seen = new Set(feedsCrudList.value.map((e) => feedEntryDisplayUrl(e)))
  if (seen.has(u)) {
    sourcesError.value = 'That URL is already in the list.'
    return
  }
  feedsCrudList.value = [...feedsCrudList.value, u]
  feedsNewUrl.value = ''
  await persistFeedsFromCrud()
}
</script>

<template>
  <footer
    class="flex h-9 min-w-0 shrink-0 items-center gap-2 border-t border-border bg-canvas px-2 text-xs text-canvas-foreground"
    data-testid="app-status-bar"
  >
    <label class="sr-only" for="status-bar-corpus-path-input">Corpus path</label>
    <input
      id="status-bar-corpus-path-input"
      v-model="corpusPathModel"
      type="text"
      data-testid="status-bar-corpus-path"
      class="h-7 min-w-[12rem] w-[min(56.25rem,calc(100vw-14rem))] shrink-0 rounded border border-border bg-elevated px-2 py-0.5 font-mono text-[11px] text-elevated-foreground placeholder:text-muted"
      placeholder="Set corpus path…"
      autocomplete="off"
    >
    <input
      ref="localFileInputRef"
      type="file"
      class="sr-only"
      multiple
      accept=".gi.json,.kg.json,application/json"
      data-testid="status-bar-local-file-input"
      @change="onLocalFilesChange"
    >
    <button
      type="button"
      class="shrink-0 rounded border border-border px-1.5 py-0.5 text-[10px] hover:bg-overlay"
      title="Choose GI/KG JSON files (offline)"
      aria-label="Choose corpus files"
      @click="triggerLocalFilePick"
    >
      Files
    </button>
    <button
      v-if="shell.healthStatus && shell.corpusPath.trim()"
      type="button"
      class="shrink-0 rounded border border-border px-1.5 py-0.5 text-[10px] hover:bg-overlay"
      title="List GI/KG artifacts from the API for this corpus path"
      data-testid="status-bar-list-artifacts"
      @click="void onListArtifactsClick()"
    >
      List
    </button>
    <div class="ml-auto flex min-w-0 shrink-0 items-center gap-2">
      <button
        v-if="showRebuildBolt"
        type="button"
        class="shrink-0 rounded border border-warning/50 px-1.5 py-0.5 text-[10px] text-warning hover:bg-warning/10"
        data-testid="status-bar-rebuild-indicator"
        title="Index refresh recommended"
        @click="openIndexDialog"
      >
        Index
      </button>
      <button
        v-if="
          shell.healthStatus &&
            shell.hasCorpusPath &&
            (shell.feedsApiAvailable || shell.operatorConfigApiAvailable)
        "
        type="button"
        class="shrink-0 rounded border border-border px-1.5 py-0.5 text-[10px] hover:bg-overlay"
        title="Feeds, job profile, and job configuration (YAML)"
        data-testid="status-bar-sources-trigger"
        @click="void openSourcesDialogDefault()"
      >
        Configuration
      </button>
      <button
        type="button"
        class="flex shrink-0 items-center gap-1 rounded border border-border px-1.5 py-0.5 hover:bg-overlay"
        data-testid="status-bar-health-trigger"
        title="Health: server status, corpus path, and advertised API routes"
        @click="openSourcesDialogHealth"
      >
        <span
          class="inline-block h-2 w-2 shrink-0 rounded-full"
          :class="healthDotClass"
          aria-hidden="true"
        />
        <span class="hidden text-[10px] sm:inline">Health</span>
      </button>
    </div>
  </footer>

  <dialog
    ref="sourcesDialogRef"
    class="box-border h-[min(28rem,82vh)] w-[min(32rem,96vw)] max-h-[82vh] overflow-hidden rounded-lg border border-border bg-surface p-0 text-xs text-surface-foreground shadow-lg backdrop:bg-black/40"
    aria-labelledby="status-bar-settings-dialog-title"
    data-testid="status-bar-sources-dialog"
  >
    <!-- Inner flex wrapper: avoid ``display:flex`` on ``<dialog>`` — it overrides UA ``display:none`` when closed. -->
    <div class="flex h-full min-h-0 flex-col gap-2 p-3">
    <div class="shrink-0 space-y-2 border-b border-border pb-2">
      <div class="flex items-start justify-between gap-2">
        <div class="min-w-0">
          <h2 id="status-bar-settings-dialog-title" class="text-sm font-semibold">
            Corpus & API
          </h2>
        </div>
        <button
          type="button"
          class="shrink-0 rounded border border-border px-2 py-0.5 text-[10px] hover:bg-overlay"
          @click="closeSourcesDialog"
        >
          Close
        </button>
      </div>
      <div class="flex w-full min-w-0 gap-1">
        <button
          v-if="shell.feedsApiAvailable"
          type="button"
          class="min-w-0 flex-1 basis-0 truncate rounded px-2 py-0.5 text-center text-[10px] hover:bg-overlay"
          :class="sourcesTab === 'feeds' ? 'bg-overlay font-medium' : 'text-muted'"
          data-testid="sources-dialog-tab-feeds"
          @click="void selectSourcesTab('feeds')"
        >
          Feeds
        </button>
        <button
          v-if="shell.operatorConfigApiAvailable"
          type="button"
          class="min-w-0 flex-1 basis-0 truncate rounded px-2 py-0.5 text-center text-[10px] hover:bg-overlay"
          :class="sourcesTab === 'profile' ? 'bg-overlay font-medium' : 'text-muted'"
          data-testid="sources-dialog-tab-profile"
          @click="void selectSourcesTab('profile')"
        >
          Job Profile
        </button>
        <button
          v-if="shell.operatorConfigApiAvailable"
          type="button"
          class="min-w-0 flex-1 basis-0 truncate rounded px-2 py-0.5 text-center text-[10px] hover:bg-overlay"
          :class="sourcesTab === 'operator' ? 'bg-overlay font-medium' : 'text-muted'"
          data-testid="sources-dialog-tab-operator"
          @click="void selectSourcesTab('operator')"
        >
          Job Configuration
        </button>
        <button
          type="button"
          class="min-w-0 flex-1 basis-0 truncate rounded px-2 py-0.5 text-center text-[10px] hover:bg-overlay"
          :class="sourcesTab === 'health' ? 'bg-overlay font-medium' : 'text-muted'"
          data-testid="sources-dialog-tab-health"
          @click="void selectSourcesTab('health')"
        >
          Health
        </button>
      </div>
    </div>
    <p v-if="sourcesBusy" class="shrink-0 text-[10px] text-muted">
      Loading…
    </p>
    <p v-if="sourcesError" class="shrink-0 rounded border border-danger/40 bg-danger/10 px-2 py-1 text-[10px] text-danger">
      {{ sourcesError }}
    </p>
    <!-- Per-tab: scrollable middle + pinned footer; flex-1 textareas fill space above actions. -->
    <div class="flex min-h-0 flex-1 flex-col overflow-hidden pr-0.5">
    <div
      v-show="sourcesTab === 'feeds' && shell.feedsApiAvailable"
      class="flex min-h-0 flex-1 flex-col gap-2"
    >
      <div class="flex min-h-0 flex-1 flex-col gap-2 overflow-hidden">
        <p
          v-if="shell.operatorConfigApiAvailable"
          class="shrink-0 text-[10px] text-muted leading-snug"
        >
          Pipeline preset + YAML (<code class="rounded bg-overlay px-0.5 font-mono text-[9px]">GET /api/operator-config</code>):
          <button
            type="button"
            class="font-medium text-surface-foreground underline decoration-dotted underline-offset-2 hover:decoration-solid"
            data-testid="sources-dialog-jump-to-profile"
            @click="void selectSourcesTab('profile')"
          >
            Job Profile
          </button>
          (packaged <code class="font-mono text-[9px]">profile:</code>)
          or
          <button
            type="button"
            class="font-medium text-surface-foreground underline decoration-dotted underline-offset-2 hover:decoration-solid"
            data-testid="sources-dialog-jump-to-operator"
            @click="void selectSourcesTab('operator')"
          >
            Job Configuration
          </button>
          (YAML without top-level <code class="font-mono text-[9px]">profile:</code>).
        </p>
        <div class="flex w-full min-w-0 shrink-0 gap-1 border-b border-border pb-1.5">
          <button
            type="button"
            class="min-w-0 flex-1 basis-0 truncate rounded px-2 py-0.5 text-center text-[10px] hover:bg-overlay"
            :class="feedsPanelTab === 'list' ? 'bg-overlay font-medium' : 'text-muted'"
            data-testid="sources-dialog-feeds-panel-list"
            @click="feedsPanelTab = 'list'"
          >
            Manage
          </button>
          <button
            type="button"
            class="min-w-0 flex-1 basis-0 truncate rounded px-2 py-0.5 text-center text-[10px] hover:bg-overlay"
            :class="feedsPanelTab === 'json' ? 'bg-overlay font-medium' : 'text-muted'"
            data-testid="sources-dialog-feeds-panel-json"
            @click="feedsPanelTab = 'json'"
          >
            Raw JSON
          </button>
        </div>
        <div class="flex min-h-0 flex-1 flex-col overflow-hidden">
          <div
            v-if="feedsPanelTab === 'list'"
            class="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto overscroll-contain"
          >
            <p class="shrink-0 text-[10px] text-muted leading-snug">
              Structured <code class="rounded bg-overlay px-0.5 font-mono text-[9px]">{{ feedsSpecRelPath }}</code> under the corpus root. Each row is one feed (URL string or object with <code class="font-mono text-[9px]">url</code>); edits save immediately. Do not duplicate feeds in operator config.
            </p>
            <p
              v-if="feedsApiOpenHref"
              class="shrink-0 text-[10px] leading-snug"
            >
              <a
                :href="feedsApiOpenHref"
                :class="healthDocLinkClass"
                target="_blank"
                rel="noopener noreferrer"
              >Open {{ feedsSpecRelPath }} (GET JSON)</a>
            </p>
            <div class="flex shrink-0 flex-wrap items-end gap-2">
              <label class="sr-only" for="sources-dialog-feeds-add-url">Add feed URL</label>
              <input
                id="sources-dialog-feeds-add-url"
                v-model="feedsNewUrl"
                type="url"
                data-testid="sources-dialog-feeds-add-url"
                class="min-w-[12rem] flex-1 rounded border border-border bg-elevated px-2 py-1 font-mono text-[11px] text-elevated-foreground placeholder:text-muted"
                placeholder="https://example.com/feed.xml"
                autocomplete="off"
                @keydown.enter.prevent="void addFeedFromInput()"
              >
              <button
                type="button"
                class="shrink-0 rounded border border-border px-2 py-1 text-[10px] hover:bg-overlay disabled:opacity-40"
                :disabled="sourcesBusy"
                data-testid="sources-dialog-feeds-add-btn"
                @click="void addFeedFromInput()"
              >
                Add feed
              </button>
            </div>
            <div
              v-if="feedsCrudList.length === 0"
              class="shrink-0 rounded border border-border/60 bg-overlay/40 px-2 py-2 text-[10px] text-muted"
            >
              No feeds yet — add a URL above or use the Raw JSON sub-tab.
            </div>
            <ul
              v-else
              class="shrink-0 list-none space-y-1.5 p-0"
              data-testid="sources-dialog-feeds-list"
            >
              <li
                v-for="(entry, idx) in feedsCrudList"
                :key="`${idx}-${feedEntryDisplayUrl(entry)}`"
                class="flex items-start gap-2 rounded border border-border bg-elevated px-2 py-1.5"
                :data-testid="`sources-dialog-feeds-row-${idx}`"
              >
                <div class="min-w-0 flex-1">
                  <template v-if="feedsEditingIndex === idx">
                    <input
                      v-model="feedsEditingDraft"
                      type="url"
                      class="w-full rounded border border-border bg-surface px-2 py-0.5 font-mono text-[11px] text-surface-foreground"
                      :data-testid="`sources-dialog-feeds-row-edit-input-${idx}`"
                      @keydown.enter.prevent="void saveFeedEdit()"
                    >
                  </template>
                  <template v-else>
                    <div class="truncate font-mono text-[11px] text-elevated-foreground">
                      {{ feedEntryDisplayUrl(entry) }}
                    </div>
                    <div
                      v-if="feedEntryHasExtraKeys(entry)"
                      class="mt-0.5 text-[9px] text-muted"
                    >
                      Has extra fields (edit URL only; save keeps other options)
                    </div>
                  </template>
                </div>
                <div class="flex shrink-0 flex-col gap-0.5 sm:flex-row sm:items-center">
                  <template v-if="feedsEditingIndex === idx">
                    <button
                      type="button"
                      class="rounded border border-border px-1.5 py-0.5 text-[10px] hover:bg-overlay disabled:opacity-40"
                      :disabled="sourcesBusy"
                      :data-testid="`sources-dialog-feeds-row-save-${idx}`"
                      @click="void saveFeedEdit()"
                    >
                      Save
                    </button>
                    <button
                      type="button"
                      class="rounded border border-border px-1.5 py-0.5 text-[10px] hover:bg-overlay"
                      :data-testid="`sources-dialog-feeds-row-cancel-${idx}`"
                      @click="cancelFeedEdit()"
                    >
                      Cancel
                    </button>
                  </template>
                  <template v-else>
                    <button
                      type="button"
                      class="rounded border border-border px-1.5 py-0.5 text-[10px] hover:bg-overlay disabled:opacity-40"
                      :disabled="sourcesBusy"
                      :data-testid="`sources-dialog-feeds-row-edit-${idx}`"
                      @click="startFeedEdit(idx)"
                    >
                      Edit
                    </button>
                    <button
                      type="button"
                      class="rounded border border-border px-1.5 py-0.5 text-[10px] text-danger hover:bg-danger/10 disabled:opacity-40"
                      :disabled="sourcesBusy"
                      :data-testid="`sources-dialog-feeds-row-delete-${idx}`"
                      @click="void deleteFeedAt(idx)"
                    >
                      Delete
                    </button>
                  </template>
                </div>
              </li>
            </ul>
          </div>
          <div
            v-else
            class="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto overscroll-contain"
          >
            <p class="shrink-0 text-[10px] text-muted leading-snug">
              Root object must include a <code class="font-mono text-[9px]">feeds</code> array (URL strings or objects with <code class="font-mono text-[9px]">url</code>). <strong class="text-surface-foreground">Apply JSON</strong> saves to the server.
            </p>
            <textarea
              v-model="feedsEditorText"
              data-testid="sources-dialog-feeds-textarea"
              class="min-h-[12rem] w-full flex-1 resize-y rounded border border-border bg-elevated p-2 font-mono text-[11px] text-elevated-foreground"
              spellcheck="false"
              aria-label="Feeds spec JSON (feeds array)"
            />
            <div class="shrink-0">
              <button
                type="button"
                class="rounded border border-border px-2 py-1 text-[10px] hover:bg-overlay disabled:opacity-40"
                :disabled="sourcesBusy"
                data-testid="sources-dialog-feeds-apply-json"
                @click="void saveFeedsFromDialog()"
              >
                Apply JSON
              </button>
            </div>
          </div>
        </div>
      </div>
      <div class="shrink-0 border-t border-border pt-2">
        <button
          type="button"
          class="rounded border border-border px-2 py-1 text-[10px] hover:bg-overlay disabled:opacity-40"
          :disabled="sourcesBusy"
          data-testid="sources-dialog-feeds-reload"
          @click="void reloadFeedsCrud()"
        >
          Reload from server
        </button>
      </div>
    </div>
    <div
      v-show="sourcesTab === 'profile' && shell.operatorConfigApiAvailable"
      class="flex min-h-0 flex-1 flex-col gap-2"
    >
      <div class="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto overscroll-contain">
        <p v-if="operatorFileHint" class="shrink-0 break-all text-[9px] text-muted leading-snug">
          {{ operatorFileHint }}
        </p>
        <p class="shrink-0 text-[10px] text-muted leading-snug">
          Packaged preset <code class="rounded bg-overlay px-0.5 font-mono text-[9px]">profile:</code> merges first; keys in <strong class="text-surface-foreground">Job Configuration</strong> win — same idea as CLI <code class="font-mono text-[9px]">--profile</code> plus <code class="font-mono text-[9px]">--config</code>. This menu is the source of truth for top-level <code class="font-mono text-[9px]">profile:</code>: <strong>None</strong> removes it even if a stale line exists on disk. If the menu is empty, no packaged presets were found (check <code class="font-mono text-[9px]">config/profiles</code>). Do not put API keys in YAML — use environment variables. RSS / feeds belong in <strong class="text-surface-foreground">Feeds</strong>; the server rejects feed keys and secrets on save.
        </p>
        <div class="flex shrink-0 flex-wrap items-center gap-2">
          <label
            for="sources-dialog-profile-select"
            class="text-[10px] text-muted shrink-0"
          >Preset</label>
          <select
            id="sources-dialog-profile-select"
            v-model="operatorProfileSelected"
            data-testid="sources-dialog-profile-select"
            class="max-w-[min(100%,16rem)] rounded border border-border bg-elevated px-2 py-1 text-[11px] text-elevated-foreground"
            aria-label="Pipeline profile preset"
          >
            <option value="">
              None
            </option>
            <option
              v-if="operatorProfileSelected && !availableProfiles.includes(operatorProfileSelected)"
              :value="operatorProfileSelected"
            >
              {{ operatorProfileSelected }} (custom)
            </option>
            <option
              v-for="n in availableProfiles"
              :key="n"
              :value="n"
            >
              {{ n }}
            </option>
          </select>
        </div>
      </div>
      <div class="shrink-0 border-t border-border pt-2">
        <button
          type="button"
          class="rounded border border-border px-2 py-1 text-[10px] hover:bg-overlay disabled:opacity-40"
          :disabled="sourcesBusy"
          data-testid="sources-dialog-save-profile"
          @click="void saveOperatorFromDialog()"
        >
          Save (applies preset + overrides on disk)
        </button>
      </div>
    </div>
    <div
      v-show="sourcesTab === 'operator' && shell.operatorConfigApiAvailable"
      class="flex min-h-0 flex-1 flex-col gap-2"
    >
      <div class="shrink-0 space-y-2">
        <p v-if="operatorFileHint" class="break-all text-[9px] text-muted leading-snug">
          {{ operatorFileHint }}
        </p>
        <p class="text-[10px] text-muted leading-snug">
          YAML overrides only (no top-level <code class="font-mono text-[9px]">profile:</code> — use the <strong class="text-surface-foreground">Job Profile</strong> tab for the preset line). Same file as the API response; saving merges with the current preset from the Job Profile tab.
        </p>
      </div>
      <textarea
        v-model="operatorYamlBody"
        data-testid="sources-dialog-operator-textarea"
        class="min-h-0 w-full flex-1 resize-none rounded border border-border bg-elevated p-2 font-mono text-[11px] text-elevated-foreground"
        spellcheck="false"
        aria-label="Job configuration (YAML)"
      />
      <div class="shrink-0 border-t border-border pt-2">
        <button
          type="button"
          class="rounded border border-border px-2 py-1 text-[10px] hover:bg-overlay disabled:opacity-40"
          :disabled="sourcesBusy"
          data-testid="sources-dialog-save-overrides"
          @click="void saveOperatorFromDialog()"
        >
          Save YAML
        </button>
      </div>
    </div>
    <div
      v-show="sourcesTab === 'health'"
      class="flex min-h-0 flex-1 flex-col gap-2"
      data-testid="sources-dialog-health-panel"
    >
      <div class="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto overscroll-contain">
        <p class="shrink-0 text-[10px] text-muted leading-snug">
          Flags from
          <a
            :href="healthApiProbeUrl('health')"
            :class="[healthDocLinkClass, 'inline font-mono text-[9px]']"
            target="_blank"
            rel="noopener noreferrer"
          ><code class="rounded bg-overlay px-0.5">GET /api/health</code></a>
        </p>
        <div class="min-h-[12rem] flex-1 rounded border border-border bg-elevated p-3 text-[10px]">
        <dl class="space-y-1">
          <div class="flex justify-between gap-2">
            <dt class="min-w-0 text-muted">
              <a
                :href="healthApiProbeUrl('health')"
                :class="healthDocLinkClass"
                target="_blank"
                rel="noopener noreferrer"
              >Health</a>
            </dt>
            <dd class="text-right font-medium leading-snug">
              <span :class="healthDialogSummary.textClass">{{ healthDialogSummary.text }}</span>
            </dd>
          </div>
        </dl>
        <dl
          v-if="shell.healthStatus"
          class="mt-1.5 space-y-1 border-t border-border/60 pt-1.5"
        >
          <div class="flex justify-between gap-2">
            <dt class="min-w-0 text-muted">
              <a
                :href="healthApiProbeUrl('artifacts')"
                :class="healthDocLinkClass"
                target="_blank"
                rel="noopener noreferrer"
              >Artifacts (graph)</a>
            </dt>
            <dd :class="shell.artifactsApiAvailable !== false ? 'text-success' : 'text-danger'">
              {{ shell.artifactsApiAvailable !== false ? 'Yes' : 'No' }}
            </dd>
          </div>
          <div class="flex justify-between gap-2">
            <dt class="min-w-0 text-muted">
              <a
                :href="healthApiProbeUrl('search')"
                :class="healthDocLinkClass"
                target="_blank"
                rel="noopener noreferrer"
              >Semantic search</a>
            </dt>
            <dd :class="shell.searchApiAvailable !== false ? 'text-success' : 'text-danger'">
              {{ shell.searchApiAvailable !== false ? 'Yes' : 'No' }}
            </dd>
          </div>
          <div class="flex justify-between gap-2">
            <dt class="min-w-0 text-muted">
              <a
                :href="healthApiProbeUrl('explore')"
                :class="healthDocLinkClass"
                target="_blank"
                rel="noopener noreferrer"
              >Graph explore</a>
            </dt>
            <dd :class="shell.exploreApiAvailable !== false ? 'text-success' : 'text-danger'">
              {{ shell.exploreApiAvailable !== false ? 'Yes' : 'No' }}
            </dd>
          </div>
          <div class="flex justify-between gap-2">
            <dt class="min-w-0 text-muted">
              <a
                :href="healthApiProbeUrl('index')"
                :class="healthDocLinkClass"
                target="_blank"
                rel="noopener noreferrer"
              >Index routes</a>
            </dt>
            <dd :class="shell.indexRoutesApiAvailable !== false ? 'text-success' : 'text-danger'">
              {{ shell.indexRoutesApiAvailable !== false ? 'Yes' : 'No' }}
            </dd>
          </div>
          <div class="flex justify-between gap-2">
            <dt class="min-w-0 text-muted">
              <a
                :href="healthApiProbeUrl('library')"
                :class="healthDocLinkClass"
                target="_blank"
                rel="noopener noreferrer"
              >Library API</a>
            </dt>
            <dd :class="shell.corpusLibraryApiAvailable ? 'text-success' : 'text-danger'">
              {{ shell.corpusLibraryApiAvailable ? 'Yes' : 'No' }}
            </dd>
          </div>
          <div class="flex justify-between gap-2">
            <dt class="min-w-0 text-muted">
              <a
                :href="healthApiProbeUrl('digest')"
                :class="healthDocLinkClass"
                target="_blank"
                rel="noopener noreferrer"
              >Digest API</a>
            </dt>
            <dd :class="shell.corpusDigestApiAvailable ? 'text-success' : 'text-danger'">
              {{ shell.corpusDigestApiAvailable ? 'Yes' : 'No' }}
            </dd>
          </div>
          <div class="flex justify-between gap-2">
            <dt class="min-w-0 text-muted">
              <a
                :href="healthApiProbeUrl('feeds')"
                :class="healthDocLinkClass"
                target="_blank"
                rel="noopener noreferrer"
              >Feeds file API</a>
            </dt>
            <dd :class="shell.feedsApiAvailable ? 'text-success' : 'text-danger'">
              {{ shell.feedsApiAvailable ? 'Yes' : 'No' }}
            </dd>
          </div>
          <div class="flex justify-between gap-2">
            <dt class="min-w-0 text-muted">
              <a
                :href="healthApiProbeUrl('operator')"
                :class="healthDocLinkClass"
                target="_blank"
                rel="noopener noreferrer"
              >Operator YAML API</a>
            </dt>
            <dd :class="shell.operatorConfigApiAvailable ? 'text-success' : 'text-danger'">
              {{ shell.operatorConfigApiAvailable ? 'Yes' : 'No' }}
            </dd>
          </div>
          <div class="flex justify-between gap-2">
            <dt class="min-w-0 text-muted">
              <a
                :href="healthApiProbeUrl('jobs')"
                :class="healthDocLinkClass"
                target="_blank"
                rel="noopener noreferrer"
              >Pipeline jobs API</a>
            </dt>
            <dd :class="shell.jobsApiAvailable ? 'text-success' : 'text-danger'">
              {{ shell.jobsApiAvailable ? 'Yes' : 'No' }}
            </dd>
          </div>
        </dl>
      </div>
      </div>
      <div class="shrink-0 space-y-2 border-t border-border pt-2">
        <button
          type="button"
          class="rounded border border-border px-2 py-1 text-[10px] hover:bg-overlay"
          @click="shell.fetchHealth()"
        >
          Retry health
        </button>
        <div
          v-if="shell.healthError"
          class="rounded border border-border bg-overlay p-2 text-[10px]"
        >
          <p class="mb-1 text-muted">
            Load files directly (no API):
          </p>
          <button
            type="button"
            class="rounded border border-border px-2 py-0.5 hover:bg-canvas"
            @click="triggerLocalFilePick"
          >
            Choose files…
          </button>
        </div>
      </div>
    </div>
    </div>
    </div>
  </dialog>

  <dialog
    ref="indexDialogRef"
    class="max-w-md rounded-lg border border-border bg-surface p-4 text-xs text-surface-foreground shadow-lg backdrop:bg-black/40"
    aria-labelledby="status-bar-index-dialog-title"
  >
    <div class="mb-2 flex items-center justify-between gap-2">
      <h2 id="status-bar-index-dialog-title" class="text-sm font-semibold">
        Vector index
      </h2>
      <button
        type="button"
        class="rounded border border-border px-2 py-0.5 text-[10px] hover:bg-overlay"
        @click="indexDialogRef?.close()"
      >
        Close
      </button>
    </div>
    <p
      v-if="indexStats.indexEnvelope?.rebuild_in_progress"
      class="mb-1 text-[10px] text-muted leading-snug"
    >
      Background index job running — stats refresh automatically.
    </p>
    <p
      v-if="indexStats.indexEnvelope?.rebuild_last_error"
      class="mb-1 text-[10px] text-danger leading-snug"
    >
      Last rebuild error: {{ indexStats.indexEnvelope.rebuild_last_error }}
    </p>
    <div class="mt-2 flex flex-wrap gap-1">
      <button
        type="button"
        class="rounded border border-border px-2 py-0.5 text-[10px] hover:bg-overlay disabled:opacity-40"
        :disabled="!shell.healthStatus || indexStats.indexLoading"
        @click="indexStats.refreshIndexStats()"
      >
        {{ indexStats.indexLoading ? 'Loading…' : 'Refresh' }}
      </button>
      <button
        type="button"
        class="rounded border border-border px-2 py-0.5 text-[10px] hover:bg-overlay disabled:opacity-40"
        :disabled="indexStats.rebuildActionsDisabled"
        @click="indexStats.requestIndexRebuild(false)"
      >
        {{ indexStats.rebuildSubmitting ? 'Queueing…' : 'Update index' }}
      </button>
      <button
        type="button"
        class="rounded border border-border px-2 py-0.5 text-[10px] hover:bg-overlay disabled:opacity-40"
        :disabled="indexStats.rebuildActionsDisabled"
        @click="indexStats.requestIndexRebuild(true)"
      >
        Full rebuild
      </button>
    </div>
  </dialog>

  <dialog
    ref="artifactListDialogRef"
    class="max-h-[min(80vh,32rem)] max-w-lg overflow-y-auto rounded-lg border border-border bg-surface p-4 text-xs text-surface-foreground shadow-lg backdrop:bg-black/40"
    aria-labelledby="status-bar-artifact-list-title"
    data-testid="artifact-list-dialog"
  >
    <div class="mb-2 flex items-center justify-between gap-2">
      <h2 id="status-bar-artifact-list-title" class="text-sm font-semibold">
        Corpus artifacts
      </h2>
      <button
        type="button"
        class="rounded border border-border px-2 py-0.5 text-[10px] hover:bg-overlay"
        @click="artifactListDialogRef?.close()"
      >
        Close
      </button>
    </div>
    <p v-if="shell.artifactsLoading" class="mt-1 text-[10px] text-muted">
      Loading…
    </p>
    <template v-else>
      <div
        v-if="shell.corpusHints.length"
        class="mt-1.5 rounded border border-warning/40 bg-warning/10 px-2 py-1.5 text-[10px] text-surface-foreground"
        role="status"
      >
        <p class="font-medium text-warning">
          Corpus path hint
        </p>
        <ul class="mt-0.5 list-inside list-disc text-muted">
          <li v-for="(h, i) in shell.corpusHints" :key="i">
            {{ h }}
          </li>
        </ul>
      </div>
      <div
        v-if="shell.artifactList.length"
        class="mt-1.5 space-y-1"
      >
        <div class="flex flex-wrap items-center gap-1">
          <button
            type="button"
            class="rounded border border-border px-1.5 py-0.5 text-[10px] hover:bg-overlay"
            @click="artifacts.selectAllListed(shell.artifactList.map((a) => a.relative_path))"
          >
            All
          </button>
          <button
            type="button"
            class="rounded border border-border px-1.5 py-0.5 text-[10px] hover:bg-overlay"
            @click="artifacts.deselectAllListed()"
          >
            None
          </button>
          <button
            type="button"
            class="rounded border border-border px-2 py-0.5 text-[10px] font-medium hover:bg-overlay disabled:opacity-40"
            :disabled="artifacts.selectedRelPaths.length === 0 || artifacts.loading"
            @click="void onLoadIntoGraphFromDialog()"
          >
            {{ artifacts.loading ? 'Loading…' : 'Load into graph' }}
          </button>
        </div>
        <div class="max-h-48 overflow-y-auto rounded border border-border bg-elevated p-1 text-[11px]">
          <label
            v-for="a in shell.artifactList"
            :key="a.relative_path"
            class="flex cursor-pointer items-start gap-1 py-0.5 hover:bg-overlay"
          >
            <input
              type="checkbox"
              class="mt-0.5 rounded border-border"
              :checked="artifacts.selectedRelPaths.includes(a.relative_path)"
              @change="artifacts.toggleSelection(a.relative_path)"
            >
            <span class="break-all">
              <span :class="a.kind === 'gi' ? 'text-gi' : 'text-kg'">{{ a.kind }}</span>
              {{ a.relative_path }}
            </span>
          </label>
        </div>
      </div>
      <p
        v-else-if="shell.artifactCount !== null && shell.artifactCount === 0"
        class="mt-1 text-[10px] text-muted"
      >
        No artifacts found.
      </p>
    </template>
    <p v-if="shell.artifactsError" class="mt-1 text-[10px] text-danger">
      {{ shell.artifactsError }}
    </p>
    <p v-if="artifacts.loadError" class="mt-1 text-[10px] text-danger">
      {{ artifacts.loadError }}
    </p>
  </dialog>
</template>
