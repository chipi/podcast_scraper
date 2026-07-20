<script setup lang="ts">
import { computed, ref, useTemplateRef, watch } from 'vue'
import { getFeeds, putFeeds, type FeedApiEntry } from '../../api/feedsApi'
import { fetchCorpusFeeds } from '../../api/corpusLibraryApi'
import { fetchCorpusCoverage, type CoverageByMonthItem } from '../../api/corpusCoverageApi'
import {
  fetchIndexTimeseries,
  type IndexTimeseriesResponse,
} from '../../api/indexStatsApi'
import {
  getOperatorConfig,
  getOperatorProfiles,
  putOperatorConfig,
  type PackagedProfile,
} from '../../api/operatorConfigApi'
import { mergeOperatorYamlProfile, splitOperatorYamlProfile } from '../../utils/operatorYamlProfile'
import { toggleScheduledJobEnabled } from '../../utils/scheduledJobsYaml'
import AppDialog from '../shared/AppDialog.vue'
import CronSchedulePreview from './CronSchedulePreview.vue'
import EnrichmentPanel from './EnrichmentPanel.vue'
import IndexTimeseriesChart from './IndexTimeseriesChart.vue'
import type { TimeseriesSeries } from './timeseriesChart'
import FeedOverrideEditor from './FeedOverrideEditor.vue'
import ScheduledJobsSection from './ScheduledJobsSection.vue'
import { useArtifactsStore } from '../../stores/artifacts'
import { useIndexStatsStore } from '../../stores/indexStats'
import { useShellStore } from '../../stores/shell'

const shell = useShellStore()
const artifacts = useArtifactsStore()
const indexStats = useIndexStatsStore()

const localFileInputRef = useTemplateRef<HTMLInputElement>('localFileInputRef')
const artifactListDialogOpen = ref(false)
const sourcesDialogOpen = ref(false)

type SourcesDialogTab = 'feeds' | 'enrichment' | 'operator' | 'scheduled' | 'index' | 'health'

const sourcesTab = ref<SourcesDialogTab>('feeds')
/** In-memory feed list (mirrors ``GET/PUT /api/feeds``); last write wins. */
const feedsCrudList = ref<FeedApiEntry[]>([])
const feedsSpecRelPath = ref('feeds.spec.yaml')
const feedsNewUrl = ref('')
const feedsEditingIndex = ref<number | null>(null)
const feedsEditingDraft = ref('')
/** When set, the Feeds Manage panel shows the per-feed override editor (#694). */
const feedsDetailIndex = ref<number | null>(null)
const feedsEditorText = ref('')
type FeedsPanelTab = 'list' | 'json'
const feedsPanelTab = ref<FeedsPanelTab>('list')
const operatorYamlBody = ref('')
const operatorProfileSelected = ref('')
const availableProfiles = ref<string[]>([])
/** Packaged profile bodies (name → content) for the Profile sub-tab "what it brings". */
const profileContents = ref<PackagedProfile[]>([])
/** Sub-tab inside the Job Configuration page: profile picker vs YAML overrides. */
const operatorPanelTab = ref<'profile' | 'config'>('profile')
const operatorFileHint = ref('')
/** Corpus path (trimmed) we last GET for operator YAML; skip re-fetch when switching Job Profile ↔ Job Configuration. */
const operatorSourcesLoadedForPath = ref('')
const sourcesBusy = ref(false)
const sourcesError = ref<string | null>(null)

/** feed_id (sha) → display title, for naming feeds in the Index section. */
const indexFeedNameMap = ref<Record<string, string>>({})
/** Corpus coverage by month — supplies the chart's "Episodes" series. */
const indexCoverageByMonth = ref<CoverageByMonthItem[]>([])
/** Indexed docs by publish month × doc_type — the chart's per-type series. */
const indexTimeseries = ref<IndexTimeseriesResponse | null>(null)

/** Title-cased doc_type for series legends ("with_gi" → "With gi", "insight" → "Insight"). */
function prettyDocType(dt: string): string {
  const s = dt.replace(/_/g, ' ').trim()
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : dt
}

/** Shared month axis = union of coverage + index-timeseries months, sorted. */
const indexChartLabels = computed<string[]>(() => {
  const set = new Set<string>()
  for (const m of indexCoverageByMonth.value) set.add(m.month)
  for (const m of indexTimeseries.value?.by_month ?? []) set.add(m.month)
  return [...set].sort()
})

/** Episodes (coverage) + one line per indexed doc_type, aligned to the axis. */
const indexChartSeries = computed<TimeseriesSeries[]>(() => {
  const labels = indexChartLabels.value
  if (!labels.length) return []
  const covByMonth = new Map(indexCoverageByMonth.value.map((m) => [m.month, m.total]))
  const series: TimeseriesSeries[] = [
    {
      key: 'episodes',
      label: 'Episodes',
      data: labels.map((m) => covByMonth.get(m) ?? 0),
      defaultEnabled: true,
    },
  ]
  const ts = indexTimeseries.value
  if (ts && Array.isArray(ts.by_month) && Array.isArray(ts.doc_types)) {
    const byMonth = new Map(ts.by_month.map((m) => [m.month, m.doc_types]))
    for (const dt of ts.doc_types) {
      series.push({
        key: `doc:${dt}`,
        label: prettyDocType(dt),
        data: labels.map((m) => byMonth.get(m)?.[dt] ?? 0),
        defaultEnabled: true,
      })
    }
  }
  return series
})

/** Index facts minus the raw-hash "Feeds indexed" row (shown by name below). */
const indexFactRows = computed(() =>
  indexStats.indexRows.filter((r) => r.k !== 'Feeds indexed'),
)

/** Indexed feeds shown by display title (falls back to a short id when unknown). */
const indexedFeedNames = computed(() => {
  const ids = indexStats.indexEnvelope?.stats?.feeds_indexed ?? []
  return ids.map((id) => {
    const name = indexFeedNameMap.value[id]
    if (name) return name
    return id.startsWith('sha256:') ? `${id.slice(0, 14)}…` : id
  })
})

/** Per-doc-type counts for the Index "Documents by type" bars (with bar %). */
const indexDocTypeBars = computed(() => {
  const counts = indexStats.indexEnvelope?.stats?.doc_type_counts
  if (!counts) return [] as { k: string; n: number; pct: number }[]
  const entries = Object.entries(counts)
    .filter(([, n]) => typeof n === 'number')
    .map(([k, n]) => ({ k, n: Number(n) }))
    .sort((a, b) => b.n - a.n)
  const max = entries.reduce((m, e) => Math.max(m, e.n), 0) || 1
  return entries.map((e) => ({ ...e, pct: Math.max(2, Math.round((e.n / max) * 100)) }))
})


/** Raw YAML of the currently-selected profile (empty when None or not loaded). */
const selectedProfileContent = computed(() => {
  const name = operatorProfileSelected.value.trim()
  if (!name) return ''
  return profileContents.value.find((p) => p.name === name)?.content ?? ''
})

/** Top-level `key: value` settings from the selected profile — the at-a-glance
 *  "what this profile brings" (skips comments / nested / the profile: line). */
const selectedProfileSettings = computed(() => {
  const out: { key: string; value: string }[] = []
  for (const raw of selectedProfileContent.value.split('\n')) {
    const line = raw.trimEnd()
    if (!line || line.startsWith('#') || line.startsWith(' ') || line.startsWith('-')) continue
    const m = /^([A-Za-z0-9_]+) *: *(.*)$/.exec(line)
    if (!m || m[1] === 'profile') continue
    const value = (m[2] ?? '').replace(/\s+#.*$/, '').trim()
    if (value) out.push({ key: m[1]!, value })
  }
  return out
})

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
  sourcesDialogOpen.value = true
}

// Dashboard surfaces (Index status card, Briefing recommendations) and the
// status-bar bolt all open the Configuration dialog at its **Index** section —
// the single home for index info + rebuild controls.
function openIndexSection(): void {
  void openSourcesDialog('index')
}

watch(
  () => indexStats.dialogOpenNonce,
  (n, prev) => {
    if (n !== prev) {
      openIndexSection()
    }
  },
)

function triggerLocalFilePick(): void {
  localFileInputRef.value?.click()
}

const emit = defineEmits<{
  localArtifactsLoaded: [loaded: boolean]
  'go-graph': []
}>()

async function onListArtifactsClick(): Promise<void> {
  await shell.fetchArtifactList()
  artifactListDialogOpen.value = true
}

async function onLoadIntoGraphFromDialog(): Promise<void> {
  await artifacts.loadSelected()
  if (artifacts.displayArtifact) {
    emit('go-graph')
    artifactListDialogOpen.value = false
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
  if (tab === 'health' || tab === 'scheduled' || tab === 'enrichment') {
    // Health is static; Scheduled + Enrichment panels fetch their own data on activation.
    return
  }
  if (tab === 'index') {
    // Refresh index stats so the Index section shows current vectors / status.
    void indexStats.refreshIndexStats()
    // Map feed_id (sha) → display title so "Feeds indexed" reads as names.
    const ip = shell.corpusPath.trim()
    if (ip) {
      void fetchCorpusFeeds(ip)
        .then((r) => {
          const m: Record<string, string> = {}
          for (const f of r.feeds) {
            if (f.display_title) m[f.feed_id] = f.display_title
          }
          indexFeedNameMap.value = m
        })
        .catch(() => {
          /* names are best-effort; fall back to short ids */
        })
      // Coverage by month → the chart's "Episodes" series.
      void fetchCorpusCoverage(ip)
        .then((r) => {
          indexCoverageByMonth.value = r.by_month
        })
        .catch(() => {
          /* chart is best-effort */
        })
      // Indexed docs by publish month × doc_type → the chart's per-type series.
      void fetchIndexTimeseries(ip)
        .then((r) => {
          indexTimeseries.value = r
        })
        .catch(() => {
          indexTimeseries.value = null
        })
    }
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
      feedsDetailIndex.value = null
      feedsPanelTab.value = 'list'
    } else if (tab === 'operator' && shell.operatorConfigApiAvailable) {
      // The Profile sub-tab needs packaged profile bodies ("what it brings");
      // fetch once and cache.
      if (profileContents.value.length === 0) {
        try {
          profileContents.value = (await getOperatorProfiles()).profiles
        } catch {
          /* picker still works without bodies; viewer shows a hint */
        }
      }
      if (operatorSourcesLoadedForPath.value === p) {
        return
      }
      const o = await getOperatorConfig(p)
      operatorFileHint.value = o.operator_config_path
      availableProfiles.value = o.available_profiles ?? []
      const sp = splitOperatorYamlProfile(o.content)
      // Preselect the env-driven default profile when the corpus has no
      // saved profile yet (#692, RFC-081 §Layer 1). Server only sets
      // ``default_profile`` when ``PODCAST_DEFAULT_PROFILE`` is on AND the
      // value is in ``available_profiles`` — so this is safe to assign
      // unconditionally without re-validating client-side.
      const defaultFromServer = (o.default_profile ?? '').trim()
      operatorProfileSelected.value =
        sp.profile.trim() || defaultFromServer
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
  sourcesDialogOpen.value = true
}

/** Open configuration dialog: Feeds tab if feeds API is on, else job configuration (YAML) tab. */
async function openSourcesDialogDefault(): Promise<void> {
  const feedsOn = shell.feedsApiAvailable
  const tab: SourcesDialogTab = feedsOn ? 'feeds' : 'operator'
  await openSourcesDialog(tab)
}

function onSourcesDialogOpenChange(next: boolean): void {
  sourcesDialogOpen.value = next
  if (!next) {
    operatorSourcesLoadedForPath.value = ''
  }
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

const viewerBuildRelease = computed((): string => {
  const w = window as Window & { __PODCAST_RELEASE__?: string }
  const runtime = (w.__PODCAST_RELEASE__ ?? '').trim()
  if (runtime) {
    return runtime
  }
  const buildTime = (import.meta.env.VITE_PODCAST_RELEASE as string | undefined)?.trim()
  return buildTime || 'local-dev'
})

const viewerBuildTimestamp = computed((): string => {
  const ts = (__VIEWER_BUILD_TIMESTAMP__ || '').trim()
  if (!ts) {
    return 'unknown-time'
  }
  return ts
})

const viewerBuildHint = computed(
  (): string => `Build: ${viewerBuildRelease.value} @ ${viewerBuildTimestamp.value}`,
)

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

function openFeedDetail(index: number): void {
  if (index < 0 || index >= feedsCrudList.value.length) {
    return
  }
  cancelFeedEdit()
  feedsDetailIndex.value = index
}

async function onFeedDetailSave(entry: FeedApiEntry): Promise<void> {
  const idx = feedsDetailIndex.value
  if (idx == null || idx < 0 || idx >= feedsCrudList.value.length) {
    return
  }
  const next = feedsCrudList.value.slice()
  next[idx] = entry
  feedsCrudList.value = next
  await persistFeedsFromCrud()
  // persist clears editing state; close the drill-in only when the save stuck.
  if (!sourcesError.value) {
    feedsDetailIndex.value = null
  }
}

/** Best-effort global ``max_episodes`` (explicit in the operator YAML body) for
 *  the per-feed override hint chip. Preset-derived defaults aren't parsed here. */
const globalMaxEpisodes = computed<number | null>(() => {
  const m = /^\s*max_episodes:\s*(\d+)\s*$/m.exec(operatorYamlBody.value || '')
  return m ? Number(m[1]) : null
})

/** Bumped after a successful scheduled-job toggle so the section re-fetches. */
const scheduledReloadNonce = ref(0)

/**
 * Enable/disable a scheduled job by rewriting only its ``enabled:`` line in the
 * operator YAML (comments preserved) and PUTting it back — the PUT triggers a
 * server-side ``scheduler.reload()`` so ``next_run_at`` refreshes (#709).
 */
async function onScheduledToggle(name: string, enabled: boolean): Promise<void> {
  const p = shell.corpusPath.trim()
  sourcesBusy.value = true
  sourcesError.value = null
  try {
    const cur = await getOperatorConfig(p)
    const next = toggleScheduledJobEnabled(cur.content, name, enabled)
    if (next == null) {
      sourcesError.value =
        `Couldn't update "${name}" automatically — edit scheduled_jobs in Job Configuration.`
      return
    }
    await putOperatorConfig(p, next)
    // Mirror the just-persisted content into the Job Profile / Configuration tabs
    // so they stay consistent (rather than forcing a re-fetch that could surprise
    // an operator mid-edit). The toggle only changed one `enabled:` line.
    const sp = splitOperatorYamlProfile(next)
    operatorProfileSelected.value = sp.profile.trim() || operatorProfileSelected.value
    operatorYamlBody.value = sp.body
    operatorSourcesLoadedForPath.value = p
    scheduledReloadNonce.value += 1
  } catch (e) {
    sourcesError.value = e instanceof Error ? e.message : String(e)
  } finally {
    sourcesBusy.value = false
  }
}
</script>

<template>
  <div class="shrink-0">
    <div
      v-if="shell.corpusVersionWarning"
      role="alert"
      data-testid="corpus-version-warning-banner"
      class="border-b border-warning/40 bg-warning/10 px-2 py-1 text-[10px] text-warning"
    >
      {{ shell.corpusVersionWarning }}
    </div>
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
      class="h-7 min-w-[12rem] w-[min(45rem,calc(100vw-18rem))] shrink-0 rounded border border-border bg-elevated px-2 py-0.5 font-mono text-[11px] text-elevated-foreground placeholder:text-muted"
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
        @click="openIndexSection"
      >
        Index
      </button>
      <span
        class="max-w-[18rem] shrink-0 truncate rounded border border-border bg-elevated px-1.5 py-0.5 font-mono text-[10px] text-elevated-foreground"
        :title="viewerBuildHint"
        data-testid="status-bar-build-label"
      >
        {{ viewerBuildHint }}
      </span>
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
  </div>

  <AppDialog
    :open="sourcesDialogOpen"
    title="Configuration"
    testid="status-bar-sources-dialog"
    close-testid="sources-dialog-close"
    width-class="w-[min(60rem,96vw)]"
    max-height-class="h-[min(40rem,88vh)]"
    body-class="flex min-h-0 flex-1 flex-col overflow-hidden px-3 pb-3 pt-3"
    @update:open="onSourcesDialogOpenChange"
  >
    <template #header>
      <p
        class="mt-0.5 max-w-full truncate font-mono text-[10px] text-muted"
        :title="viewerBuildHint"
        data-testid="sources-dialog-build-label"
      >
        {{ viewerBuildHint }}
      </p>
    </template>
    <!-- Left sub-nav rail + content column (widened modal). Rail replaces the old
         top tab-strip so per-feed editors / future sections get vertical room. -->
    <div class="flex min-h-0 flex-1 gap-3 overflow-hidden">
      <nav
        class="flex w-44 shrink-0 flex-col gap-0.5 overflow-y-auto pr-1"
        aria-label="Configuration sections"
      >
        <button
          v-if="shell.feedsApiAvailable"
          type="button"
          class="w-full truncate rounded px-2 py-1 text-left text-[11px] hover:bg-overlay"
          :class="sourcesTab === 'feeds' ? 'bg-overlay font-medium' : 'text-muted'"
          data-testid="sources-dialog-tab-feeds"
          @click="void selectSourcesTab('feeds')"
        >
          Feeds
        </button>
        <button
          type="button"
          class="w-full truncate rounded px-2 py-1 text-left text-[11px] hover:bg-overlay"
          :class="sourcesTab === 'enrichment' ? 'bg-overlay font-medium' : 'text-muted'"
          data-testid="sources-dialog-tab-enrichment"
          @click="void selectSourcesTab('enrichment')"
        >
          Enrichment
        </button>
        <button
          v-if="shell.operatorConfigApiAvailable"
          type="button"
          class="w-full truncate rounded px-2 py-1 text-left text-[11px] hover:bg-overlay"
          :class="sourcesTab === 'operator' ? 'bg-overlay font-medium' : 'text-muted'"
          data-testid="sources-dialog-tab-operator"
          @click="void selectSourcesTab('operator')"
        >
          Job Configuration
        </button>
        <button
          v-if="shell.operatorConfigApiAvailable"
          type="button"
          class="w-full truncate rounded px-2 py-1 text-left text-[11px] hover:bg-overlay"
          :class="sourcesTab === 'scheduled' ? 'bg-overlay font-medium' : 'text-muted'"
          data-testid="sources-dialog-tab-scheduled"
          @click="void selectSourcesTab('scheduled')"
        >
          Scheduled
        </button>
        <button
          type="button"
          class="w-full truncate rounded px-2 py-1 text-left text-[11px] hover:bg-overlay"
          :class="sourcesTab === 'index' ? 'bg-overlay font-medium' : 'text-muted'"
          data-testid="sources-dialog-tab-index"
          @click="void selectSourcesTab('index')"
        >
          Index
        </button>
        <button
          type="button"
          class="w-full truncate rounded px-2 py-1 text-left text-[11px] hover:bg-overlay"
          :class="sourcesTab === 'health' ? 'bg-overlay font-medium' : 'text-muted'"
          data-testid="sources-dialog-tab-health"
          @click="void selectSourcesTab('health')"
        >
          Health
        </button>
      </nav>
      <!-- Content column: loading / error banners pinned above the active section. -->
      <div class="flex min-h-0 flex-1 flex-col gap-2 overflow-hidden border-l border-border pl-3">
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
          Pipeline preset + YAML (<code class="rounded bg-overlay px-0.5 font-mono text-[9px]">GET /api/operator-config</code>) live in
          <button
            type="button"
            class="font-medium text-surface-foreground underline decoration-dotted underline-offset-2 hover:decoration-solid"
            data-testid="sources-dialog-jump-to-operator"
            @click="void selectSourcesTab('operator')"
          >
            Job Configuration
          </button>
          (packaged <code class="font-mono text-[9px]">profile:</code> preset + YAML overrides).
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
            <FeedOverrideEditor
              v-if="feedsDetailIndex != null && feedsCrudList[feedsDetailIndex] != null"
              :entry="feedsCrudList[feedsDetailIndex]!"
              :global-max-episodes="globalMaxEpisodes"
              :busy="sourcesBusy"
              @save="onFeedDetailSave"
              @back="feedsDetailIndex = null"
            />
            <template v-else>
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
                      Overrides set — use Configure to edit
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
                      class="rounded border border-border px-1.5 py-0.5 text-[10px] hover:bg-overlay disabled:opacity-40"
                      :disabled="sourcesBusy"
                      :data-testid="`sources-dialog-feeds-row-configure-${idx}`"
                      @click="openFeedDetail(idx)"
                    >
                      Configure
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
            </template>
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
    <!-- Enrichment tab: per-enricher health + last-run + run/re-enable controls (RFC-088). -->
    <div
      v-show="sourcesTab === 'enrichment'"
      class="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto"
    >
      <EnrichmentPanel :corpus-path="shell.corpusPath" />
    </div>
    <div
      v-show="sourcesTab === 'operator' && shell.operatorConfigApiAvailable"
      class="flex min-h-0 flex-1 flex-col gap-2"
    >
      <!-- One page, two sub-tabs: Profile (pick + what it brings) | Configuration (YAML). -->
      <div class="flex w-full min-w-0 shrink-0 gap-1 border-b border-border pb-1.5">
        <button
          type="button"
          class="min-w-0 flex-1 basis-0 truncate rounded px-2 py-0.5 text-center text-[10px] hover:bg-overlay"
          :class="operatorPanelTab === 'profile' ? 'bg-overlay font-medium' : 'text-muted'"
          data-testid="sources-dialog-operator-subtab-profile"
          @click="operatorPanelTab = 'profile'"
        >
          Profile
        </button>
        <button
          type="button"
          class="min-w-0 flex-1 basis-0 truncate rounded px-2 py-0.5 text-center text-[10px] hover:bg-overlay"
          :class="operatorPanelTab === 'config' ? 'bg-overlay font-medium' : 'text-muted'"
          data-testid="sources-dialog-operator-subtab-config"
          @click="operatorPanelTab = 'config'"
        >
          Configuration
        </button>
      </div>

      <!-- Profile sub-panel: picker + "what this profile brings" -->
      <div
        v-show="operatorPanelTab === 'profile'"
        class="flex min-h-0 flex-1 flex-col gap-2"
      >
        <div class="shrink-0 space-y-2">
          <div class="flex flex-wrap items-center gap-2">
            <label
              for="sources-dialog-profile-select"
              class="text-[10px] text-muted shrink-0"
            >Profile</label>
            <select
              id="sources-dialog-profile-select"
              v-model="operatorProfileSelected"
              data-testid="sources-dialog-profile-select"
              class="max-w-[min(100%,16rem)] rounded border border-border bg-elevated px-2 py-1 text-[11px] text-elevated-foreground"
              aria-label="Pipeline profile"
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
          <p class="text-[10px] text-muted leading-snug">
            A <strong class="text-surface-foreground">profile</strong> is a packaged preset of pipeline settings (providers, models, …). It merges first; explicit keys in <strong class="text-surface-foreground">Configuration</strong> override it. <strong>None</strong> = no preset.
          </p>
        </div>
        <div
          class="min-h-0 flex-1 overflow-y-auto rounded border border-border bg-elevated/40 p-2"
          data-testid="sources-dialog-profile-content"
        >
          <p v-if="!operatorProfileSelected" class="text-[10px] text-muted">
            Select a profile to see what it configures.
          </p>
          <template v-else>
            <p class="mb-1 text-[10px] font-medium text-surface-foreground">
              What “{{ operatorProfileSelected }}” brings
            </p>
            <ul
              v-if="selectedProfileSettings.length"
              class="mb-2 space-y-0.5"
              data-testid="sources-dialog-profile-settings"
            >
              <li
                v-for="s in selectedProfileSettings"
                :key="s.key"
                class="flex gap-1 text-[10px]"
              >
                <span class="shrink-0 font-mono text-muted">{{ s.key }}:</span>
                <span class="break-all font-mono text-surface-foreground">{{ s.value }}</span>
              </li>
            </ul>
            <p v-else class="mb-2 text-[10px] text-muted">
              No structured settings found (or profile bodies unavailable).
            </p>
            <details v-if="selectedProfileContent">
              <summary class="cursor-pointer text-[10px] font-medium text-muted hover:text-surface-foreground">
                Full profile YAML
              </summary>
              <pre class="mt-1 max-h-60 overflow-auto whitespace-pre-wrap break-words rounded border border-border bg-canvas/70 p-2 font-mono text-[10px] text-surface-foreground">{{ selectedProfileContent }}</pre>
            </details>
          </template>
        </div>
      </div>

      <!-- Configuration sub-panel: YAML overrides + cron preview -->
      <div
        v-show="operatorPanelTab === 'config'"
        class="flex min-h-0 flex-1 flex-col gap-2"
      >
        <div class="shrink-0 space-y-2">
          <p v-if="operatorFileHint" class="break-all text-[9px] text-muted leading-snug">
            {{ operatorFileHint }}
          </p>
          <p class="text-[10px] text-muted leading-snug">
            YAML overrides (no top-level <code class="font-mono text-[9px]">profile:</code> — set that in the <strong class="text-surface-foreground">Profile</strong> sub-tab). Keys here override the profile; secrets via environment only; RSS / feeds belong in <strong class="text-surface-foreground">Feeds</strong>.
          </p>
        </div>
        <textarea
          v-model="operatorYamlBody"
          data-testid="sources-dialog-operator-textarea"
          class="min-h-0 w-full flex-1 resize-none rounded border border-border bg-elevated p-2 font-mono text-[11px] text-elevated-foreground"
          spellcheck="false"
          aria-label="Job configuration (YAML)"
        />
        <CronSchedulePreview :yaml="operatorYamlBody" />
      </div>

      <!-- Shared Save: persists the profile preset + YAML overrides together. -->
      <div class="shrink-0 border-t border-border pt-2">
        <button
          type="button"
          class="rounded border border-border px-2 py-1 text-[10px] hover:bg-overlay disabled:opacity-40"
          :disabled="sourcesBusy"
          data-testid="sources-dialog-save-overrides"
          @click="void saveOperatorFromDialog()"
        >
          Save
        </button>
      </div>
    </div>
    <div
      v-show="sourcesTab === 'scheduled' && shell.operatorConfigApiAvailable"
      class="flex min-h-0 flex-1 flex-col gap-2"
    >
      <ScheduledJobsSection
        :corpus-path="shell.corpusPath"
        :active="sourcesTab === 'scheduled'"
        :reload-nonce="scheduledReloadNonce"
        :busy="sourcesBusy"
        @toggle="onScheduledToggle"
      />
    </div>
    <div
      v-show="sourcesTab === 'index'"
      class="flex min-h-0 flex-1 flex-col gap-2"
      data-testid="sources-dialog-index-panel"
    >
      <div class="min-h-0 flex-1 space-y-2 overflow-y-auto overscroll-contain">
        <p class="text-[10px] text-muted leading-snug">
          The vector search index (LanceDB) powers Search and the relational
          surfaces. Rebuild after adding episodes or changing the embedding model.
        </p>
        <!-- Reindex reasons / health banner -->
        <div
          v-if="indexStats.indexHealthBanner"
          class="rounded border px-2 py-1.5 text-[10px] leading-snug"
          :class="indexStats.indexHealthBanner.kind === 'warn'
            ? 'border-warning/40 bg-warning/10 text-surface-foreground'
            : 'border-border bg-elevated/40 text-muted'"
          data-testid="sources-dialog-index-banner"
        >
          <p
            v-for="(line, i) in indexStats.indexHealthBanner.lines"
            :key="i"
          >
            {{ line }}
          </p>
        </div>
        <p
          v-if="indexStats.indexError"
          class="rounded border border-danger/40 bg-danger/10 px-2 py-1 text-[10px] text-danger"
        >
          {{ indexStats.indexError }}
        </p>
        <!-- Index facts -->
        <div
          v-if="indexStats.indexRows.length"
          class="rounded border border-border bg-elevated/40 p-2"
          data-testid="sources-dialog-index-stats"
        >
          <ul class="space-y-0.5">
            <li
              v-for="row in indexFactRows"
              :key="row.k"
              class="flex gap-2 text-[10px]"
            >
              <span class="w-28 shrink-0 text-muted">{{ row.k }}</span>
              <span class="break-all font-mono text-surface-foreground">{{ row.v }}</span>
            </li>
          </ul>
          <div
            v-if="indexedFeedNames.length"
            class="mt-1.5 border-t border-border pt-1.5"
          >
            <p class="mb-0.5 text-[10px] text-muted">Feeds indexed ({{ indexedFeedNames.length }})</p>
            <ul class="space-y-0.5">
              <li
                v-for="(name, i) in indexedFeedNames"
                :key="i"
                class="truncate text-[10px] text-surface-foreground"
                :title="name"
              >
                {{ name }}
              </li>
            </ul>
          </div>
          <div
            v-if="indexDocTypeBars.length"
            class="mt-1.5 border-t border-border pt-1.5"
            data-testid="sources-dialog-index-doctypes"
          >
            <p class="mb-1 text-[10px] text-muted">Documents by type</p>
            <ul class="space-y-1">
              <li
                v-for="d in indexDocTypeBars"
                :key="d.k"
                class="text-[10px]"
              >
                <div class="flex items-baseline justify-between gap-2">
                  <span class="truncate text-surface-foreground">{{ d.k }}</span>
                  <span class="shrink-0 font-mono text-muted">{{ d.n.toLocaleString() }}</span>
                </div>
                <div class="mt-0.5 h-1.5 w-full overflow-hidden rounded bg-overlay">
                  <div class="h-full rounded bg-primary" :style="{ width: d.pct + '%' }" />
                </div>
              </li>
            </ul>
          </div>
        </div>
        <p
          v-else
          class="rounded border border-border/60 bg-overlay/40 px-2 py-2 text-[10px] text-muted"
        >
          No index built yet — run <strong class="text-surface-foreground">Full rebuild</strong> below.
        </p>
        <p
          v-if="indexStats.indexEnvelope?.rebuild_in_progress"
          class="text-[10px] text-muted leading-snug"
        >
          Background index job running — stats refresh automatically.
        </p>
        <p
          v-if="indexStats.indexEnvelope?.rebuild_last_error"
          class="text-[10px] text-danger leading-snug"
        >
          Last rebuild error: {{ indexStats.indexEnvelope.rebuild_last_error }}
        </p>
        <IndexTimeseriesChart
          v-if="indexChartLabels.length"
          :labels="indexChartLabels"
          :series="indexChartSeries"
        />
      </div>
      <div class="shrink-0 flex flex-wrap gap-1 border-t border-border pt-2">
        <button
          type="button"
          class="rounded border border-border px-2 py-1 text-[10px] hover:bg-overlay disabled:opacity-40"
          :disabled="!shell.healthStatus || indexStats.indexLoading"
          @click="indexStats.refreshIndexStats()"
        >
          {{ indexStats.indexLoading ? 'Loading…' : 'Refresh' }}
        </button>
        <button
          type="button"
          class="rounded border border-border px-2 py-1 text-[10px] hover:bg-overlay disabled:opacity-40"
          data-testid="index-dialog-update"
          :disabled="indexStats.rebuildActionsDisabled"
          @click="indexStats.requestIndexRebuild(false)"
        >
          {{ indexStats.rebuildSubmitting ? 'Queueing…' : 'Update index' }}
        </button>
        <button
          type="button"
          class="rounded border border-border px-2 py-1 text-[10px] hover:bg-overlay disabled:opacity-40"
          data-testid="index-dialog-full-rebuild"
          :disabled="indexStats.rebuildActionsDisabled"
          @click="indexStats.requestIndexRebuild(true)"
        >
          Full rebuild
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
          <!-- "Graph explore" row retired in Search v3 §S1 (Explore surface merged into Search). -->
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
    </div>
  </AppDialog>


  <AppDialog
    :open="artifactListDialogOpen"
    title="Corpus artifacts"
    testid="artifact-list-dialog"
    close-testid="artifact-list-close"
    width-class="w-[min(100%,32rem)]"
    max-height-class="max-h-[min(80vh,32rem)]"
    @update:open="artifactListDialogOpen = $event"
  >
    <div class="px-4 py-3 text-xs">
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
    </div>
  </AppDialog>
</template>
