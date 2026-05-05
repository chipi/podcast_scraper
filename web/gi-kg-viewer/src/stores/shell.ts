import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'
import posthog from 'posthog-js'
import { fetchWithTimeout } from '../api/httpClient'
import { StaleGeneration } from '../utils/staleGeneration'

const CORPUS_PATH_STORAGE_KEY = 'ps_corpus_path'

function readInitialCorpusPath(): string {
  try {
    if (typeof localStorage !== 'undefined') {
      const stored = localStorage.getItem(CORPUS_PATH_STORAGE_KEY)
      if (stored != null) {
        return stored
      }
    }
  } catch {
    /* ignore quota / private mode */
  }
  return (import.meta.env.VITE_DEFAULT_CORPUS_PATH as string | undefined) ?? ''
}

export type LeftPanelSurface = 'search' | 'explore'

export const useShellStore = defineStore('shell', () => {
  const corpusPath = ref(readInitialCorpusPath())

  /** Left query column: Search (default) vs Explore mode (`LeftPanel.vue` slide). */
  const leftPanelSurface = ref<LeftPanelSurface>('search')

  function setLeftPanelSurface(surface: LeftPanelSurface): void {
    if (leftPanelSurface.value !== surface) {
      posthog.capture('left_panel_surface_switched', { surface })
    }
    leftPanelSurface.value = surface
  }

  watch(corpusPath, (v) => {
    try {
      if (typeof localStorage !== 'undefined') {
        localStorage.setItem(CORPUS_PATH_STORAGE_KEY, v)
      }
    } catch {
      /* ignore */
    }
    if (v.trim()) {
      posthog.capture('corpus_path_changed')
    }
  })
  const healthStatus = ref<string | null>(null)
  const healthError = ref<string | null>(null)
  /** True only when /api/health reports corpus_library_api (avoids 404 on /api/corpus/* catalog). */
  const corpusLibraryApiAvailable = ref(false)
  /**
   * True when digest is usable: explicit `corpus_digest_api`, or omitted while catalog is
   * available (health JSON from builds before the flag existed — infer digest from library).
   */
  const corpusDigestApiAvailable = ref(false)
  /**
   * Covers / artwork + core HTTP routes: assume **on** until health says otherwise.
   * (Initial `false` made every row “No” if UI rendered before flags were applied, or after HMR.)
   */
  const corpusBinaryApiAvailable = ref(true)
  const artifactsApiAvailable = ref(true)
  const searchApiAvailable = ref(true)
  const exploreApiAvailable = ref(true)
  const indexRoutesApiAvailable = ref(true)
  const corpusMetricsApiAvailable = ref(true)
  /** True when /api/health reports CIL query routes. */
  const cilQueriesApiAvailable = ref(true)
  /** From GET /api/health when server exposes optional search enrichment. */
  const enrichedSearchAvailable = ref(false)
  /** True only when GET/PUT /api/feeds is advertised (strict). */
  const feedsApiAvailable = ref(false)
  /** True only when GET/PUT /api/operator-config is advertised (strict). */
  const operatorConfigApiAvailable = ref(false)
  /** True only when POST/GET /api/jobs is advertised (strict). */
  const jobsApiAvailable = ref(false)
  const artifactsLoading = ref(false)
  const artifactsError = ref<string | null>(null)
  const artifactCount = ref<number | null>(null)
  const artifactList = ref<
    {
      name: string
      relative_path: string
      kind: string
      size_bytes: number
      mtime_utc: string
      publish_date: string
    }[]
  >([])
  /** Server-resolved absolute corpus path (returned by /api/artifacts). */
  const resolvedCorpusPath = ref<string | null>(null)
  /** Server hints (e.g. multi-feed corpus root for unified search index). */
  const corpusHints = ref<string[]>([])

  const hasCorpusPath = computed(() => corpusPath.value.trim().length > 0)

  const healthFetchGate = new StaleGeneration()
  const artifactListFetchGate = new StaleGeneration()

  /** API may send lowercase `ok`; show **OK** in the shell UI. */
  const healthStatusDisplay = computed(() => {
    const raw = healthStatus.value
    if (raw == null || !String(raw).trim()) {
      return ''
    }
    const s = String(raw).trim()
    return s.toLowerCase() === 'ok' ? 'OK' : s
  })

  function healthAdvertisesRoute(value: unknown): boolean {
    return value !== false
  }

  async function fetchHealth(): Promise<void> {
    const seq = healthFetchGate.bump()
    healthError.value = null
    try {
      const res = await fetchWithTimeout('/api/health')
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`)
      }
      const body = (await res.json()) as {
        status?: string
        corpus_library_api?: boolean
        corpus_digest_api?: boolean
        corpus_binary_api?: boolean
        artifacts_api?: boolean
        search_api?: boolean
        explore_api?: boolean
        index_routes_api?: boolean
        corpus_metrics_api?: boolean
        cil_queries_api?: boolean
        enriched_search_available?: boolean
        feeds_api?: boolean
        operator_config_api?: boolean
        jobs_api?: boolean
      }
      if (healthFetchGate.isStale(seq)) {
        return
      }
      const rawStatus = body.status ?? 'unknown'
      const st = typeof rawStatus === 'string' ? rawStatus.trim() : String(rawStatus)
      /** Canonicalize so refetch / server casing does not spuriously notify watchers. */
      healthStatus.value = st.toLowerCase() === 'ok' ? 'ok' : st || 'unknown'
      corpusLibraryApiAvailable.value = body.corpus_library_api === true
      const digestFlag = body.corpus_digest_api
      if (digestFlag === false) {
        corpusDigestApiAvailable.value = false
      } else if (digestFlag === true) {
        corpusDigestApiAvailable.value = true
      } else {
        corpusDigestApiAvailable.value = corpusLibraryApiAvailable.value
      }
      corpusBinaryApiAvailable.value = healthAdvertisesRoute(body.corpus_binary_api)
      artifactsApiAvailable.value = healthAdvertisesRoute(body.artifacts_api)
      searchApiAvailable.value = healthAdvertisesRoute(body.search_api)
      exploreApiAvailable.value = healthAdvertisesRoute(body.explore_api)
      indexRoutesApiAvailable.value = healthAdvertisesRoute(body.index_routes_api)
      corpusMetricsApiAvailable.value = healthAdvertisesRoute(body.corpus_metrics_api)
      cilQueriesApiAvailable.value = healthAdvertisesRoute(body.cil_queries_api)
      enrichedSearchAvailable.value = body.enriched_search_available === true
      feedsApiAvailable.value = body.feeds_api === true
      operatorConfigApiAvailable.value = body.operator_config_api === true
      jobsApiAvailable.value = body.jobs_api === true
    } catch (e) {
      if (healthFetchGate.isStale(seq)) {
        return
      }
      healthStatus.value = null
      corpusLibraryApiAvailable.value = false
      corpusDigestApiAvailable.value = false
      corpusBinaryApiAvailable.value = false
      artifactsApiAvailable.value = false
      searchApiAvailable.value = false
      exploreApiAvailable.value = false
      indexRoutesApiAvailable.value = false
      corpusMetricsApiAvailable.value = false
      cilQueriesApiAvailable.value = false
      enrichedSearchAvailable.value = false
      feedsApiAvailable.value = false
      operatorConfigApiAvailable.value = false
      jobsApiAvailable.value = false
      healthError.value = e instanceof Error ? e.message : String(e)
    }
  }

  async function fetchArtifactList(): Promise<void> {
    const seq = artifactListFetchGate.bump()
    artifactsError.value = null
    artifactCount.value = null
    artifactList.value = []
    resolvedCorpusPath.value = null
    corpusHints.value = []
    if (!hasCorpusPath.value) {
      artifactsError.value = 'Set a corpus directory path (local output folder).'
      artifactsLoading.value = false
      return
    }
    artifactsLoading.value = true
    try {
      const q = new URLSearchParams({ path: corpusPath.value.trim() })
      const res = await fetchWithTimeout(`/api/artifacts?${q.toString()}`)
      if (!res.ok) {
        const detail = await res.text()
        throw new Error(detail || `HTTP ${res.status}`)
      }
      const body = (await res.json()) as {
        path?: string
        hints?: string[]
        artifacts?: {
          name: string
          relative_path: string
          kind: string
          size_bytes: number
          mtime_utc: string
          publish_date: string
        }[]
      }
      if (artifactListFetchGate.isStale(seq)) {
        return
      }
      const list = Array.isArray(body.artifacts) ? body.artifacts : []
      artifactList.value = list
      artifactCount.value = list.length
      if (typeof body.path === 'string' && body.path.trim()) {
        resolvedCorpusPath.value = body.path.trim()
      }
      corpusHints.value = Array.isArray(body.hints) ? body.hints.filter((h) => h.trim()) : []
    } catch (e) {
      if (artifactListFetchGate.isStale(seq)) {
        return
      }
      artifactsError.value = e instanceof Error ? e.message : String(e)
    } finally {
      if (artifactListFetchGate.isCurrent(seq)) {
        artifactsLoading.value = false
      }
    }
  }

  return {
    corpusPath,
    leftPanelSurface,
    setLeftPanelSurface,
    healthStatus,
    healthStatusDisplay,
    healthError,
    corpusLibraryApiAvailable,
    corpusDigestApiAvailable,
    corpusBinaryApiAvailable,
    artifactsApiAvailable,
    searchApiAvailable,
    exploreApiAvailable,
    indexRoutesApiAvailable,
    corpusMetricsApiAvailable,
    cilQueriesApiAvailable,
    enrichedSearchAvailable,
    feedsApiAvailable,
    operatorConfigApiAvailable,
    jobsApiAvailable,
    artifactsLoading,
    artifactsError,
    artifactCount,
    artifactList,
    resolvedCorpusPath,
    corpusHints,
    hasCorpusPath,
    fetchHealth,
    fetchArtifactList,
  }
})
