import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

export const useShellStore = defineStore('shell', () => {
  const corpusPath = ref(
    (import.meta.env.VITE_DEFAULT_CORPUS_PATH as string | undefined) ?? '',
  )
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
    }[]
  >([])
  /** Server-resolved absolute corpus path (returned by /api/artifacts). */
  const resolvedCorpusPath = ref<string | null>(null)
  /** Server hints (e.g. multi-feed corpus root for unified search index). */
  const corpusHints = ref<string[]>([])

  /**
   * Digest (or other views) sets this before switching to Library; Library consumes once
   * and opens episode detail for this metadata relative path.
   */
  const pendingLibraryMetadataRelpath = ref<string | null>(null)

  const hasCorpusPath = computed(() => corpusPath.value.trim().length > 0)

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

  function setPendingLibraryEpisode(metadataRelativePath: string): void {
    const t = metadataRelativePath.trim()
    pendingLibraryMetadataRelpath.value = t || null
  }

  function takePendingLibraryEpisode(): string | null {
    const v = pendingLibraryMetadataRelpath.value
    pendingLibraryMetadataRelpath.value = null
    return v
  }

  async function fetchHealth(): Promise<void> {
    healthError.value = null
    corpusBinaryApiAvailable.value = true
    artifactsApiAvailable.value = true
    searchApiAvailable.value = true
    exploreApiAvailable.value = true
    indexRoutesApiAvailable.value = true
    corpusMetricsApiAvailable.value = true
    try {
      const res = await fetch('/api/health')
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
      }
      healthStatus.value = body.status ?? 'unknown'
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
    } catch (e) {
      healthStatus.value = null
      corpusLibraryApiAvailable.value = false
      corpusDigestApiAvailable.value = false
      corpusBinaryApiAvailable.value = false
      artifactsApiAvailable.value = false
      searchApiAvailable.value = false
      exploreApiAvailable.value = false
      indexRoutesApiAvailable.value = false
      corpusMetricsApiAvailable.value = false
      healthError.value = e instanceof Error ? e.message : String(e)
    }
  }

  async function fetchArtifactList(): Promise<void> {
    artifactsError.value = null
    artifactCount.value = null
    artifactList.value = []
    resolvedCorpusPath.value = null
    corpusHints.value = []
    if (!hasCorpusPath.value) {
      artifactsError.value = 'Set a corpus directory path (local output folder).'
      return
    }
    artifactsLoading.value = true
    try {
      const q = new URLSearchParams({ path: corpusPath.value.trim() })
      const res = await fetch(`/api/artifacts?${q.toString()}`)
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
        }[]
      }
      const list = Array.isArray(body.artifacts) ? body.artifacts : []
      artifactList.value = list
      artifactCount.value = list.length
      if (typeof body.path === 'string' && body.path.trim()) {
        resolvedCorpusPath.value = body.path.trim()
      }
      corpusHints.value = Array.isArray(body.hints) ? body.hints.filter((h) => h.trim()) : []
    } catch (e) {
      artifactsError.value = e instanceof Error ? e.message : String(e)
    } finally {
      artifactsLoading.value = false
    }
  }

  return {
    corpusPath,
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
    artifactsLoading,
    artifactsError,
    artifactCount,
    artifactList,
    resolvedCorpusPath,
    corpusHints,
    pendingLibraryMetadataRelpath,
    hasCorpusPath,
    fetchHealth,
    fetchArtifactList,
    setPendingLibraryEpisode,
    takePendingLibraryEpisode,
  }
})
