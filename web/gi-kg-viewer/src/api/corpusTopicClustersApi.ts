import { dedupeInFlight } from './inFlightDedupe'
import { fetchWithTimeout } from './httpClient'

export type TopicClustersMember = {
  topic_id?: string
  label?: string
  similarity_to_centroid?: number
  episode_ids?: string[]
}

export type TopicClustersCluster = {
  /** Viewer-only compound parent id (`tc:…`). v2 name; v1 used `cluster_id`. */
  graph_compound_parent_id?: string
  /** @deprecated v1 — use `graph_compound_parent_id` */
  cluster_id?: string
  canonical_label?: string
  /** CIL `topic_id_aliases` merge target (`topic:…`). v2 name; v1 used `canonical_topic_id`. */
  cil_alias_target_topic_id?: string
  /** @deprecated v1 — use `cil_alias_target_topic_id` */
  canonical_topic_id?: string
  member_count?: number
  members?: TopicClustersMember[]
}

export type TopicClustersDocument = {
  schema_version?: string
  model?: string
  threshold?: number
  clusters?: TopicClustersCluster[]
  singletons?: number
  topic_count?: number
  cluster_count?: number
}

/** Reader-supported `schema_version` values from ``topic_clusters.json`` writers. */
export const TOPIC_CLUSTERS_SCHEMA_VERSIONS_KNOWN = ['1', '2'] as const

export type TopicClustersFetchResult =
  | {
      status: 'ok'
      document: TopicClustersDocument
      /** Present when ``schema_version`` is set but not in :data:`TOPIC_CLUSTERS_SCHEMA_VERSIONS_KNOWN`. */
      schemaWarning?: string
    }
  | { status: 'missing' }
  | { status: 'error'; message: string }

function corpusQuery(path: string): string {
  const q = new URLSearchParams()
  const t = path.trim()
  if (t) {
    q.set('path', t)
  }
  const s = q.toString()
  return s ? `?${s}` : ''
}

/**
 * Soft validation: unknown ``schema_version`` still returns ``status: 'ok'`` with a warning string
 * (and ``console.warn`` in dev) so older or experimental files keep loading.
 */
export function topicClustersSchemaWarning(doc: TopicClustersDocument): string | undefined {
  const v = doc.schema_version
  if (v == null || String(v).trim() === '') {
    return undefined
  }
  const s = String(v).trim()
  if ((TOPIC_CLUSTERS_SCHEMA_VERSIONS_KNOWN as readonly string[]).includes(s)) {
    return undefined
  }
  return (
    `Unknown topic_clusters schema_version "${s}" ` +
    `(supported: ${TOPIC_CLUSTERS_SCHEMA_VERSIONS_KNOWN.join(', ')}). Viewer may mis-render.`
  )
}

/**
 * Fetch RFC-075 ``topic_clusters.json`` via the viewer API.
 * Prefer this over :func:`fetchTopicClustersDocument` when you need missing vs error vs schema warnings.
 */
export async function fetchTopicClustersFromApi(corpusPath: string): Promise<TopicClustersFetchResult> {
  const url = `/api/corpus/topic-clusters${corpusQuery(corpusPath)}`
  try {
    const res = await dedupeInFlight(url, () =>
      fetchWithTimeout(url, undefined, { timeoutDetail: 'corpus/topic-clusters' }),
    )
    if (res.status === 404) {
      return { status: 'missing' }
    }
    if (!res.ok) {
      const text = await res.text().catch(() => '')
      return {
        status: 'error',
        message: text.trim() || `HTTP ${res.status} topic-clusters`,
      }
    }
    const document = (await res.json()) as TopicClustersDocument
    const schemaWarning = topicClustersSchemaWarning(document)
    if (import.meta.env.DEV && schemaWarning) {
      console.warn(`[corpusTopicClustersApi] ${schemaWarning}`)
    }
    return { status: 'ok', document, schemaWarning }
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e)
    return { status: 'error', message }
  }
}

/**
 * Fetch RFC-075 ``topic_clusters.json`` via the viewer API. Returns null on 404; throws on other errors.
 * @deprecated Prefer :func:`fetchTopicClustersFromApi` for load status and schema warnings.
 */
export async function fetchTopicClustersDocument(
  corpusPath: string,
): Promise<TopicClustersDocument | null> {
  const r = await fetchTopicClustersFromApi(corpusPath)
  if (r.status === 'ok') {
    return r.document
  }
  if (r.status === 'missing') {
    return null
  }
  throw new Error(r.message)
}
