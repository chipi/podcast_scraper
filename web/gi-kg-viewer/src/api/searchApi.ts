import { fetchWithTimeout } from './httpClient'

/** Query-time join from ``search/topic_clusters.json`` (``kg_topic`` hits only). */
export interface TopicClusterHitMeta {
  graph_compound_parent_id: string
  canonical_label: string
  cil_alias_target_topic_id?: string
}

export interface SearchHit {
  doc_id: string
  score: number
  metadata: Record<string, unknown>
  text: string
  /**
   * Retrieval tier (PRD-033 FR1.1): 'insight' (synthesized), 'segment' (raw
   * transcript), or 'aux' (kg_entity/kg_topic/quote/summary). Server-derived from
   * metadata.doc_type; defaults to 'aux' against pre-#884 servers.
   */
  source_tier?: string
  supporting_quotes?: Record<string, unknown>[] | null
  /** Chunk-to-Insight lift when the server enriches transcript rows. */
  lifted?: Record<string, unknown> | null
}

/** Counters for the current search response page (after top-k slice). */
export interface CorpusSearchLiftStats {
  transcript_hits_returned: number
  lift_applied: number
}

/** Search v3 §S4b: one aggregated cluster over the response's hit page. */
export interface SearchClusterGroup {
  cluster_id: string | null
  /** ``topic_cluster`` / ``theme_cluster`` / ``topic`` / ``ungrouped``. */
  cluster_kind: string
  label: string
  size: number
  /** Indices into ``results`` (in results order) belonging to this cluster. */
  hit_indices: number[]
}

/** Search v3 §S4b: one cross-Person corroboration pair (ADR-108). */
export interface SearchConsensusPair {
  topic_id: string
  topic_label?: string | null
  person_a_id: string
  person_a_label?: string | null
  person_b_id: string
  person_b_label?: string | null
  insight_a_id: string
  insight_b_id: string
  insight_a_text: string
  insight_b_text: string
  contradiction_score: number
  cosine_similarity?: number | null
}

export interface SearchResponse {
  query: string
  results: SearchHit[]
  error?: string | null
  detail?: string | null
  /**
   * Detected query intent (PRD-033 FR1.4): entity_lookup / raw_evidence /
   * temporal_tracking / cross_show_synthesis / semantic. Null on error or pre-#884
   * servers.
   */
  query_type?: string | null
  lift_stats?: CorpusSearchLiftStats | null
  /** Optional server hint when enrichment was requested but failed (non-fatal). */
  enrichment_error?: string | null
  /** Search v3 §S4b: the operator applied (``cluster`` / ``consensus``); null when
   * the endpoint returned the plain top-k page. */
  operator?: string | null
  /** Populated only when ``operator=cluster``. */
  clusters?: SearchClusterGroup[] | null
  /** Populated only when ``operator=consensus``. */
  consensus_pairs?: SearchConsensusPair[] | null
}

export interface SearchRequestOptions {
  path: string
  types?: string[]
  feed?: string
  since?: string
  speaker?: string
  /** Topic substring (Search v3 SearchTopicChip). Server-side after the follow-up
   * to S1: matches kg_topic id/text and insight ABOUT-edge topic labels. */
  topic?: string
  groundedOnly?: boolean
  topK?: number
  embeddingModel?: string
  /** Default true: collapse duplicate kg_entity/kg_topic surfaces server-side. */
  dedupeKgSurfaces?: boolean
  /** Search v3 §S4b: request a server-side result-set operator pass. When set
   * the server returns the top-k page AS-IS and adds ``clusters`` or
   * ``consensus_pairs`` alongside. Passing an unknown value is a no-op. */
  operator?: 'cluster' | 'consensus'
  /** Search v3 §S5: request the shipped QueryEnricher chain (RFC-088 chunk 5).
   * Server decorates each hit's ``metadata.query_enrichments.related_topics``
   * when the enrichment output is available; response also carries
   * ``enrichment_error`` when the chain failed non-fatally. */
  enrichResults?: boolean
}

export async function searchCorpus(
  q: string,
  options: SearchRequestOptions,
): Promise<SearchResponse> {
  const root = options.path.trim()
  const params = new URLSearchParams({ path: root, q: q.trim() })
  const tk = options.topK ?? 10
  params.set('top_k', String(Math.min(100, Math.max(1, tk))))
  if (options.groundedOnly) params.set('grounded_only', 'true')
  if (options.feed?.trim()) params.set('feed', options.feed.trim())
  if (options.since?.trim()) params.set('since', options.since.trim())
  if (options.speaker?.trim()) params.set('speaker', options.speaker.trim())
  if (options.topic?.trim()) params.set('topic', options.topic.trim())
  if (options.embeddingModel?.trim()) {
    params.set('embedding_model', options.embeddingModel.trim())
  }
  for (const t of options.types ?? []) {
    if (t.trim()) params.append('type', t.trim())
  }
  if (options.dedupeKgSurfaces === false) {
    params.set('dedupe_kg_surfaces', 'false')
  }
  if (options.operator) {
    params.set('operator', options.operator)
  }
  if (options.enrichResults) {
    params.set('enrich_results', 'true')
  }
  const res = await fetchWithTimeout(`/api/search?${params.toString()}`)
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t || `HTTP ${res.status}`)
  }
  return (await res.json()) as SearchResponse
}
