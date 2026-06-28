/**
 * RFC-088 chunk-8 follow-up: small in-memory cache for corpus-scope
 * enrichment envelopes the subject rails consume.
 *
 * The TopicEntityView + PersonLandingView call
 * getCorpusEnrichmentEnvelope on every subject change. The payloads
 * (topic_cooccurrence_corpus / temporal_velocity / grounding_rate /
 * guest_coappearance) are corpus-scope — same JSON for every subject
 * — so a per-(corpusPath, enricherId) cache eliminates the N+1
 * fetch when browsing related subjects.
 *
 * Cache invalidates on corpusPath change. There is no TTL — the
 * payloads are written once per enrichment run and the operator can
 * force a refresh from the viewer's Enrichment tab, which calls
 * invalidateEnrichmentCache() after a re-enable / run.
 */

import {
  getCorpusEnrichmentEnvelope,
  type CorpusEnrichmentEnvelope,
} from '../api/enrichmentApi'

interface CacheEntry {
  // The promise's payload generic is erased here — callers re-narrow via
  // the fetchCachedCorpusEnvelope<TData> generic at the call site.
  promise: Promise<CorpusEnrichmentEnvelope<unknown> | null>
  corpusPath: string
}

const _cache = new Map<string, CacheEntry>()

function key(corpusPath: string, enricherId: string): string {
  return `${corpusPath.trim()}::${enricherId}`
}

export async function fetchCachedCorpusEnvelope<TData = Record<string, unknown>>(
  corpusPath: string,
  enricherId: string,
): Promise<CorpusEnrichmentEnvelope<TData> | null> {
  const k = key(corpusPath, enricherId)
  const hit = _cache.get(k)
  if (hit && hit.corpusPath === corpusPath.trim()) {
    return hit.promise as Promise<CorpusEnrichmentEnvelope<TData> | null>
  }
  const promise = getCorpusEnrichmentEnvelope<TData>(corpusPath, enricherId)
  _cache.set(k, {
    promise: promise as Promise<CorpusEnrichmentEnvelope<unknown> | null>,
    corpusPath: corpusPath.trim(),
  })
  // Drop the cache entry if the underlying fetch rejected so the next
  // call re-tries instead of returning the broken promise forever.
  promise.catch(() => {
    if (_cache.get(k)?.promise === promise) {
      _cache.delete(k)
    }
  })
  return promise
}

/** Invalidate one or all entries. Use after a re-enable / job submit. */
export function invalidateEnrichmentCache(opts?: {
  corpusPath?: string
  enricherId?: string
}): void {
  if (!opts) {
    _cache.clear()
    return
  }
  const targetCorpus = opts.corpusPath?.trim()
  const targetId = opts.enricherId
  for (const k of [..._cache.keys()]) {
    const entry = _cache.get(k)
    if (!entry) continue
    const corpusMatches = !targetCorpus || entry.corpusPath === targetCorpus
    const idMatches = !targetId || k.endsWith(`::${targetId}`)
    if (corpusMatches && idMatches) {
      _cache.delete(k)
    }
  }
}
