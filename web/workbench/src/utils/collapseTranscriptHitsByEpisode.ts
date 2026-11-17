import type { SearchHit } from '../api/searchApi'

/**
 * A "cluster" of episode-scoped hits that all point at the same episode.
 * Emitted by ``collapseTranscriptHitsByEpisode`` so ``SearchPanel`` can
 * render one card per episode instead of one card per matching chunk /
 * metadata field. The cluster's position in the results list is the
 * position of its FIRST member (which, because ``search.results`` is
 * score-sorted, is the highest-scoring member on that episode).
 *
 * Foldable doc_types (2026-07-22 UX cleanup):
 *   - ``transcript`` — transcript chunk hits.
 *   - ``episode_title`` / ``episode_description`` / ``summary_short`` —
 *     episode-level metadata surfaces (indexer.py rows carry a
 *     ``matched_field`` marker so the card can render "matched: Title /
 *     Description / Summary" chips).
 *
 * Insight / quote / kg_topic / kg_entity / ``summary`` (per-bullet) rows
 * pass through the collapse untouched — the user's mental model is that
 * insight hits are distinct claims worth surfacing individually.
 */
export interface TranscriptClusterHit {
  __kind: 'transcript_cluster'
  episodeId: string
  episodeTitle: string
  feedTitle: string | null
  publishDate: string | null
  metadataRelativePath: string | null
  /** Original transcript hits in the order they appeared in ``results`` (highest score first). */
  members: SearchHit[]
  topScore: number
}

export type CollapsedSearchRow = SearchHit | TranscriptClusterHit

export function isTranscriptClusterHit(
  row: CollapsedSearchRow,
): row is TranscriptClusterHit {
  return (row as TranscriptClusterHit).__kind === 'transcript_cluster'
}

const FOLDABLE_DOC_TYPES = new Set([
  'transcript',
  'episode_title',
  'episode_description',
  'summary_short',
])

function isFoldableHit(hit: SearchHit): boolean {
  const md = (hit.metadata ?? {}) as Record<string, unknown>
  const docType = typeof md.doc_type === 'string' ? md.doc_type : ''
  if (!FOLDABLE_DOC_TYPES.has(docType)) return false
  // Compound hits (transcript segment with a lifted GI insight) render as
  // their own valuable standalone card in ``ResultCard`` — do NOT fold
  // them into a cluster or the "+ insight" compound badge disappears.
  if (docType === 'transcript' && hit.lifted != null && typeof hit.lifted === 'object') {
    return false
  }
  return true
}

/**
 * Group episode-scoped hits by ``metadata.episode_id`` into one
 * ``TranscriptClusterHit`` per episode. Foldable doc_types are the
 * transcript chunks plus the episode-level metadata surfaces
 * (``episode_title`` / ``episode_description`` / ``summary_short``).
 *
 * Passes every other row through unchanged. Never re-orders — insight /
 * quote / kg hits keep their scored position; the cluster occupies the
 * slot of its highest-scoring member.
 *
 * Foldable hits WITHOUT a resolvable ``episode_id`` are also passed
 * through as plain rows so the caller can still see them (they're rare
 * and usually indicate a corpus-side metadata gap).
 */
export function collapseTranscriptHitsByEpisode(
  hits: readonly SearchHit[],
): CollapsedSearchRow[] {
  const out: CollapsedSearchRow[] = []
  const episodeToClusterIndex = new Map<string, number>()

  for (const hit of hits) {
    const md = (hit.metadata ?? {}) as Record<string, unknown>
    const epIdRaw = md.episode_id
    const epId = typeof epIdRaw === 'string' ? epIdRaw.trim() : ''

    if (!isFoldableHit(hit) || !epId) {
      out.push(hit)
      continue
    }

    const existingIdx = episodeToClusterIndex.get(epId)
    if (existingIdx != null) {
      const cluster = out[existingIdx] as TranscriptClusterHit
      cluster.members.push(hit)
      if (hit.score > cluster.topScore) cluster.topScore = hit.score
      continue
    }

    const cluster: TranscriptClusterHit = {
      __kind: 'transcript_cluster',
      episodeId: epId,
      episodeTitle:
        typeof md.episode_title === 'string' && md.episode_title.trim()
          ? md.episode_title.trim()
          : epId,
      feedTitle:
        typeof md.feed_title === 'string' && md.feed_title.trim()
          ? md.feed_title.trim()
          : null,
      publishDate:
        typeof md.publish_date === 'string' && md.publish_date.trim()
          ? md.publish_date.trim()
          : null,
      metadataRelativePath:
        typeof md.source_metadata_relative_path === 'string'
        && md.source_metadata_relative_path.trim()
          ? md.source_metadata_relative_path.trim()
          : null,
      members: [hit],
      topScore: hit.score,
    }
    episodeToClusterIndex.set(epId, out.length)
    out.push(cluster)
  }
  return out
}
