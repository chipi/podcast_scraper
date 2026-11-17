/**
 * Collect Cytoscape graph node ids whose GI/KG provenance belongs to a corpus episode
 * (metadata path and/or episode_id), excluding **Episode** container nodes — used for
 * episode “territory” soft-focus on the graph without requiring Episode type visibility.
 */
import type { ParsedArtifact, RawGraphNode } from '../types/artifact'
import {
  guessMetadataRelPathFromArtifactRelPath,
  logicalEpisodeIdFromGraphNodeId,
  metadataPathFromEpisodeProperties,
  normalizeCorpusMetadataPath,
} from './graphEpisodeMetadata'

function episodeIdOnNode(n: RawGraphNode): string | null {
  const p = n.properties
  if (!p || typeof p !== 'object') return null
  const e = (p as Record<string, unknown>).episode_id
  return typeof e === 'string' && e.trim() ? e.trim() : null
}

function metadataPathsOnNode(n: RawGraphNode): string[] {
  const p = n.properties
  if (!p || typeof p !== 'object') return []
  const out: string[] = []
  for (const k of ['metadata_relative_path', 'source_metadata_relative_path'] as const) {
    const v = (p as Record<string, unknown>)[k]
    if (typeof v === 'string' && v.trim()) {
      out.push(normalizeCorpusMetadataPath(v))
    }
  }
  return out
}

/**
 * Returns raw graph node ids (Cytoscape ids) for non-Episode nodes tied to ``wantMetadataPath``
 * or any matching logical episode id discovered on Episode rows / ``sourceCorpusRelPathByEpisodeId``.
 */
export function collectGraphNodeIdsForEpisodeTerritory(
  art: ParsedArtifact | null,
  wantMetadataPath: string,
): string[] {
  const want = normalizeCorpusMetadataPath(wantMetadataPath)
  if (!want || !art?.data?.nodes) return []

  const logicalIds = new Set<string>()
  const nodes = art.data.nodes as RawGraphNode[]

  for (const n of nodes) {
    if (!n || n.id == null) continue
    if (String(n.type || '') !== 'Episode') continue
    const mp = metadataPathFromEpisodeProperties(n)
    if (mp && normalizeCorpusMetadataPath(mp) === want) {
      const lid =
        logicalEpisodeIdFromGraphNodeId(String(n.id)) || episodeIdOnNode(n)
      if (lid) logicalIds.add(lid)
    }
  }

  const byEp = art.sourceCorpusRelPathByEpisodeId
  if (byEp && typeof byEp === 'object') {
    for (const [eid, relPath] of Object.entries(byEp)) {
      const rp = typeof relPath === 'string' ? relPath.trim() : ''
      if (!rp) continue
      const metaGuess = guessMetadataRelPathFromArtifactRelPath(rp)
      const candidates = [rp, metaGuess].filter(Boolean) as string[]
      for (const c of candidates) {
        if (normalizeCorpusMetadataPath(c) === want) {
          const t = eid.trim()
          if (t) logicalIds.add(t)
          break
        }
      }
    }
  }

  const out = new Set<string>()
  for (const n of nodes) {
    if (!n || n.id == null) continue
    const ty = String(n.type || '')
    if (ty === 'Episode') continue
    const id = String(n.id)
    const eidOn = episodeIdOnNode(n)
    if (eidOn && logicalIds.has(eidOn)) {
      out.add(id)
      continue
    }
    for (const p of metadataPathsOnNode(n)) {
      if (p === want) {
        out.add(id)
        break
      }
    }
  }
  return Array.from(out)
}
