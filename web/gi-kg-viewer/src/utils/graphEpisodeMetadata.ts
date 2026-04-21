/**
 * Map graph Episode nodes to corpus ``metadata_relative_path`` for Library episode detail.
 */
import type { ParsedArtifact, RawGraphNode } from '../types/artifact'
import { fetchCorpusEpisodes } from '../api/corpusLibraryApi'

function normalizeMetadataRelPath(p: string): string {
  return p.trim().replace(/\\/g, '/')
}

const META_KEYS = ['metadata_relative_path', 'source_metadata_relative_path'] as const

/** Bare episode id from Cytoscape / unified node ids (matches ``ParsedArtifact.episodeId`` when single-episode GI). */
export function logicalEpisodeIdFromGraphNodeId(cyId: string): string | null {
  const s = cyId.trim()
  if (!s) return null
  if (s.startsWith('__unified_ep__:')) {
    const rest = s.slice('__unified_ep__:'.length).trim()
    return rest || null
  }
  const m = s.match(
    /^(?:[gk]:(?:kg:)?(?:ep(?:isode)?):|episode:)(.+)$/,
  )
  if (m?.[1]?.trim()) {
    return m[1].trim()
  }
  return null
}

/**
 * Whether a graph tap should open **Episode** rail (metadata detail) vs graph node detail.
 * When ``rawNode`` is missing — e.g. ``findRawNodeInArtifact`` used only the ego slice while
 * the episode exists on the merged graph — fall back to episode-shaped ``cyId`` so catalog
 * resolution still runs.
 */
export function graphCyIdRepresentsEpisodeNode(
  cyId: string,
  rawNode: RawGraphNode | null,
): boolean {
  if (rawNode?.type === 'Episode') {
    return true
  }
  if (rawNode != null) {
    return false
  }
  return logicalEpisodeIdFromGraphNodeId(cyId) != null
}

export function metadataPathFromEpisodeProperties(node: RawGraphNode | null): string | null {
  if (!node?.properties || typeof node.properties !== 'object') return null
  const p = node.properties as Record<string, unknown>
  for (const k of META_KEYS) {
    const v = p[k]
    if (typeof v === 'string' && v.trim()) {
      return v.trim()
    }
  }
  return null
}

/**
 * ``metadata/foo.gi.json`` → ``metadata/foo.metadata.json`` (same stem as corpus layout).
 */
export function guessMetadataRelPathFromArtifactRelPath(rel: string): string | null {
  const t = rel.trim().replace(/\\/g, '/')
  if (!t) return null
  const lower = t.toLowerCase()
  if (lower.endsWith('.gi.json')) {
    return `${t.slice(0, -'.gi.json'.length)}.metadata.json`
  }
  if (lower.endsWith('.kg.json')) {
    return `${t.slice(0, -'.kg.json'.length)}.metadata.json`
  }
  return null
}

/**
 * Match a logical episode id to one of the loaded GI/KG files and derive metadata path.
 *
 * Prefer ``ParsedArtifact.sourceCorpusRelPath`` (set when each file was fetched) over
 * ``selectedRelPaths[i]``: ``loadSelected`` skips ``.bridge.json`` rows when building
 * ``parsedList``, so parallel indices are not guaranteed to refer to the same file.
 */
export function resolveEpisodeMetadataFromLoadedArtifacts(
  logicalEpisodeId: string,
  parsedList: ParsedArtifact[],
  selectedRelPaths: string[],
): string | null {
  const want = logicalEpisodeId.trim()
  if (!want) return null
  for (let i = 0; i < parsedList.length; i++) {
    const art = parsedList[i]
    const epId = art?.episodeId?.trim()
    if (!epId || epId.startsWith('merged:')) continue
    if (epId !== want) continue
    const fromArtifact =
      (typeof art.sourceCorpusRelPath === 'string' && art.sourceCorpusRelPath.trim()) || ''
    const rel = (fromArtifact || selectedRelPaths[i]?.trim()).trim()
    if (!rel) continue
    let guessed = guessMetadataRelPathFromArtifactRelPath(rel)
    if (guessed) return guessed
    const map = art.sourceCorpusRelPathByEpisodeId
    if (map && typeof map === 'object') {
      const mapped = map[want]
      if (typeof mapped === 'string' && mapped.trim()) {
        guessed = guessMetadataRelPathFromArtifactRelPath(mapped.trim())
        if (guessed) return guessed
      }
    }
  }
  return null
}

/** Paginate corpus episode list until ``episode_id`` matches (or cap pages). */
export async function resolveEpisodeMetadataViaCorpusCatalog(
  corpusPath: string,
  logicalEpisodeId: string,
  maxPages = 8,
  /** When true (e.g. graph node focus changed), stop paging and return null. */
  shouldCancel?: () => boolean,
): Promise<string | null> {
  const want = logicalEpisodeId.trim()
  if (!want || !corpusPath.trim()) return null
  let cursor: string | null = null
  for (let page = 0; page < maxPages; page++) {
    if (shouldCancel?.()) {
      return null
    }
    const body = await fetchCorpusEpisodes(corpusPath.trim(), {
      limit: 200,
      cursor,
    })
    if (shouldCancel?.()) {
      return null
    }
    const hit = body.items.find(
      (it) => it.episode_id === want && it.metadata_relative_path?.trim(),
    )
    if (hit?.metadata_relative_path?.trim()) {
      return hit.metadata_relative_path.trim()
    }
    cursor = body.next_cursor ?? null
    if (!cursor) break
  }
  return null
}

/**
 * Corpus ``metadata_relative_path`` values to compare across Windows/posix.
 */
export function normalizeCorpusMetadataPath(p: string): string {
  return normalizeMetadataRelPath(p)
}

/**
 * Episode nodes in the merged artifact whose ``metadata_relative_path`` (or source variant) matches.
 * Returns stable logical ids for ``resolveCyNodeId`` / library highlight (corpus ``episode_id`` or id stem).
 */
/**
 * Cytoscape node id for an **Episode** row whose metadata path matches ``wantMetadataPath``
 * (merged/filtered artifact ids match the canvas). Deterministic when several rows match.
 */
export function findEpisodeGraphNodeIdForMetadataPath(
  art: ParsedArtifact | null,
  wantMetadataPath: string,
): string | null {
  const want = normalizeCorpusMetadataPath(wantMetadataPath)
  if (!want || !art?.data?.nodes) return null
  const ids: string[] = []
  for (const n of art.data.nodes as RawGraphNode[]) {
    if (!n || n.id == null || n.type !== 'Episode') continue
    const mp = metadataPathFromEpisodeProperties(n)
    if (!mp) continue
    if (normalizeCorpusMetadataPath(mp) !== want) continue
    ids.push(String(n.id))
  }
  if (ids.length === 0) return null
  ids.sort()
  return ids[0] ?? null
}

/**
 * Resolve Episode Cytoscape id for Library / Digest **Open in graph** when corpus ``metadata_relative_path``
 * text does not exactly match graph row properties (encoding / punctuation drift) but ``episode_id`` does.
 */
export function findEpisodeGraphNodeIdForMetadataPathOrEpisodeId(
  art: ParsedArtifact | null,
  wantMetadataPath: string,
  fallbackEpisodeId: string | null | undefined,
): string | null {
  const byMeta = findEpisodeGraphNodeIdForMetadataPath(art, wantMetadataPath)
  if (byMeta) return byMeta
  const eid = fallbackEpisodeId?.trim()
  if (!eid || !art?.data?.nodes) return null
  for (const n of art.data.nodes as RawGraphNode[]) {
    if (!n || n.type !== 'Episode' || n.id == null) continue
    const fromId = logicalEpisodeIdFromGraphNodeId(String(n.id))
    const p = n.properties as Record<string, unknown> | undefined
    const fromProp = typeof p?.episode_id === 'string' ? p.episode_id.trim() : ''
    if ((fromId && fromId === eid) || (fromProp && fromProp === eid)) {
      return String(n.id)
    }
  }
  return null
}

export function logicalEpisodeIdsMatchingMetadataPath(
  art: ParsedArtifact | null,
  wantPath: string,
): string[] {
  const want = normalizeCorpusMetadataPath(wantPath)
  if (!want || !art?.data?.nodes) return []
  const nodes = art.data.nodes as RawGraphNode[]
  const out: string[] = []
  const seen = new Set<string>()
  for (const n of nodes) {
    if (!n || n.type !== 'Episode') continue
    const mp = metadataPathFromEpisodeProperties(n)
    if (!mp) continue
    if (normalizeCorpusMetadataPath(mp) !== want) continue
    const cyId = n.id != null ? String(n.id) : ''
    const logical =
      logicalEpisodeIdFromGraphNodeId(cyId) ||
      (typeof n.properties?.episode_id === 'string'
        ? n.properties.episode_id.trim()
        : '')
    if (logical && !seen.has(logical)) {
      seen.add(logical)
      out.push(logical)
    }
  }
  return out
}

/**
 * Prefer the Cytoscape id already tied to the Episode subject rail (graph double-tap); else scan the artifact.
 */
export function logicalEpisodeIdsForLibraryGraphSync(
  art: ParsedArtifact | null,
  metadataRelativePath: string,
  graphConnectionsCyId: string | null,
): string[] {
  const fromCy = graphConnectionsCyId?.trim()
  if (fromCy) {
    const logical = logicalEpisodeIdFromGraphNodeId(fromCy)
    if (logical) return [logical]
  }
  return logicalEpisodeIdsMatchingMetadataPath(art, metadataRelativePath)
}
