/**
 * Read a KG 3.0 file with the KG 2.x names the Workbench reads (the server's
 * ``podcast_scraper.kg.read_view``). R3b rewrites served KG files to 3.0 and news items are KG 3.0
 * from the start: ``ContentItem`` / ``Source`` / ``HAS_ITEM`` / ``CONTRIBUTES_TO`` with a host or
 * guest role, ``item_id``, ``feed_id``, ``published_at`` and ``extraction.document_ref``. Each 2.x
 * name is filled in from its 3.0 name only when the file does not carry it; a 2.x file comes back
 * unchanged (the same object). The input is never mutated.
 */
import type { ArtifactData } from '../types/artifact'

const NODE_TYPE_2: Record<string, string> = { ContentItem: 'Episode', Source: 'Podcast' }
const EDGE_BY_ROLE_2: Record<string, string> = { host: 'HOSTS', guest: 'GUESTS_ON' }

type Loose = Record<string, unknown>

function isObj(v: unknown): v is Loose {
  return v != null && typeof v === 'object' && !Array.isArray(v)
}

export function isKgV3(data: ArtifactData): boolean {
  const d = data as Loose
  if (String(d.schema_version ?? '').startsWith('3.')) return true
  if ('item_id' in d && !('episode_id' in d)) return true
  const nodes = Array.isArray(d.nodes) ? d.nodes : []
  return nodes.some((n) => isObj(n) && String(n.type) in NODE_TYPE_2)
}

export function kgReadView(data: ArtifactData): ArtifactData {
  if (!isObj(data) || !isKgV3(data)) return data
  const out = structuredClone(data) as Loose
  if (!('episode_id' in out) && out.item_id) out.episode_id = out.item_id
  for (const node of Array.isArray(out.nodes) ? out.nodes : []) {
    if (!isObj(node) || !(String(node.type) in NODE_TYPE_2)) continue
    node.type = NODE_TYPE_2[String(node.type)]
    const props = node.properties
    if (node.type === 'Episode' && isObj(props)) {
      if (!('podcast_id' in props) && props.feed_id) props.podcast_id = props.feed_id
      if (!('publish_date' in props) && props.published_at) props.publish_date = props.published_at
    }
  }
  for (const edge of Array.isArray(out.edges) ? out.edges : []) {
    if (!isObj(edge)) continue
    if (edge.type === 'HAS_ITEM') edge.type = 'HAS_EPISODE'
    else if (edge.type === 'CONTRIBUTES_TO') {
      const role = String((isObj(edge.properties) ? edge.properties.role : '') ?? '')
      if (role in EDGE_BY_ROLE_2) edge.type = EDGE_BY_ROLE_2[role]
    }
  }
  const ex = out.extraction
  if (isObj(ex) && !('transcript_ref' in ex) && ex.document_ref) ex.transcript_ref = ex.document_ref
  return out as ArtifactData
}
