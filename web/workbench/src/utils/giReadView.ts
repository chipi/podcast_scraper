/**
 * Read a GI 4.0 file with the GI 3.1 names the Workbench reads (the server's
 * ``podcast_scraper.gi_read_view``). GI 4.0 (R3b) renames ``Episode`` / ``Podcast`` /
 * ``HAS_EPISODE`` to ``ContentItem`` / ``Source`` / ``HAS_ITEM``, renames the evidence fields and
 * nests the podcast-only fields under ``properties.podcast_episode``. Each 3.1 name is filled in
 * from its 4.0 name only when the file does not carry it; a 3.1 file comes back unchanged (the
 * same object). The input is never mutated.
 */
import type { ArtifactData } from '../types/artifact'

const NODE_TYPE_3_1: Record<string, string> = { ContentItem: 'Episode', Source: 'Podcast' }
const EDGE_TYPE_3_1: Record<string, string> = { HAS_ITEM: 'HAS_EPISODE' }

/** 3.1 name <- 4.0 name, per node type. */
const RENAMED: Record<string, [string, string][]> = {
  Episode: [['publish_date', 'published_at']],
  Insight: [['episode_id', 'item_id']],
  Quote: [
    ['episode_id', 'item_id'],
    ['transcript_ref', 'document_ref'],
    ['timestamp_start_ms', 'start_ms'],
    ['timestamp_end_ms', 'end_ms'],
  ],
}

/** Fields GI 4.0 nests under ``properties.podcast_episode`` (ADR-181). */
const NESTED: Record<string, string[]> = {
  Episode: ['podcast_id'],
  Insight: ['speaker', 'speaker_voice_type'],
  Quote: ['speaker_id', 'speaker_name', 'speaker_voice_type'],
}

type Loose = Record<string, unknown>

function isObj(v: unknown): v is Loose {
  return v != null && typeof v === 'object' && !Array.isArray(v)
}

export function isGiV4(data: ArtifactData): boolean {
  const d = data as Loose
  if ('item_id' in d) return true
  const nodes = Array.isArray(d.nodes) ? d.nodes : []
  if (nodes.some((n) => isObj(n) && String(n.type) in NODE_TYPE_3_1)) return true
  const edges = Array.isArray(d.edges) ? d.edges : []
  return edges.some((e) => isObj(e) && String(e.type) in EDGE_TYPE_3_1)
}

function fillProps(props: Loose, kind: string): void {
  for (const [old, next] of RENAMED[kind] ?? []) {
    if (!(old in props) && props[next] != null) props[old] = props[next]
  }
  const nested = props.podcast_episode
  if (isObj(nested)) {
    for (const name of NESTED[kind] ?? []) {
      if (!(name in props) && name in nested) props[name] = nested[name]
    }
  }
}

export function giReadView(data: ArtifactData): ArtifactData {
  if (!isObj(data) || !isGiV4(data)) return data
  const out = structuredClone(data) as Loose
  if (!('episode_id' in out) && out.item_id) out.episode_id = out.item_id
  for (const node of Array.isArray(out.nodes) ? out.nodes : []) {
    if (!isObj(node)) continue
    const kind = NODE_TYPE_3_1[String(node.type)] ?? node.type
    node.type = kind
    if (isObj(node.properties)) fillProps(node.properties, String(kind))
  }
  for (const edge of Array.isArray(out.edges) ? out.edges : []) {
    if (isObj(edge) && String(edge.type) in EDGE_TYPE_3_1) edge.type = EDGE_TYPE_3_1[String(edge.type)]
  }
  return out as ArtifactData
}
