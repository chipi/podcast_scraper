import type { BridgeDocument, BridgeIdentity } from '../types/bridge'

export function normalizeCilIdForBridgeLookup(cyOrBareId: string): string {
  let s = String(cyOrBareId).trim()
  let prev = ''
  while (s !== prev) {
    prev = s
    if (s.startsWith('g:')) {
      s = s.slice(2)
    } else if (s.startsWith('k:') && !s.startsWith('kg:')) {
      s = s.slice(2)
    } else if (s.startsWith('kg:')) {
      s = s.slice(3)
    }
  }
  return s
}

export function parseBridgeDocument(raw: unknown): BridgeDocument | null {
  if (!raw || typeof raw !== 'object') return null
  const o = raw as Record<string, unknown>
  const identitiesRaw = o.identities
  const identities: BridgeIdentity[] = []
  if (Array.isArray(identitiesRaw)) {
    for (const row of identitiesRaw) {
      if (!row || typeof row !== 'object') continue
      const r = row as Record<string, unknown>
      const id = typeof r.id === 'string' ? r.id.trim() : ''
      if (!id) continue
      const type = typeof r.type === 'string' ? r.type : ''
      const display_name = typeof r.display_name === 'string' ? r.display_name : ''
      const aliases: string[] = []
      if (Array.isArray(r.aliases)) {
        for (const a of r.aliases) {
          if (typeof a === 'string' && a.trim()) aliases.push(a.trim())
        }
      }
      const src = r.sources
      let gi = false
      let kg = false
      if (src && typeof src === 'object') {
        const s = src as Record<string, unknown>
        gi = s.gi === true
        kg = s.kg === true
      }
      identities.push({ id, type, display_name, aliases, sources: { gi, kg } })
    }
  }
  return {
    schema_version: typeof o.schema_version === 'string' ? o.schema_version : undefined,
    episode_id: typeof o.episode_id === 'string' ? o.episode_id : undefined,
    emitted_at: typeof o.emitted_at === 'string' ? o.emitted_at : undefined,
    identities,
  }
}

export function bridgeIdentityByIdMap(doc: BridgeDocument | null): Map<string, BridgeIdentity> {
  const m = new Map<string, BridgeIdentity>()
  for (const row of doc?.identities ?? []) {
    if (row?.id) m.set(String(row.id).trim(), row)
  }
  return m
}

/** Human-readable cross-layer line for node detail (GI / KG / both). */
export function crossLayerPresenceLabel(src: { gi: boolean; kg: boolean }): string {
  if (src.gi && src.kg) return 'Grounded Insights and Knowledge graph'
  if (src.gi) return 'Grounded Insights only'
  if (src.kg) return 'Knowledge graph only'
  return ''
}

export function bridgeIdentityForGraphNodeId(
  doc: BridgeDocument | null,
  graphNodeId: string | null | undefined,
): BridgeIdentity | null {
  if (doc == null || graphNodeId == null) return null
  const bare = normalizeCilIdForBridgeLookup(graphNodeId)
  if (!/^(person|org|topic):/.test(bare)) return null
  return bridgeIdentityByIdMap(doc).get(bare) ?? null
}
