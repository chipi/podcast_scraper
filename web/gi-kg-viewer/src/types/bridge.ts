/** RFC-072 per-episode bridge.json (canonical identity registry). */

export interface BridgeIdentity {
  id: string
  type: string
  display_name: string
  aliases: string[]
  sources: { gi: boolean; kg: boolean }
}

export interface BridgeDocument {
  schema_version?: string
  episode_id?: string
  emitted_at?: string
  identities?: BridgeIdentity[]
}
