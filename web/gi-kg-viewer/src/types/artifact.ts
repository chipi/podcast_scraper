/** Raw GI/KG JSON graph shapes (loosely typed). */

export interface RawGraphNode {
  id?: string | number
  type?: string
  properties?: Record<string, unknown>
}

export interface RawGraphEdge {
  type?: string
  from?: string | number
  to?: string | number
}

export interface ArtifactData {
  nodes?: RawGraphNode[]
  edges?: RawGraphEdge[]
  episode_id?: string
  model_version?: string
  prompt_version?: string
  extraction?: { model_version?: string; extracted_at?: string }
  [key: string]: unknown
}

export type ArtifactKind = 'gi' | 'kg' | 'both' | 'unknown'

export interface ParsedArtifact {
  name: string
  kind: ArtifactKind
  episodeId: string | null
  nodes: number
  edges: number
  nodeTypes: Record<string, number>
  data: ArtifactData
}

export interface GraphFilterState {
  allowedTypes: Record<string, boolean>
  hideUngroundedInsights: boolean
  legendSoloVisual: string | null
  /** Merged GI+KG graphs (`g:` / `k:` id prefixes); both true by default. */
  showGiLayer: boolean
  showKgLayer: boolean
}
