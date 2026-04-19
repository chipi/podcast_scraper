/** Raw GI/KG JSON graph shapes (loosely typed). */

export interface RawGraphNode {
  id?: string | number
  type?: string
  properties?: Record<string, unknown>
  /** Cytoscape compound parent id (topic cluster overlay). */
  parent?: string
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
  /**
   * Corpus-root-relative path to the loaded `.gi.json` / `.kg.json` (API load only).
   * Used to resolve bare `transcript_ref` filenames next to the artifact on disk.
   */
  sourceCorpusRelPath?: string | null
  /**
   * When multiple GI files are merged, ``sourceCorpusRelPath`` is null; map each
   * ``episode_id`` (from GI JSON) to that episode's ``.gi.json`` corpus-relative path
   * so Quote nodes can resolve ``transcript_ref`` per episode.
   */
  sourceCorpusRelPathByEpisodeId?: Record<string, string> | null
}

export interface GraphFilterState {
  allowedTypes: Record<string, boolean>
  /** Edge `type` string from artifact; `(unknown)` when missing. */
  allowedEdgeTypes: Record<string, boolean>
  hideUngroundedInsights: boolean
  /** Merged GI+KG graphs (`g:` / `k:` id prefixes); both true by default. */
  showGiLayer: boolean
  showKgLayer: boolean
}
