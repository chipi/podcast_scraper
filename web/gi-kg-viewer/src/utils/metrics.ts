/**
 * Artifact overview metrics (v1 parity with web/gi-kg-viz/shared.js computeMetrics).
 */
import type { ArtifactData, ParsedArtifact } from '../types/artifact'
import { graphNodeLegendLabel } from './colors'
import { visualNodeTypeCounts } from './visualGroup'

export interface MetricRow {
  k: string
  v: string
}

export interface ArtifactMetricsResult {
  rows: MetricRow[]
  edgeTypeCounts: Record<string, number>
  visualNodeTypeCounts: Record<string, number>
}

function edgeTypeCounts(data: ArtifactData): Record<string, number> {
  const edges = Array.isArray(data.edges) ? data.edges : []
  const et: Record<string, number> = {}
  for (const e of edges) {
    const t = e && typeof e.type === 'string' ? e.type : '?'
    et[t] = (et[t] || 0) + 1
  }
  return et
}

export function computeArtifactMetrics(art: ParsedArtifact): ArtifactMetricsResult {
  const rows: MetricRow[] = []
  const edgeTypes = edgeTypeCounts(art.data)
  const nodes = Array.isArray(art.data.nodes) ? art.data.nodes : []
  const visualNodeTypeCountsOut = visualNodeTypeCounts(nodes)

  rows.push({ k: 'File', v: art.name })
  rows.push({
    k: 'Layer',
    v:
      art.kind === 'gi'
        ? 'Grounded insights (GIL)'
        : art.kind === 'kg'
          ? 'Knowledge graph (KG)'
          : art.kind === 'both'
            ? 'GI + KG (combined)'
            : 'Unknown',
  })
  rows.push({ k: 'Episode', v: art.episodeId || '—' })
  rows.push({ k: 'Nodes', v: String(art.nodes) })
  rows.push({ k: 'Edges', v: String(art.edges) })

  const d = art.data

  if (art.kind === 'gi') {
    rows.push({
      k: 'Model',
      v: typeof d.model_version === 'string' ? d.model_version : '—',
    })
    rows.push({
      k: 'Prompt',
      v: typeof d.prompt_version === 'string' ? d.prompt_version : '—',
    })
    let insights = 0
    let groundedTrue = 0
    let quotes = 0
    let speakers = 0
    for (const n of nodes) {
      if (n.type === 'Insight') {
        insights += 1
        if (n.properties && n.properties.grounded === true) {
          groundedTrue += 1
        }
      }
      if (n.type === 'Quote') quotes += 1
      if (n.type === 'Speaker') speakers += 1
    }
    rows.push({ k: 'Insights', v: String(insights) })
    rows.push({ k: 'Grounded (true)', v: String(groundedTrue) })
    rows.push({
      k: 'Not grounded',
      v: String(Math.max(0, insights - groundedTrue)),
    })
    if (insights > 0) {
      const pct = ((100 * groundedTrue) / insights).toFixed(1)
      rows.push({ k: '% grounded', v: `${pct}%` })
    }
    rows.push({ k: 'Quotes', v: String(quotes) })
    if (speakers > 0) {
      rows.push({ k: 'Speakers', v: String(speakers) })
    }
    if (visualNodeTypeCountsOut.Entity_person) {
      rows.push({
        k: graphNodeLegendLabel('Entity_person'),
        v: String(visualNodeTypeCountsOut.Entity_person),
      })
    }
    if (visualNodeTypeCountsOut.Entity_organization) {
      rows.push({
        k: graphNodeLegendLabel('Entity_organization'),
        v: String(visualNodeTypeCountsOut.Entity_organization),
      })
    }
  } else if (art.kind === 'kg') {
    const ex =
      d.extraction && typeof d.extraction === 'object'
        ? (d.extraction as { model_version?: string; extracted_at?: string })
        : {}
    rows.push({ k: 'Extraction', v: ex.model_version || '—' })
    rows.push({ k: 'Extracted at', v: ex.extracted_at || '—' })
    const nt = art.nodeTypes
    if (nt.Topic != null) {
      rows.push({ k: 'Topics', v: String(nt.Topic) })
    }
    const ep = visualNodeTypeCountsOut.Entity_person || 0
    const eo = visualNodeTypeCountsOut.Entity_organization || 0
    if (ep > 0 || eo > 0) {
      if (ep > 0) {
        rows.push({
          k: graphNodeLegendLabel('Entity_person'),
          v: String(ep),
        })
      }
      if (eo > 0) {
        rows.push({
          k: graphNodeLegendLabel('Entity_organization'),
          v: String(eo),
        })
      }
    } else if (nt.Entity != null) {
      rows.push({ k: 'Entities', v: String(nt.Entity) })
    }
  } else if (art.kind === 'both') {
    if (typeof d.model_version === 'string') {
      rows.push({ k: 'GI model', v: d.model_version })
    }
    if (typeof d.prompt_version === 'string') {
      rows.push({ k: 'GI prompt', v: d.prompt_version })
    }
    const ex =
      d.extraction && typeof d.extraction === 'object'
        ? (d.extraction as { model_version?: string; extracted_at?: string })
        : {}
    if (ex.model_version || ex.extracted_at) {
      rows.push({ k: 'KG extraction', v: ex.model_version || '—' })
      if (ex.extracted_at) {
        rows.push({ k: 'KG extracted at', v: ex.extracted_at })
      }
    }
    if (visualNodeTypeCountsOut.Entity_person) {
      rows.push({
        k: graphNodeLegendLabel('Entity_person'),
        v: String(visualNodeTypeCountsOut.Entity_person),
      })
    }
    if (visualNodeTypeCountsOut.Entity_organization) {
      rows.push({
        k: graphNodeLegendLabel('Entity_organization'),
        v: String(visualNodeTypeCountsOut.Entity_organization),
      })
    }
  }

  return {
    rows,
    edgeTypeCounts: edgeTypes,
    visualNodeTypeCounts: visualNodeTypeCountsOut,
  }
}
