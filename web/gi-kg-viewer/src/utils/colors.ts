/**
 * Node fill colors — graph-v3 L brand-palette pass. Shifts each type
 * toward a richer, less-neon tone while preserving hue separation so
 * dark + light themes both stay readable. Fixes the pre-L collision
 * where Entity_person and TopicCluster both rendered purple #9775fa;
 * Entity_person now reads amber (humans-as-warm), TopicCluster keeps
 * a distinct violet. Callers of ``graphNodeTypeChrome`` (chips, legend,
 * CIL topic pill) shift with the graph — the palette is a single
 * source of truth per UXS-014.
 */
export const GRAPH_NODE_UNKNOWN_FILL = '#868e96'

export const graphNodeTypeStyles = Object.freeze({
  /** UI **E** chips use the same hex in ``searchResultActionStyles`` — keep in sync. */
  Episode: {
    background: '#4a6bc7',
    border: '#2f4a99',
    labelColor: '#ffffff',
  },
  Insight: {
    background: '#2f9e6a',
    border: '#1f6b48',
    labelColor: '#f5f7fa',
  },
  Quote: {
    background: '#e0a020',
    border: '#a86f10',
    labelColor: '#0d1117',
  },
  Speaker: {
    background: '#7db88e',
    border: '#3f7d59',
    labelColor: '#0d1117',
  },
  Topic: {
    background: '#b078d0',
    border: '#6d3f8a',
    labelColor: '#0d1117',
  },
  TopicCluster: {
    background: '#8574c7',
    border: '#4f3f8a',
    labelColor: '#e9ecef',
  },
  Entity_person: {
    background: '#c49a28',
    border: '#8f6f10',
    labelColor: '#0d1117',
  },
  Entity_organization: {
    background: '#4e9b8f',
    border: '#2f6b62',
    labelColor: '#ffffff',
  },
  Entity: {
    background: '#c49a28',
    border: '#8f6f10',
    labelColor: '#0d1117',
  },
  Podcast: {
    background: '#6b7fbc',
    border: '#3f5199',
    labelColor: '#ffffff',
  },
} as const)

export const graphNodeTypesOrdered = Object.freeze([
  'Episode',
  'Insight',
  'Quote',
  'Speaker',
  'Topic',
  'TopicCluster',
  'Entity_person',
  'Entity_organization',
  'Podcast',
] as const)

export type GraphVisualType = (typeof graphNodeTypesOrdered)[number] | '?' | 'Entity'

export function graphNodeLegendLabel(key: string): string {
  if (key === 'Entity_person') return 'Entity (person)'
  if (key === 'Entity_organization') return 'Entity (organization)'
  if (key === 'TopicCluster') return 'Topic cluster'
  return key
}

export function graphNodeFill(type: string): string {
  const styles = graphNodeTypeStyles as Record<string, { background: string }>
  const s = styles[type] || (type === 'Entity' ? styles.Entity_person : undefined)
  return s ? s.background : GRAPH_NODE_UNKNOWN_FILL
}

type NodeChrome = { background: string; border: string; labelColor: string }

/** Fill / border / label colors aligned with Cytoscape node styling (``visualGroupForNode`` keys). */
/**
 * CIL topic chips (topic-cluster members) when ``clusterMemberAppearance="quote"`` (Episode rail):
 * amber/orange aligned with graph ``Quote`` chrome. Digest Recent uses ``kg`` Tailwind classes instead.
 */
export const cilClusteredTopicPillChrome = Object.freeze({
  borderColor: graphNodeTypeStyles.Quote.border,
  /** Translucent fill — readable on dark and light shells. */
  backgroundColor: 'rgba(250, 176, 5, 0.38)',
} as const)

export function graphNodeTypeChrome(visualType: string): NodeChrome {
  const styles = graphNodeTypeStyles as Record<string, NodeChrome>
  let s = styles[visualType]
  if (!s && visualType === 'Entity') {
    s = styles.Entity_person
  }
  return (
    s ?? {
      background: GRAPH_NODE_UNKNOWN_FILL,
      border: '#495057',
      labelColor: '#ffffff',
    }
  )
}
