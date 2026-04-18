/**
 * Node fill colors (v1 parity). Future: map to UXS-001 tokens / CSS variables.
 */
export const GRAPH_NODE_UNKNOWN_FILL = '#868e96'

export const graphNodeTypeStyles = Object.freeze({
  /** UI **E** chips use the same hex in ``searchResultActionStyles`` — keep in sync. */
  Episode: {
    background: '#4c6ef5',
    border: '#364fc7',
    labelColor: '#ffffff',
  },
  Insight: {
    background: '#40c057',
    border: '#2f9e44',
    labelColor: '#0d1117',
  },
  Quote: {
    background: '#fab005',
    border: '#e67700',
    labelColor: '#0d1117',
  },
  Speaker: {
    background: '#69db7c',
    border: '#2b8a3e',
    labelColor: '#0d1117',
  },
  Topic: {
    background: '#da77f2',
    border: '#862e9c',
    labelColor: '#0d1117',
  },
  TopicCluster: {
    background: '#9775fa',
    border: '#5f3dc4',
    labelColor: '#e9ecef',
  },
  Entity_person: {
    background: '#9775fa',
    border: '#5f3dc4',
    labelColor: '#ffffff',
  },
  Entity_organization: {
    background: '#12b886',
    border: '#087f5b',
    labelColor: '#ffffff',
  },
  Entity: {
    background: '#9775fa',
    border: '#5f3dc4',
    labelColor: '#ffffff',
  },
  Podcast: {
    background: '#748ffc',
    border: '#4263eb',
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
 * RFC-075 CIL topic chips in Digest / Episode rail: same amber/orange as graph
 * ``node.search-hit`` emphasis and ``Quote`` node chrome so cluster membership is obvious.
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
