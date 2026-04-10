/**
 * Node fill colors (v1 parity). Future: map to UXS-001 tokens / CSS variables.
 */
export const GRAPH_NODE_UNKNOWN_FILL = '#868e96'

export const graphNodeTypeStyles = Object.freeze({
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
  'Entity_person',
  'Entity_organization',
  'Podcast',
] as const)

export type GraphVisualType = (typeof graphNodeTypesOrdered)[number] | '?' | 'Entity'

export function graphNodeLegendLabel(key: string): string {
  if (key === 'Entity_person') return 'Entity (person)'
  if (key === 'Entity_organization') return 'Entity (organization)'
  return key
}

export function graphNodeFill(type: string): string {
  const styles = graphNodeTypeStyles as Record<string, { background: string }>
  const s = styles[type] || (type === 'Entity' ? styles.Entity_person : undefined)
  return s ? s.background : GRAPH_NODE_UNKNOWN_FILL
}
