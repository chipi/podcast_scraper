/** Single-letter glyph for a graph visual group (matches graph legend intent). */
export function graphTypeAvatarLetter(visualType: string): string {
  if (visualType === 'TopicCluster') return 'TC'
  if (visualType === 'Entity_person') return 'P'
  if (visualType === 'Entity_organization') return 'O'
  const base = visualType.split('_')[0] ?? visualType
  const c = base.trim()[0]
  return c ? c.toUpperCase() : '?'
}
