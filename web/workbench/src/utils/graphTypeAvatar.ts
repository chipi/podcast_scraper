/** Single-letter glyph for a graph visual group (matches graph legend intent). */
export function graphTypeAvatarLetter(visualType: string): string {
  if (visualType === 'TopicCluster') return 'TC'
  if (visualType === 'Entity_person') return 'P'
  if (visualType === 'Entity_organization') return 'O'
  // #2057: without a glyph the legend showed a third colour it never named, so the slate nodes
  // read as "some other thing" rather than as the Object kind.
  if (visualType === 'Entity_object') return 'T'
  const base = visualType.split('_')[0] ?? visualType
  const c = base.trim()[0]
  return c ? c.toUpperCase() : '?'
}
