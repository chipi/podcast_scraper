import { describe, expect, it } from 'vitest'

import { clusterDisplayLabel } from './topicClusterDisplay'

describe('clusterDisplayLabel (#656 Stage A)', () => {
  it('prefers canonical_label when non-empty', () => {
    expect(
      clusterDisplayLabel({
        canonical_label: 'oil prices',
        cil_alias_target_topic_id: 'topic:oil-prices',
        canonical_topic_id: 'topic:legacy-id',
      }),
    ).toBe('oil prices')
  })

  it('trims whitespace from canonical_label', () => {
    expect(clusterDisplayLabel({ canonical_label: '  ai spending  ' })).toBe('ai spending')
  })

  it('falls back to cil_alias_target_topic_id when canonical_label is absent', () => {
    expect(clusterDisplayLabel({ cil_alias_target_topic_id: 'topic:shadow-fleet' })).toBe(
      'topic:shadow-fleet',
    )
  })

  it('falls back to cil_alias_target_topic_id when canonical_label is whitespace-only', () => {
    // Guard against post-#653 slug writers emitting "" or "   " as
    // canonical_label — the card should not render as an empty pill.
    expect(
      clusterDisplayLabel({
        canonical_label: '   ',
        cil_alias_target_topic_id: 'topic:naval-blockade',
      }),
    ).toBe('topic:naval-blockade')
  })

  it('falls back to legacy canonical_topic_id (v1 field)', () => {
    expect(clusterDisplayLabel({ canonical_topic_id: 'topic:legacy' })).toBe('topic:legacy')
  })

  it('returns literal "Cluster" when no identifier fields are present', () => {
    expect(clusterDisplayLabel({})).toBe('Cluster')
    expect(clusterDisplayLabel({ canonical_label: '', cil_alias_target_topic_id: '' })).toBe(
      'Cluster',
    )
  })

  it('handles the real #655 post-merge shape', () => {
    // Representative of what ``my-manual-run4/search/topic_clusters.json``
    // actually looks like post-#655 — the label is short, the graph
    // compound id is present, and members carry individual topics.
    expect(
      clusterDisplayLabel({
        graph_compound_parent_id: 'tc:oil-prices-cluster',
        canonical_label: 'oil prices',
        cil_alias_target_topic_id: 'topic:oil-prices',
        member_count: 3,
        members: [
          { topic_id: 'topic:oil-prices', label: 'oil prices' },
          { topic_id: 'topic:crude-oil-prices', label: 'crude oil prices' },
          { topic_id: 'topic:brent-prices', label: 'brent prices' },
        ],
      }),
    ).toBe('oil prices')
  })
})
