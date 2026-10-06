import { describe, expect, it } from 'vitest'
import type { CorpusResolvedEpisodeArtifact } from '../api/corpusLibraryApi'
import type { TopicClustersDocument } from '../api/corpusTopicClustersApi'
import { topicEpisodeIdsFromClusters, topicHandoffArtifactPaths } from './topicHandoffEpisodes'

const row = (stem: string, date: string, kg = true) =>
  ({
    episode_id: `ep-${stem}`,
    publish_date: date,
    gi_relative_path: `m/${stem}.gi.json`,
    kg_relative_path: kg ? `m/${stem}.kg.json` : null,
  }) as unknown as CorpusResolvedEpisodeArtifact

describe('topicEpisodeIdsFromClusters', () => {
  it("returns the topic member's episode ids across clusters, once each", () => {
    const doc = {
      clusters: [
        { members: [{ topic_id: 'topic:a', episode_ids: ['e1', ' e2 '] }, { topic_id: 'topic:b', episode_ids: ['e9'] }] },
        { members: [{ topic_id: 'topic:a', episode_ids: ['e2', 'e3'] }] },
      ],
    } as unknown as TopicClustersDocument
    expect(topicEpisodeIdsFromClusters(doc, 'topic:a')).toEqual(['e1', 'e2', 'e3'])
    expect(topicEpisodeIdsFromClusters(doc, 'topic:none')).toEqual([])
    expect(topicEpisodeIdsFromClusters(null, 'topic:a')).toEqual([])
  })
})

describe('topicHandoffArtifactPaths', () => {
  it('takes the newest episodes first, up to the cap', () => {
    const paths = topicHandoffArtifactPaths(
      [row('old', '2024-01-01'), row('new', '2025-06-01'), row('mid', '2025-01-01', false)],
      2,
    )
    expect(paths).toEqual(['m/new.gi.json', 'm/new.kg.json', 'm/mid.gi.json'])
  })
})
