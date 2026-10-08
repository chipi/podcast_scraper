import { describe, expect, it } from 'vitest'
import type { CorpusResolvedEpisodeArtifact } from '../api/corpusLibraryApi'
import { topicHandoffArtifactPaths } from './topicHandoffEpisodes'

const row = (stem: string, date: string, kg = true) =>
  ({
    episode_id: `ep-${stem}`,
    publish_date: date,
    gi_relative_path: `m/${stem}.gi.json`,
    kg_relative_path: kg ? `m/${stem}.kg.json` : null,
  }) as unknown as CorpusResolvedEpisodeArtifact

describe('topicHandoffArtifactPaths', () => {
  it('takes the newest episodes first, up to the cap', () => {
    const paths = topicHandoffArtifactPaths(
      [row('old', '2024-01-01'), row('new', '2025-06-01'), row('mid', '2025-01-01', false)],
      2,
    )
    expect(paths).toEqual(['m/new.gi.json', 'm/new.kg.json', 'm/mid.gi.json'])
  })
})
