import { afterEach, describe, expect, it } from 'vitest'

import { episodeArtifactPaths } from '../extensions'
import { episodeArtifactRelPaths, episodeArtifactRelPathsForStem } from './episodeArtifactPaths'

afterEach(() => episodeArtifactPaths.reset())

describe('episodeArtifactRelPaths', () => {
  it('honours the has_* flags when the payload carries them', () => {
    const ep = {
      has_gi: false,
      gi_relative_path: 'a/e.gi.json',
      has_kg: true,
      kg_relative_path: 'a/e.kg.json',
    }
    expect(episodeArtifactRelPaths(ep)).toEqual(['a/e.kg.json'])
  })

  it('takes the paths that are set when the payload has no flags (node-episodes rows)', () => {
    expect(
      episodeArtifactRelPaths({ gi_relative_path: ' a/e.gi.json ', kg_relative_path: 'a/e.kg.json' }),
    ).toEqual(['a/e.gi.json', 'a/e.kg.json'])
    expect(episodeArtifactRelPaths({ kg_relative_path: '' })).toEqual([])
  })

  it('appends what extensions add from the same payload, without duplicates', () => {
    episodeArtifactPaths.register({
      id: 'x',
      paths: (ep) => [String(ep.extra_path ?? ''), 'a/e.kg.json'],
    })
    const ep = { has_kg: true, kg_relative_path: 'a/e.kg.json', extra_path: 'a/e.x.json' }
    expect(episodeArtifactRelPaths(ep)).toEqual(['a/e.kg.json', 'a/e.x.json'])
  })
})

describe('episodeArtifactRelPathsForStem', () => {
  it('lists the KG, the GI only when the corpus has GI, then extension files', () => {
    expect(episodeArtifactRelPathsForStem('a/e', { withGi: false })).toEqual(['a/e.kg.json'])
    expect(episodeArtifactRelPathsForStem('a/e', { withGi: true })).toEqual(['a/e.gi.json', 'a/e.kg.json'])
    episodeArtifactPaths.register({ id: 'x', forStem: (stem) => [`${stem}.x.json`] })
    expect(episodeArtifactRelPathsForStem('a/e', { withGi: false })).toEqual(['a/e.kg.json', 'a/e.x.json'])
  })
})
