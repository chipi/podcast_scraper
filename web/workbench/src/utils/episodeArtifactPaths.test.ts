import { afterEach, describe, expect, it } from 'vitest'

import { episodeArtifactPaths } from '../extensions'
import { episodeArtifactRelPaths, episodeArtifactRelPathsForStem } from './episodeArtifactPaths'

afterEach(() => episodeArtifactPaths.reset())

describe('episodeArtifactRelPaths', () => {
  it('honours the has_kg flag when the payload carries it', () => {
    expect(episodeArtifactRelPaths({ has_kg: true, kg_relative_path: 'a/e.kg.json' })).toEqual(['a/e.kg.json'])
    expect(episodeArtifactRelPaths({ has_kg: false, kg_relative_path: 'a/e.kg.json' })).toEqual([])
  })

  it('takes the KG path when the payload has no flags (node-episodes rows), and no other kind', () => {
    // A GI path is a downstream build's to add (extension point `episodeArtifactPaths`).
    expect(
      episodeArtifactRelPaths({ gi_relative_path: 'a/e.gi.json', kg_relative_path: 'a/e.kg.json' }),
    ).toEqual(['a/e.kg.json'])
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
  it('lists the KG, then extension files', () => {
    expect(episodeArtifactRelPathsForStem('a/e')).toEqual(['a/e.kg.json'])
    episodeArtifactPaths.register({ id: 'x', forStem: (stem) => [`${stem}.x.json`] })
    expect(episodeArtifactRelPathsForStem('a/e')).toEqual(['a/e.kg.json', 'a/e.x.json'])
  })
})
