import { describe, expect, it } from 'vitest'
import type { SearchHit } from '../api/searchApi'
import { sourceMetadataRelativePathFromSearchHit } from './searchHitLibrary'

describe('sourceMetadataRelativePathFromSearchHit', () => {
  it('returns trimmed path when present', () => {
    const hit: SearchHit = {
      doc_id: 'd',
      score: 1,
      text: 't',
      metadata: { source_metadata_relative_path: '  metadata/ep.metadata.json  ' },
    }
    expect(sourceMetadataRelativePathFromSearchHit(hit)).toBe('metadata/ep.metadata.json')
  })

  it('returns null when missing or blank', () => {
    const base: SearchHit = { doc_id: 'd', score: 1, text: 't', metadata: {} }
    expect(sourceMetadataRelativePathFromSearchHit(base)).toBeNull()
    expect(
      sourceMetadataRelativePathFromSearchHit({
        ...base,
        metadata: { source_metadata_relative_path: '   ' },
      }),
    ).toBeNull()
  })
})
