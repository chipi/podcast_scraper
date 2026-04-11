import { describe, expect, it } from 'vitest'
import { graphTypeAvatarLetter } from './graphTypeAvatar'

describe('graphTypeAvatarLetter', () => {
  it('maps entity splits and base types', () => {
    expect(graphTypeAvatarLetter('Entity_person')).toBe('P')
    expect(graphTypeAvatarLetter('Entity_organization')).toBe('O')
    expect(graphTypeAvatarLetter('Insight')).toBe('I')
    expect(graphTypeAvatarLetter('Episode')).toBe('E')
  })
})
