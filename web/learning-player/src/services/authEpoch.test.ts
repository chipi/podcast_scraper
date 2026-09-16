/**
 * Session-key fingerprint tracking — the signal that separates "my session expired" from "the
 * server invalidated everyone" (incident 2026-09-16).
 */
import { beforeEach, describe, expect, it } from 'vitest'
import { clearAuthEpoch, noteAuthEpoch } from './authEpoch'

beforeEach(() => {
  localStorage.clear()
})

describe('noteAuthEpoch', () => {
  it('a FIRST sighting is not a rotation', () => {
    // A fresh install has no prior key to have been rotated away from. Reporting true here would
    // make every first launch look like an incident.
    expect(noteAuthEpoch('abc123')).toBe(false)
  })

  it('the same epoch again is not a rotation', () => {
    noteAuthEpoch('abc123')
    expect(noteAuthEpoch('abc123')).toBe(false)
  })

  it('a CHANGED epoch is a rotation — every issued token just died at once', () => {
    noteAuthEpoch('abc123')
    expect(noteAuthEpoch('def456')).toBe(true)
  })

  it('ignores null/undefined (auth not configured) rather than storing it', () => {
    noteAuthEpoch('abc123')
    expect(noteAuthEpoch(null)).toBe(false)
    expect(noteAuthEpoch(undefined)).toBe(false)
    // The remembered baseline must survive, or turning auth off and on would read as a rotation.
    expect(noteAuthEpoch('abc123')).toBe(false)
  })

  it('clearing resets the baseline, so the next sighting is a first one', () => {
    noteAuthEpoch('abc123')
    clearAuthEpoch()
    expect(noteAuthEpoch('def456')).toBe(false)
  })
})
