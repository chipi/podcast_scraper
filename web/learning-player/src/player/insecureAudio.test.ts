import { describe, expect, it } from 'vitest'
import { isInsecureAudio } from './insecureAudio'

describe('isInsecureAudio', () => {
  const bbc = 'http://open.live.bbc.co.uk/mediaselector/6/redir/version/2.0/mediaset/x/proto/http/vpid/p0.mp3'

  it('an http source on an https page, in a browser, is blocked for good', () => {
    expect(isInsecureAudio('https:', bbc, false)).toBe(true)
  })

  it('an https source is never the reason', () => {
    expect(isInsecureAudio('https:', 'https://cdn.example.com/a.mp3', false)).toBe(false)
  })

  it('in the apps the page is their own bundle: not this verdict', () => {
    expect(isInsecureAudio('https:', bbc, true)).toBe(false)
  })

  it('a local http page (dev, e2e) can load http audio', () => {
    expect(isInsecureAudio('http:', bbc, false)).toBe(false)
  })
})
