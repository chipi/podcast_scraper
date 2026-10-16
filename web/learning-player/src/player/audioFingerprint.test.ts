import { describe, expect, it } from 'vitest'

import { BANDS, decodeWindow, locate, spectra, SR } from './audioFingerprint'

// The server's spectra (podcast_scraper.audio_sync.fingerprint) for the same 1 s signal: the app's
// fingerprint must be the server's, or no checkpoint would ever match.
const SERVER: Record<number, number[]> = {
  0: [0.85021, 3.53962, 4.53106, 2.1696, -0.86344, -2.0733, -0.37749, 5.54537, 1.24819, -2.29331, -2.9096, 5.10051, 2.53757, -4.49334, -5.90718, -6.60447],
  7: [0.89207, 3.68511, 4.69943, 2.1935, -0.7409, -1.47652, -0.33094, 5.55333, 1.2549, -2.3239, -3.95582, -2.78581, 5.10965, -0.3675, -4.91545, -6.49117],
  23: [0.84873, 3.63658, 4.65006, 2.14855, -0.82973, -1.7404, -0.39496, 5.50965, 1.21193, -2.33192, -3.7314, -4.7184, -4.83618, 0.53381, 5.06596, -5.02228],
  46: [1.29872, 3.83489, 4.77491, 2.64319, -0.34615, -1.12893, 0.13694, 6.03287, 1.73494, -1.81407, -3.21891, -4.23249, -5.05377, -5.83628, -4.41508, 5.5892],
}

function signal(n: number, phase = 0): Float32Array {
  return Float32Array.from({ length: n }, (_, i) => {
    const t = i / SR + phase
    return (
      0.5 * Math.sin(2 * Math.PI * 440 * t) +
      0.3 * Math.sin(2 * Math.PI * 1250 * t * (1 + t)) +
      0.2 * Math.sin(2 * Math.PI * 97 * t) * Math.cos(2 * Math.PI * 3 * t)
    )
  })
}

describe('audio fingerprint', () => {
  it('is the server fingerprint, frame for frame', () => {
    const s = spectra(signal(SR))
    expect(s.length / BANDS).toBe(47)
    for (const [frame, values] of Object.entries(SERVER)) {
      const f = Number(frame)
      values.forEach((v, b) => expect(s[f * BANDS + b]).toBeCloseTo(v, 3))
    }
  })

  it('finds a window where it sits in a longer stretch', () => {
    const long = spectra(signal(SR * 4))
    const window = long.slice(60 * BANDS, 110 * BANDS)
    const hit = locate(window, long)
    expect(hit.frame).toBe(60)
    expect(hit.match).toBeGreaterThan(0.99)
  })

  it('reads a checkpoint the server wrote (int8, base64, scaled)', () => {
    // bytes 20, -40, 127 at scale 20
    const b64 = btoa(String.fromCharCode(20, 256 - 40, 127))
    expect(Array.from(decodeWindow(b64, 20))).toEqual([1, -2, 6.35].map((v) => Math.fround(v)))
  })
})
