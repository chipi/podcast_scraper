import { describe, expect, it } from 'vitest'

import { BANDS, FPS, spectra, SR } from './audioFingerprint'
import { AudioSync, parseMp3Layout, streamSeconds, type AudioReference, type StreamAccess } from './audioSync'

// Deterministic "speech": noise whose spectrum keeps changing, so every window is unique.
function talk(seconds: number, seed: number): Float32Array {
  let x = seed >>> 0
  const out = new Float32Array(Math.round(seconds * SR))
  for (let i = 0; i < out.length; i++) {
    x = (Math.imul(x, 1664525) + 1013904223) >>> 0
    const t = i / SR
    const env = 0.5 + 0.5 * Math.sin(2 * Math.PI * (0.37 + 0.11 * Math.sin(t * 0.7)) * t)
    out[i] = ((x / 2 ** 32) * 2 - 1) * env + 0.3 * Math.sin(2 * Math.PI * (200 + 150 * Math.sin(t * 1.3)) * t)
  }
  return out
}

function concat(...parts: Float32Array[]): Float32Array {
  const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0))
  let at = 0
  for (const p of parts) {
    out.set(p, at)
    at += p.length
  }
  return out
}

function checkpoint(original: Float32Array, t: number, kind: 'talk_start' | 'after_ad' | 'interval') {
  const scale = 20
  const spec = spectra(original.subarray(Math.round(t * SR), Math.round((t + 10) * SR) + 512))
  const n = 10 * FPS
  const bytes = Array.from(spec.subarray(0, n * BANDS), (v) => {
    const q = Math.max(-127, Math.min(127, Math.round(v * scale)))
    return String.fromCharCode(q < 0 ? q + 256 : q)
  })
  return { t, kind, frames: btoa(bytes.join('')) }
}

// MPEG-1 layer III, 64 kbit/s: 8000 bytes a second, so in this fake one byte is one 8 kHz sample.
const HEADER = Uint8Array.from([0xff, 0xfb, 0x50, 0x00])

function fakeStream(stream: Float32Array): StreamAccess & { requests: number } {
  const access = {
    requests: 0,
    async range(start: number, end: number) {
      access.requests++
      if (start === 0 && end === 65535) return { bytes: HEADER, total: stream.length }
      const b = new Uint8Array(8)
      new DataView(b.buffer).setUint32(0, start)
      new DataView(b.buffer).setUint32(4, Math.min(end, stream.length - 1))
      return { bytes: b, total: stream.length }
    },
    async decode(bytes: Uint8Array) {
      const v = new DataView(bytes.buffer, bytes.byteOffset)
      return stream.slice(v.getUint32(0), v.getUint32(4) + 1)
    },
  }
  return access
}

// Each test fingerprints minutes of synthetic audio: ~0.1 s on a laptop, but CI's shared runner ran
// this file 50x slower (42 s; 4.5 s a test against the 5 s default, 2026-10-16). The work is the
// test, so it gets the time.
describe('audio sync on the listener stream', { timeout: 30_000 }, () => {
  const original = talk(240, 7)
  const ad = talk(35, 99)
  // The listener's host put a 35 s ad at 100 s that our copy did not have.
  const stream = concat(original.subarray(0, 100 * SR), ad, original.subarray(100 * SR))
  const reference: AudioReference = {
    scale: 20,
    window_s: 10,
    duration_s: 240,
    checkpoints: [checkpoint(original, 10, 'talk_start'), checkpoint(original, 100, 'after_ad'), checkpoint(original, 180, 'interval')],
  }

  it('finds each checkpoint at this listener offset', async () => {
    const sync = new AudioSync(reference, fakeStream(stream))
    const map = await sync.ensure(240)
    expect(sync.state).toBe('synced')
    expect(map).not.toBeNull()
    expect(map![0][0]).toBe(0)
    expect(map![0][1]).toBeCloseTo(0, 1)
    expect(map![1][0]).toBe(100)
    expect(map![1][1]).toBeCloseTo(35, 1)
  })

  it('only fetches what playback has reached (plus a minute)', async () => {
    const access = fakeStream(stream)
    const sync = new AudioSync(reference, access)
    await sync.ensure(20)
    const early = access.requests
    expect(sync.map()).toEqual([[0, expect.closeTo(0, 1)]])
    await sync.ensure(240)
    expect(access.requests).toBeGreaterThan(early)
  })

  it('stands down on a stream whose bytes do not map to seconds', async () => {
    const vbr = Uint8Array.from([0xff, 0xfb, 0x50, 0x00, ...Array.from('....Xing', (c) => c.charCodeAt(0))])
    expect(parseMp3Layout(vbr, 1000)).toBeNull()
    const access: StreamAccess = {
      range: async () => ({ bytes: vbr, total: 1000 }),
      decode: async () => new Float32Array(0),
    }
    const sync = new AudioSync(reference, access)
    expect(await sync.ensure(240)).toBeNull()
    expect(sync.state).toBe('unavailable')
  })

  it('works when the page cannot see the file size (cross-origin, no exposed Content-Range)', async () => {
    const access = fakeStream(stream)
    const blind: StreamAccess = {
      range: async (a, b) => ({ ...(await access.range(a, b)), total: null }),
      decode: access.decode,
      size: async () => null,
    }
    const map = await new AudioSync(reference, blind).ensure(240)
    expect(map![1][1]).toBeCloseTo(35, 1)
  })

  it('finds the audio behind an ID3 tag larger than the first range (cover art)', async () => {
    const tag = 320_064
    const header = [0x49, 0x44, 0x33, 3, 0, 0, (tag >> 21) & 0x7f, (tag >> 14) & 0x7f, (tag >> 7) & 0x7f, tag & 0x7f]
    const access = fakeStream(stream)
    const asked: number[] = []
    const shifted: StreamAccess = {
      async range(a, b) {
        asked.push(a)
        if (a === 0) return { bytes: Uint8Array.from([...header, ...new Array(65526).fill(0)]), total: null }
        if (a === tag + 10 && b === tag + 10 + 8191) return { bytes: HEADER, total: null }
        return access.range(a - (tag + 10), b - (tag + 10))
      },
      decode: access.decode,
    }
    const map = await new AudioSync(reference, shifted).ensure(240)
    expect(asked[1]).toBe(tag + 10)
    expect(map![1][1]).toBeCloseTo(35, 1)
  })

  it('reads the layout of a constant-bitrate file behind an ID3 tag', () => {
    const id3 = [0x49, 0x44, 0x33, 3, 0, 0, 0, 0, 0, 20, ...new Array(20).fill(0)]
    const layout = parseMp3Layout(Uint8Array.from([...id3, 0xff, 0xfb, 0x90, 0x00]), 30 + 128000 * 60 / 8)
    expect(layout).toEqual({ audioStart: 30, bitrate: 128000, totalBytes: 30 + 960000 })
    expect(streamSeconds(layout!)).toBeCloseTo(60)
  })

  describe('RFC-129 v2: a long window after each break', () => {
    // Built once, outside the tests: synthesising the audio and its spectra is the slow part.
    const long360 = talk(360, 11)
    const small = withLong(long360, 100, 35)
    const big = withLong(long360, 100, 138)

    function withLong(original: Float32Array, at: number, ad: number) {
      const stream = concat(original.subarray(0, at * SR), talk(ad, 99), original.subarray(at * SR))
      const ref: AudioReference = {
        scale: 20,
        window_s: 10,
        reach_s: 90,
        probe_s: 12,
        duration_s: original.length / SR,
        checkpoints: [
          { ...checkpoint(original, 10, 'talk_start'), has_long: true },
          { ...checkpoint(original, at, 'after_ad'), has_long: true },
        ],
      }
      // As the builder cuts them: each ends where the original's next ad starts (here the break).
      const window = (t: number, until: number) =>
        spectra(original.subarray(t * SR, Math.min(original.length, until * SR) + 512))
      const longs = [window(10, at), window(at, at + 192)]
      const long = async (i: number) => longs[i] ?? null
      return { stream, ref, long }
    }

    function recording(stream: Float32Array, blind = false) {
      const access = fakeStream(stream)
      const seconds: number[] = []
      const rec: StreamAccess = {
        async range(a, b) {
          if (!(a === 0 && b === 65535)) seconds.push((b - a + 1) / SR)
          const r = await access.range(a, b)
          return blind ? { ...r, total: null } : r
        },
        decode: access.decode,
        size: async () => null,
      }
      return { rec, seconds }
    }

    it('finds a changed break with short probes, never a wide range', async () => {
      const { stream, ref, long } = small
      const { rec, seconds } = recording(stream)
      const sync = new AudioSync(ref, rec, null, long)
      const map = await sync.ensure(360)
      expect(map![1][0]).toBe(100)
      expect(map![1][1]).toBeCloseTo(35, 1)
      expect(Math.max(...seconds)).toBeLessThanOrEqual(14.1) // 14 s narrow checks, 12 s probes
    })

    it('reaches a jump beyond the window by stepping out (+138 s, as seen on prod)', async () => {
      const { stream, ref, long } = big
      const { rec, seconds } = recording(stream, true) // no file size: no end-of-stream hint
      const sync = new AudioSync(ref, rec, null, long)
      const map = await sync.ensure(360)
      expect(map![1][1]).toBeCloseTo(138, 1)
      expect(Math.max(...seconds)).toBeLessThanOrEqual(14.1)
      expect(seconds.length).toBeLessThanOrEqual(6)
    })

    it('without a long window (offline download, interval) it still finds the break', async () => {
      const { stream, ref } = small
      const map = await new AudioSync(ref, fakeStream(stream), null, null).ensure(360)
      expect(map![1][1]).toBeCloseTo(35, 1)
    })
  })
})

