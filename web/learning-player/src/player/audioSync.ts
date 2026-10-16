/**
 * Sync on the listener's own stream (RFC-129, #2359).
 *
 * Hosts insert ads per request, so the stream this listener plays can be shifted from our
 * transcript, and shifted differently from anyone else's. The server keeps a reference: fingerprint
 * windows of the audio we transcribed, at the start of talk, after each ad and every few minutes
 * (`GET /episodes/{slug}/audio-reference`). This fetches short byte ranges of the listener's stream
 * near each window, fingerprints them (`audioFingerprint.ts`, identical to the server's) and finds
 * where each window sits: that is this listener's offset there. The offsets form the same
 * `AudioTimeMap` the player already converts through.
 *
 * Checkpoints are resolved lazily, as playback or a seek approaches them, so a listener downloads
 * a ~20 s range per checkpoint they actually reach, not the episode twice. On iOS and Android
 * `fetch` goes through the native stack (CapacitorHttp), which ignores the browser's cross-origin
 * rules; on the web a host must allow cross-origin reads, or the sync reports itself unavailable.
 */

import { decodeWindow, FPS, locate, SR, spectra } from './audioFingerprint'
import type { AudioTimeMap } from './audioTimeMap'

export interface AudioCheckpoint {
  t: number
  kind: 'talk_start' | 'after_ad' | 'interval'
  frames: string
}

export interface AudioReference {
  scale: number
  window_s: number
  duration_s: number
  checkpoints: AudioCheckpoint[]
}

export interface Mp3Layout {
  /** First byte of MPEG audio, after any ID3v2 tag. */
  audioStart: number
  /** Constant bitrate, bits per second. */
  bitrate: number
  totalBytes: number
}

const MIN_MATCH = 0.8
const CHUNK_BEFORE_S = 5
const CHUNK_AFTER_S = 15
const SEARCH_STEP_S = 30
const SEARCH_STEPS = 6 // ±180 s around each guess
const LOOKAHEAD_S = 60

const BITRATES_V1_L3 = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
const BITRATES_V2_L3 = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0]

/**
 * Where the audio starts and its bitrate, from the first bytes of the file. `null` for anything
 * but constant-bitrate MPEG layer III: bytes then do not map to seconds, and the sync stands
 * down (the server's map, if any, still applies).
 */
export function parseMp3Layout(head: Uint8Array, totalBytes: number): Mp3Layout | null {
  let pos = 0
  if (head[0] === 0x49 && head[1] === 0x44 && head[2] === 0x33) {
    const size = ((head[6] & 0x7f) << 21) | ((head[7] & 0x7f) << 14) | ((head[8] & 0x7f) << 7) | (head[9] & 0x7f)
    pos = 10 + size + (head[5] & 0x10 ? 10 : 0)
  }
  while (pos + 4 <= head.length && !(head[pos] === 0xff && (head[pos + 1] & 0xe0) === 0xe0)) pos++
  if (pos + 4 > head.length) return null
  const version = (head[pos + 1] >> 3) & 3 // 3 = MPEG1, 2 = MPEG2, 0 = MPEG2.5
  const layer = (head[pos + 1] >> 1) & 3 // 1 = layer III
  if (layer !== 1 || version === 1) return null
  const kbps = (version === 3 ? BITRATES_V1_L3 : BITRATES_V2_L3)[head[pos + 2] >> 4]
  if (!kbps) return null
  // A Xing tag in the first frame marks variable bitrate ("Info" is the constant-bitrate twin).
  const tag = String.fromCharCode(...head.subarray(pos + 4, Math.min(head.length, pos + 64)))
  if (tag.includes('Xing') || tag.includes('VBRI')) return null
  return { audioStart: pos, bitrate: kbps * 1000, totalBytes }
}

export function streamSeconds(layout: Mp3Layout): number {
  return ((layout.totalBytes - layout.audioStart) * 8) / layout.bitrate
}

function byteAt(layout: Mp3Layout, seconds: number): number {
  return Math.max(layout.audioStart, Math.floor(layout.audioStart + (Math.max(0, seconds) * layout.bitrate) / 8))
}

export interface StreamAccess {
  /** Bytes [start, end] inclusive, and the file's total size when the response says. */
  range(start: number, end: number): Promise<{ bytes: Uint8Array; total: number | null }>
  /** Mono samples at 8 kHz from a run of MP3 frames. */
  decode(bytes: Uint8Array): Promise<Float32Array>
}

/** `fetch` with a Range header, and the platform's decoder resampled to 8 kHz. */
export function browserStreamAccess(url: string): StreamAccess {
  return {
    async range(start, end) {
      const res = await fetch(url, { headers: { Range: `bytes=${start}-${end}` } })
      if (res.status !== 206 && res.status !== 200) throw new Error(`range ${res.status}`)
      const bytes = new Uint8Array(await res.arrayBuffer())
      const total = Number((res.headers.get('content-range') ?? '').split('/')[1])
      return { bytes: res.status === 206 ? bytes : bytes.subarray(start, end + 1), total: Number.isFinite(total) && total > 0 ? total : res.status === 200 ? bytes.length : null }
    },
    async decode(bytes) {
      const Ctx = window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
      const ctx = new Ctx()
      try {
        const buf = await ctx.decodeAudioData(bytes.slice().buffer)
        const off = new OfflineAudioContext(1, Math.max(1, Math.ceil(buf.duration * SR)), SR)
        const src = off.createBufferSource()
        src.buffer = buf
        src.connect(off.destination)
        src.start()
        return (await off.startRendering()).getChannelData(0)
      } finally {
        void ctx.close()
      }
    },
  }
}

export type SyncState = 'starting' | 'syncing' | 'synced' | 'unavailable'

/**
 * One playback's sync. `ensure(t)` resolves every checkpoint up to transcript second `t` (plus a
 * lookahead) and returns the map so far. Never throws: a failure leaves the map as it was.
 */
export class AudioSync {
  state: SyncState = 'starting'
  private layout: Mp3Layout | null = null
  private found = new Map<number, number | null>() // checkpoint index -> offset, or null: not found
  private busy: Promise<AudioTimeMap | null> | null = null
  private windows: Float32Array[]
  private readonly reference: AudioReference
  private readonly access: StreamAccess
  private readonly fallback: AudioTimeMap | null

  constructor(reference: AudioReference, access: StreamAccess, fallback: AudioTimeMap | null = null) {
    this.reference = reference
    this.access = access
    this.fallback = fallback
    this.windows = reference.checkpoints.map((c) => decodeWindow(c.frames, reference.scale))
  }

  /** The map from what has been found so far, or null when nothing has. */
  map(): AudioTimeMap | null {
    const pairs: [number, number][] = []
    this.reference.checkpoints.forEach((c, i) => {
      const off = this.found.get(i)
      if (off == null) return
      const last = pairs[pairs.length - 1]
      if (!last || Math.abs(last[1] - off) > 0.05) pairs.push([c.t, off])
    })
    if (!pairs.length) return null
    pairs[0] = [0, pairs[0][1]]
    return pairs
  }

  ensure(transcriptSeconds: number): Promise<AudioTimeMap | null> {
    if (this.busy) return this.busy.then(() => this.ensure(transcriptSeconds))
    this.busy = this.run(transcriptSeconds).finally(() => {
      this.busy = null
    })
    return this.busy
  }

  private async run(upTo: number): Promise<AudioTimeMap | null> {
    try {
      if (!this.layout) {
        const head = await this.access.range(0, 65535)
        this.layout = head.total ? parseMp3Layout(head.bytes, head.total) : null
        if (!this.layout) {
          this.state = 'unavailable'
          return null
        }
      }
      this.state = 'syncing'
      const endOffset = streamSeconds(this.layout) - this.reference.duration_s
      for (const [i, cp] of this.reference.checkpoints.entries()) {
        if (cp.t > upTo + LOOKAHEAD_S) break
        if (this.found.has(i)) continue
        this.found.set(i, await this.locateCheckpoint(i, endOffset))
      }
      this.state = 'synced'
    } catch {
      this.state = this.found.size ? 'synced' : 'unavailable'
    }
    return this.map()
  }

  private lastOffset(before: number): number | null {
    for (let i = before - 1; i >= 0; i--) {
      const off = this.found.get(i)
      if (off != null) return off
    }
    return null
  }

  private async locateCheckpoint(i: number, endOffset: number): Promise<number | null> {
    const layout = this.layout as Mp3Layout
    const cp = this.reference.checkpoints[i]
    const guesses: number[] = []
    const add = (g: number | null | undefined) => {
      if (g == null || !Number.isFinite(g)) return
      if (!guesses.some((x) => Math.abs(x - g) < SEARCH_STEP_S / 2)) guesses.push(g)
    }
    const last = this.lastOffset(i)
    add(last)
    add(endOffset)
    add(fallbackOffset(this.fallback, cp.t))
    add(0)
    const base = last ?? endOffset
    for (let k = 1; k <= SEARCH_STEPS; k++) {
      add(base + k * SEARCH_STEP_S)
      add(base - k * SEARCH_STEP_S)
    }
    for (const g of guesses) {
      const from = cp.t + g - CHUNK_BEFORE_S
      const to = cp.t + g + this.reference.window_s + CHUNK_AFTER_S
      if (to <= 0) continue
      const start = byteAt(layout, from)
      const end = Math.min(layout.totalBytes - 1, byteAt(layout, to))
      if (end <= start) continue
      const { bytes } = await this.access.range(start, end)
      const samples = await this.access.decode(bytes)
      const hit = locate(this.windows[i], spectra(samples))
      if (hit.match >= MIN_MATCH) {
        const chunkStart = ((start - layout.audioStart) * 8) / layout.bitrate
        return chunkStart + hit.frame / FPS - cp.t
      }
    }
    return null
  }
}

function fallbackOffset(map: AudioTimeMap | null, t: number): number | null {
  if (!map || !map.length) return null
  let off = map[0][1]
  for (const [at, o] of map) if (t >= at) off = o
  return off
}

