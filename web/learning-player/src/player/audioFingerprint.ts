/**
 * The audio fingerprint of RFC-129, computed exactly as the server does
 * (`podcast_scraper.audio_sync.fingerprint.spectra`): per 20 ms frame of 8 kHz mono audio, the log
 * energy in 16 log-spaced bands, with the frame's mean removed (the spectrum's shape, not its
 * level). The app computes it on short ranges of the stream it plays and lines them up with the
 * reference the server built from the audio we transcribed.
 */

export const SR = 8000
export const HOP = 160
export const NFFT = 512
export const BANDS = 16
export const FPS = SR / HOP

// np.geomspace(4, NFFT // 2, BANDS + 1).astype(int), copied from numpy: it truncates floats, and
// recomputing it here could land a bin off.
const BAND_EDGES = [4, 5, 6, 8, 11, 14, 19, 24, 31, 41, 53, 69, 90, 117, 152, 197, 256]
// np.hanning(NFFT)
const WINDOW = Float64Array.from({ length: NFFT }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (NFFT - 1)))

/** In-place radix-2 FFT of length NFFT. */
function fft(re: Float64Array, im: Float64Array): void {
  const n = re.length
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1
    for (; j & bit; bit >>= 1) j ^= bit
    j ^= bit
    if (i < j) {
      ;[re[i], re[j]] = [re[j], re[i]]
      ;[im[i], im[j]] = [im[j], im[i]]
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len
    const wr = Math.cos(ang)
    const wi = Math.sin(ang)
    for (let i = 0; i < n; i += len) {
      let cr = 1
      let ci = 0
      for (let k = 0; k < len / 2; k++) {
        const a = i + k
        const b = a + len / 2
        const tr = re[b] * cr - im[b] * ci
        const ti = re[b] * ci + im[b] * cr
        re[b] = re[a] - tr
        im[b] = im[a] - ti
        re[a] += tr
        im[a] += ti
        const ncr = cr * wr - ci * wi
        ci = cr * wi + ci * wr
        cr = ncr
      }
    }
  }
}

/** Frames × BANDS, row-major. */
export function spectra(samples: Float32Array): Float32Array {
  const n = samples.length >= NFFT ? Math.floor((samples.length - NFFT) / HOP) + 1 : 0
  const out = new Float32Array(n * BANDS)
  const re = new Float64Array(NFFT)
  const im = new Float64Array(NFFT)
  const bands = new Float64Array(BANDS)
  for (let f = 0; f < n; f++) {
    const off = f * HOP
    for (let i = 0; i < NFFT; i++) {
      re[i] = samples[off + i] * WINDOW[i]
      im[i] = 0
    }
    fft(re, im)
    let mean = 0
    for (let b = 0; b < BANDS; b++) {
      const lo = BAND_EDGES[b]
      const hi = Math.max(BAND_EDGES[b + 1], lo + 1)
      let p = 0
      for (let k = lo; k < hi; k++) p += re[k] * re[k] + im[k] * im[k]
      bands[b] = Math.log10(p + 1e-9)
      mean += bands[b]
    }
    mean /= BANDS
    for (let b = 0; b < BANDS; b++) out[f * BANDS + b] = bands[b] - mean
  }
  return out
}

/** A reference checkpoint's frames (base64 int8, times `scale`) as spectra. */
export function decodeWindow(b64: string, scale: number): Float32Array {
  const bin = atob(b64)
  const out = new Float32Array(bin.length)
  for (let i = 0; i < bin.length; i++) {
    const v = bin.charCodeAt(i)
    out[i] = (v > 127 ? v - 256 : v) / scale
  }
  return out
}

/**
 * Where `window` (frames × BANDS) sits in `spec`: the frame index with the highest mean per-frame
 * correlation, and that correlation (1 for the same audio). As `reference.locate` on the server.
 */
export function locate(window: Float32Array, spec: Float32Array): { frame: number; match: number } {
  const n = window.length / BANDS
  const m = spec.length / BANDS
  if (m < n || n === 0) return { frame: 0, match: -1 }
  const a = new Float64Array(window.length)
  const an = new Float64Array(n)
  for (let f = 0; f < n; f++) {
    let mean = 0
    for (let b = 0; b < BANDS; b++) mean += window[f * BANDS + b]
    mean /= BANDS
    let ss = 0
    for (let b = 0; b < BANDS; b++) {
      const v = window[f * BANDS + b] - mean
      a[f * BANDS + b] = v
      ss += v * v
    }
    an[f] = Math.sqrt(ss) + 1e-9
  }
  // Each stream frame's centred spectrum and norm, once.
  const s = new Float64Array(spec.length)
  const sn = new Float64Array(m)
  for (let f = 0; f < m; f++) {
    let mean = 0
    for (let b = 0; b < BANDS; b++) mean += spec[f * BANDS + b]
    mean /= BANDS
    let ss = 0
    for (let b = 0; b < BANDS; b++) {
      const v = spec[f * BANDS + b] - mean
      s[f * BANDS + b] = v
      ss += v * v
    }
    sn[f] = Math.sqrt(ss) + 1e-9
  }
  let best = -2
  let at = 0
  for (let j = 0; j + n <= m; j++) {
    let total = 0
    for (let f = 0; f < n; f++) {
      let dot = 0
      const ai = f * BANDS
      const si = (j + f) * BANDS
      for (let b = 0; b < BANDS; b++) dot += a[ai + b] * s[si + b]
      total += dot / (an[f] * sn[j + f])
    }
    const c = total / n
    if (c > best) {
      best = c
      at = j
    }
  }
  return { frame: at, match: best }
}
