/**
 * Shareable entity cards (#2036) — the quote-led EDITORIAL card, generalized from the highlight
 * card (`useShareCard`). Renders a show/episode/topic/storyline/person/org to a portrait PNG in the
 * app's own design language (near-black canvas, serif display, mono kickers, ONE accent, square,
 * lots of air) and shares it: Web Share (image) → native text → PNG download. Also exposes a link
 * + text share for the Share menu (card / link / text).
 *
 * Bridge-only (PRD-035 Principle 4): a card carries transcript-derived text + KG metadata only,
 * never source audio.
 */
import { isNative, saveAndShareText } from '../services/native'

/** The design tokens the card draws in — the default dark theme (theme/directions.css). */
const CANVAS = '#07090a'
const FG = '#d6e2d8'
const MUTED = '#7f958a'
const BORDER = '#1e2a28'
const DEFAULT_ACCENT = '#8ad2e5' // --lp-topic; per-kind accent overrides via the model
const SERIF = "Georgia, 'Times New Roman', ui-serif, serif"
const UI = 'Inter, system-ui, -apple-system, sans-serif'
const MONO = "ui-monospace, 'SF Mono', monospace"
const WORDMARK = 'closelistening.app'
const W = 1080
const H = 1440
const PAD = 96

/** The data a card renders — assembled per entity kind by the callers below. */
export interface EntityCardModel {
  kicker: string // "TOPIC" / "EPISODE · CROSS-SHOW"
  title: string
  quote?: string | null // a signature take/insight (optional — card is clean without it)
  byline?: string | null // "— Dr. Elena Fischer" / "42 min · 3 insights"
  stats?: string | null // "28 episodes · 10 voices"
  hot?: string | null // the one accent-coloured stat, e.g. "↑ 2.3× rising"
  accent?: string | null // per-kind accent hex; falls back to the brand cyan
  url?: string | null // canonical link for "Share link"
}

/** The card's text — the Web-Share `text`, the alt text, and the native/text-share fallback. */
export function entityCardText(m: EntityCardModel): string {
  const lines = [m.kicker ? m.kicker.toUpperCase() : '', m.title].filter(Boolean)
  if (m.quote) lines.push(`“${m.quote}”`)
  if (m.byline) lines.push(m.byline)
  const statline = [m.stats, m.hot].filter(Boolean).join(' · ')
  if (statline) lines.push(statline)
  lines.push(WORDMARK)
  return lines.join('\n')
}

function wrap(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string[] {
  const out: string[] = []
  for (const para of text.split('\n')) {
    let line = ''
    for (const word of para.split(/\s+/)) {
      const cand = line ? `${line} ${word}` : word
      if (ctx.measureText(cand).width > maxWidth && line) {
        out.push(line)
        line = word
      } else {
        line = cand
      }
    }
    out.push(line)
  }
  return out
}

/** Render the model to a portrait PNG blob, or null when canvas isn't available (e.g. jsdom). */
export async function renderEntityCard(m: EntityCardModel): Promise<Blob | null> {
  const canvas = document.createElement('canvas')
  canvas.width = W
  canvas.height = H
  const ctx = canvas.getContext('2d')
  if (!ctx) return null
  const accent = m.accent || DEFAULT_ACCENT

  ctx.fillStyle = CANVAS
  ctx.fillRect(0, 0, W, H)
  ctx.strokeStyle = BORDER
  ctx.lineWidth = 2
  ctx.strokeRect(1, 1, W - 2, H - 2)

  const maxW = W - PAD * 2
  let y = PAD + 30

  ctx.textBaseline = 'alphabetic'
  ctx.fillStyle = MUTED
  ctx.font = `26px ${MONO}`
  ctx.fillText(m.kicker.toUpperCase(), PAD, y)
  y += 70

  // Title (serif, wrapped, large).
  ctx.fillStyle = FG
  ctx.font = `700 92px ${SERIF}`
  for (const line of wrap(ctx, m.title, maxW)) {
    y += 96
    ctx.fillText(line, PAD, y)
  }

  // The single accent: a short hairline under the title.
  y += 54
  ctx.fillStyle = accent
  ctx.fillRect(PAD, y, 88, 4)
  y += 4

  if (m.quote) {
    ctx.fillStyle = FG
    ctx.font = `italic 46px ${SERIF}`
    y += 34
    for (const line of wrap(ctx, `“${m.quote}”`, maxW)) {
      y += 64
      ctx.fillText(line, PAD, y)
    }
  }
  if (m.byline) {
    ctx.fillStyle = MUTED
    ctx.font = `30px ${UI}`
    y += 52
    ctx.fillText(m.byline, PAD, y)
  }

  // Footer block, anchored to the bottom: stat line then wordmark.
  const statY = H - PAD - 40
  ctx.font = `26px ${MONO}`
  let x = PAD
  if (m.stats) {
    ctx.fillStyle = MUTED
    ctx.fillText(m.stats.toUpperCase(), x, statY)
    x += ctx.measureText(m.stats.toUpperCase()).width
  }
  if (m.hot) {
    const sep = m.stats ? '   ·   ' : ''
    if (sep) {
      ctx.fillStyle = MUTED
      ctx.fillText(sep, x, statY)
      x += ctx.measureText(sep).width
    }
    ctx.fillStyle = accent
    ctx.fillText(m.hot.toUpperCase(), x, statY)
  }
  // Wordmark with a single accent dot.
  const wy = H - PAD + 8
  ctx.fillStyle = accent
  ctx.beginPath()
  ctx.arc(PAD + 5, wy - 8, 6, 0, Math.PI * 2)
  ctx.fill()
  ctx.fillStyle = MUTED
  ctx.font = `24px ${MONO}`
  ctx.fillText(WORDMARK, PAD + 24, wy)

  return await new Promise<Blob | null>((resolve) => canvas.toBlob((b) => resolve(b), 'image/png'))
}

/** Share the card image: Web Share (files) → native text → PNG download. */
export async function shareEntityCard(m: EntityCardModel): Promise<void> {
  const text = entityCardText(m)
  const blob = await renderEntityCard(m)
  if (blob && typeof navigator !== 'undefined' && 'share' in navigator) {
    const file = new File([blob], 'closelistening-card.png', { type: 'image/png' })
    const nav = navigator as Navigator & { canShare?: (d: unknown) => boolean }
    if (!nav.canShare || nav.canShare({ files: [file] })) {
      try {
        await navigator.share({ files: [file], text })
        return
      } catch {
        /* cancelled/unsupported → fall through */
      }
    }
  }
  if (isNative()) {
    await saveAndShareText('closelistening-card.txt', text, 'text/plain')
    return
  }
  if (blob && typeof document !== 'undefined') {
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'closelistening-card.png'
    a.click()
    URL.revokeObjectURL(url)
  }
}

/** Share the entity's link: Web Share (url) → clipboard copy. Returns 'shared' | 'copied' | 'none'. */
export async function shareEntityLink(m: EntityCardModel): Promise<'shared' | 'copied' | 'none'> {
  const url = m.url
  if (!url) return 'none'
  if (typeof navigator !== 'undefined' && 'share' in navigator) {
    try {
      await navigator.share({ title: m.title, text: m.title, url })
      return 'shared'
    } catch {
      /* cancelled/unsupported → try clipboard */
    }
  }
  if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(url)
      return 'copied'
    } catch {
      return 'none'
    }
  }
  return 'none'
}
