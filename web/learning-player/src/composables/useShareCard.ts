/**
 * Shareable quote cards (PRD-046 FR5 / #1418) — the outward growth loop. Renders a highlight to a
 * square image (quote + guest/topic + episode + timestamp + wordmark) and shares it via the Web
 * Share API, falling back to a PNG download or a native text share.
 *
 * Bridge-only (PRD-035 Principle 4): a card carries the TRANSCRIPT quote + metadata ONLY — never
 * any source audio. The graph refs (guest/topic) make the card carry the graph, not a flat clip.
 */
import type { Highlight } from '../services/types'
import { formatTime } from '../player/transcriptSync'
import { isNative, saveAndShareText } from '../services/native'

const CARD_SIZE = 1080
const WORDMARK = 'closelistening.app'

/** The card's text content — also the Web-Share `text`, the alt text, and the native fallback. */
export function shareCardText(h: Highlight, episodeTitle: string): string {
  const lines: string[] = []
  if (h.quote_text) lines.push(`“${h.quote_text}”`)
  const at = h.start_ms != null ? ` · ${formatTime(h.start_ms / 1000)}` : ''
  lines.push(`— ${episodeTitle}${at}`)
  const refs = (h.graph_refs ?? []).map((r) => r.label)
  if (refs.length) lines.push(refs.join(' · '))
  lines.push(WORDMARK)
  return lines.join('\n')
}

// Split a single over-long spaceless token (CJK/URL) so each piece fits — greedy wrap alone would
// clip it at the canvas edge.
function hardBreak(ctx: CanvasRenderingContext2D, word: string, maxWidth: number): string[] {
  const pieces: string[] = []
  let cur = ''
  for (const ch of word) {
    if (cur && ctx.measureText(cur + ch).width > maxWidth) {
      pieces.push(cur)
      cur = ch
    } else {
      cur += ch
    }
  }
  if (cur) pieces.push(cur)
  return pieces
}

// Wrap text to a pixel width on a canvas context (greedy word wrap; long tokens hard-broken).
function wrapLines(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string[] {
  const out: string[] = []
  let line = ''
  for (const word of text.split(/\s+/)) {
    for (const token of hardBreak(ctx, word, maxWidth)) {
      const candidate = line ? `${line} ${token}` : token
      if (ctx.measureText(candidate).width > maxWidth && line) {
        out.push(line)
        line = token
      } else {
        line = candidate
      }
    }
  }
  if (line) out.push(line)
  return out
}

/** Render the highlight to a PNG blob, or null when canvas isn't available (e.g. jsdom). */
export async function renderCard(h: Highlight, episodeTitle: string): Promise<Blob | null> {
  if (typeof document === 'undefined') return null
  const canvas = document.createElement('canvas')
  canvas.width = CARD_SIZE
  canvas.height = CARD_SIZE
  const ctx = canvas.getContext('2d')
  if (!ctx) return null

  ctx.fillStyle = '#0E0D10'
  ctx.fillRect(0, 0, CARD_SIZE, CARD_SIZE)

  const pad = 96
  const quote = h.quote_text ? `“${h.quote_text}”` : episodeTitle
  ctx.fillStyle = '#F5F4F7'
  ctx.font = '600 56px system-ui, sans-serif'
  const lines = wrapLines(ctx, quote, CARD_SIZE - pad * 2).slice(0, 8)
  lines.forEach((line, i) => ctx.fillText(line, pad, pad + 120 + i * 76))

  ctx.fillStyle = '#9B96A8'
  ctx.font = '400 34px system-ui, sans-serif'
  const at = h.start_ms != null ? ` · ${formatTime(h.start_ms / 1000)}` : ''
  ctx.fillText(`— ${episodeTitle}${at}`.slice(0, 60), pad, CARD_SIZE - pad - 96)

  const refs = (h.graph_refs ?? []).map((r) => r.label).join('  ·  ')
  if (refs) {
    ctx.fillStyle = '#C7B8FF'
    ctx.fillText(refs.slice(0, 60), pad, CARD_SIZE - pad - 40)
  }

  ctx.fillStyle = '#6E6980'
  ctx.font = '600 30px system-ui, sans-serif'
  ctx.fillText(WORDMARK, pad, CARD_SIZE - pad + 8)

  return await new Promise<Blob | null>((resolve) => canvas.toBlob((b) => resolve(b), 'image/png'))
}

/** Build + share a highlight's card. Web Share (files) → native text share → PNG download. */
export async function shareHighlightCard(h: Highlight, episodeTitle: string): Promise<void> {
  const text = shareCardText(h, episodeTitle)
  const blob = await renderCard(h, episodeTitle)

  if (blob && typeof navigator !== 'undefined' && 'share' in navigator) {
    const file = new File([blob], 'closelistening-card.png', { type: 'image/png' })
    const nav = navigator as Navigator & { canShare?: (d: unknown) => boolean }
    if (!nav.canShare || nav.canShare({ files: [file] })) {
      try {
        await navigator.share({ files: [file], text })
        return
      } catch {
        /* user cancelled or unsupported → fall through to the fallbacks */
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
