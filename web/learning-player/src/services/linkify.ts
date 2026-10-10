/**
 * Split plain text into text and link segments, for the publisher's episode description.
 *
 * Ingest strips the feed's HTML (`rss/parser.py :: _strip_html`), so an `<a href>` whose text is a
 * word is gone; what survives is a URL the publisher wrote out — about half of prod's descriptions
 * have one (2026-10-10: 1473 of 2921). Rendered as segments, never as HTML: the text is untrusted,
 * and only `http(s)` targets become links.
 */
export type TextSegment = { text: string; href?: string }

const URL_RE = /\b(?:https?:\/\/|www\.)[^\s<>"]+/gi
const TRAILING = /[.,;:!?'"’”]+$/

/** Drop sentence punctuation, and a `)` / `]` with no opener inside the link, off the end. */
function trimTail(url: string): string {
  let u = url
  for (;;) {
    const before = u
    u = u.replace(TRAILING, '')
    for (const [open, close] of [['(', ')'], ['[', ']']] as const) {
      if (u.endsWith(close) && u.split(open).length < u.split(close).length) u = u.slice(0, -1)
    }
    if (u === before) return u
  }
}

export function linkify(text: string): TextSegment[] {
  const out: TextSegment[] = []
  let last = 0
  for (const m of text.matchAll(URL_RE)) {
    const raw = trimTail(m[0])
    const start = m.index ?? 0
    if (!raw || /^www\.$/i.test(raw)) continue
    if (start > last) out.push({ text: text.slice(last, start) })
    out.push({ text: raw, href: /^www\./i.test(raw) ? `https://${raw}` : raw })
    last = start + raw.length
  }
  if (last < text.length) out.push({ text: text.slice(last) })
  return out
}
