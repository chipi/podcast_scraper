import { describe, expect, it } from 'vitest'

/**
 * Guardrail — a show's NAME is never rendered without a line cap (operator 2026-09-30).
 *
 * Show names are the raw RSS title, and some are paragraphs: "BRAVE Southeast Asia Tech: Singapore,
 * Indonesia, Vietnam, Philippines, Thailand & Malaysia Startups, Founders & Venture Capital VC
 * (English)" ran eleven lines down the player and the show page. The fix is a safety net, not a
 * shortened name: every place a show name is interpolated sits inside an element that caps it —
 * `.lp-show-name` (two lines, or `--3` / `--4`), or an existing `truncate` / `line-clamp-*`.
 *
 * This finds every template interpolation of a show-name expression and checks the nearest opening
 * tags around it. A new surface that prints a show name uncapped fails here, naming the file.
 */

const components = import.meta.glob('../**/*.vue', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>

/** Expressions that ARE a show's name. */
const SHOW_NAME = /\{\{[^}]*\b(podcast_title|showTitle|currentShowTitle|show\.title|show\?\.title)\b[^}]*\}\}/g
// `sr-only` text is never painted, so it cannot break a layout — and screen readers must get the
// FULL name, so capping it would be wrong.
const CAPPED = /\blp-show-name\b|\btruncate\b|\bline-clamp-\d|\bsr-only\b/

function template(src: string): string {
  const m = src.match(/<template>([\s\S]*)<\/template>\s*(<style|$)/)
  return m ? m[1] : ''
}

/** The last `depth` opening tags before `index` (a crude walk; enough for these templates). */
function openingTagsBefore(html: string, index: number, depth = 2): string[] {
  const tags: string[] = []
  let cursor = index
  while (tags.length < depth && cursor > 0) {
    const start = html.lastIndexOf('<', cursor - 1)
    if (start < 0) break
    const end = html.indexOf('>', start)
    const tag = html.slice(start, end + 1)
    cursor = start
    if (tag.startsWith('</') || tag.startsWith('<!--')) continue
    tags.push(tag)
  }
  return tags
}

describe('show-name safety net', () => {
  it('every show-name interpolation is inside a line-capped element', () => {
    const uncapped: string[] = []
    let seen = 0
    for (const [path, src] of Object.entries(components)) {
      if (path.includes('.test.')) continue
      const html = template(src)
      for (const m of html.matchAll(SHOW_NAME)) {
        seen++
        const tags = openingTagsBefore(html, m.index ?? 0)
        if (!tags.some((t) => CAPPED.test(t))) {
          uncapped.push(`${path.replace('../', 'src/')}: ${m[0].trim()}`)
        }
      }
    }
    // Not vacuous: the player, show page, cards, rows, tiles and Home all print a show name.
    expect(seen).toBeGreaterThanOrEqual(10)
    expect(uncapped, `show names rendered with no line cap:\n${uncapped.join('\n')}`).toEqual([])
  })
})
