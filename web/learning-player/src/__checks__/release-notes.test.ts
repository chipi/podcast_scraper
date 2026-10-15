/**
 * The release notes (public/releases/) are static pages shipped beside the app (operator
 * 2026-10-15). Nothing builds or type-checks them, so this does: every image and every relative
 * link they reference must be a file that ships, and a relative link must name its file. Inside
 * the iOS and Android apps the pages load from the app's own bundle, where a folder link
 * (`1.0.4/`) does not resolve to its index.html and falls back to the app instead.
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

const ROOT = resolve(__dirname, '../../public/releases')

function pages(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name)
    if (statSync(p).isDirectory()) return pages(p)
    return name.endsWith('.html') ? [p] : []
  })
}

function refs(html: string): string[] {
  return [...html.matchAll(/(?:src|href)="([^"]+)"/g)].map((m) => m[1])
}

describe('release notes pages', () => {
  const all = pages(ROOT)

  it('has the landing page and at least one release', () => {
    expect(all.map((p) => p.slice(ROOT.length))).toEqual(expect.arrayContaining(['/index.html']))
    expect(all.length).toBeGreaterThan(1)
  })

  it.each(all.map((p) => [p.slice(ROOT.length), p]))('%s: every relative reference ships', (_, page) => {
    const html = readFileSync(page, 'utf8')
    const relative = refs(html).filter((r) => !/^(https?:|mailto:|#|\/)/.test(r))
    for (const ref of relative) {
      const file = ref.split('#')[0]
      expect(file, `${ref} names a folder, not a file`).not.toMatch(/\/$/)
      expect(existsSync(resolve(dirname(page), file)), `${ref} is missing`).toBe(true)
    }
  })

  it.each(all.map((p) => [p.slice(ROOT.length), p]))('%s: links into the app use real app paths', (_, page) => {
    const html = readFileSync(page, 'utf8')
    const absolute = refs(html).filter((r) => r.startsWith('/'))
    for (const ref of absolute) expect(['/settings']).toContain(ref)
  })
})
