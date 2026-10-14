import { describe, expect, it } from 'vitest'
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join, relative, resolve } from 'node:path'

/**
 * Hover styles only where a pointer can hover (operator 2026-10-14: an orange card border "only
 * when we drag the screen" — "make sure we don't have this anywhere else").
 *
 * A touch screen has no hover to end: iOS applies `:hover` to whatever a touch lands on, the start
 * of a drag included, and keeps it until the next tap elsewhere. So every hover style was, on a
 * phone, a state stuck on the last card touched. Tailwind's `hover:` is guarded by
 * `future.hoverOnlyWhenSupported` (it then emits `@media (hover: hover) and (pointer: fine)`);
 * hand-written `:hover` rules must sit inside the same media query.
 */

const ROOT = resolve(process.cwd())
const SRC = join(ROOT, 'src')

function styleFiles(dir: string, acc: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const full = join(dir, name)
    if (statSync(full).isDirectory()) {
      if (name === 'node_modules' || name === '__checks__') continue
      styleFiles(full, acc)
      continue
    }
    if (/\.(css|vue)$/.test(name)) acc.push(full)
  }
  return acc
}

/** The CSS of a file: a stylesheet whole, a component's `<style>` blocks. */
function css(file: string): string {
  const text = readFileSync(file, 'utf8')
  if (file.endsWith('.css')) return text
  return [...text.matchAll(/<style[^>]*>([\s\S]*?)<\/style>/g)].map((m) => m[1]).join('\n')
}

/** Every rule whose selector uses `:hover` and has no `@media (hover: hover)` around it. */
function unguardedHovers(source: string): string[] {
  const text = source.replace(/\/\*[\s\S]*?\*\//g, '')
  const stack: string[] = []
  const out: string[] = []
  let header = ''
  for (const ch of text) {
    if (ch === '{') {
      const h = header.trim()
      if (/:hover\b/.test(h) && !stack.some((s) => /@media[^{]*\(hover:\s*hover\)/.test(s))) out.push(h)
      stack.push(h)
      header = ''
    } else if (ch === '}') {
      stack.pop()
      header = ''
    } else if (ch === ';') {
      header = ''
    } else {
      header += ch
    }
  }
  return out
}

describe('hover only where a pointer can hover (2026-10-14)', () => {
  it('Tailwind emits `hover:` only for a hovering pointer', () => {
    expect(readFileSync(join(ROOT, 'tailwind.config.js'), 'utf8')).toMatch(/hoverOnlyWhenSupported:\s*true/)
  })

  it('no hand-written :hover rule applies on a touch screen', () => {
    const offenders = styleFiles(SRC).flatMap((file) =>
      unguardedHovers(css(file)).map((sel) => `${relative(SRC, file)}  ${sel.slice(0, 100)}`),
    )
    expect(
      offenders,
      'Put these inside @media (hover: hover) and (pointer: fine) { … }: on a phone a :hover sticks ' +
        'to the last thing touched.',
    ).toEqual([])
  })

  it('the check finds an unguarded hover and passes a guarded one', () => {
    expect(unguardedHovers('.a:hover { color: red; }')).toEqual(['.a:hover'])
    expect(unguardedHovers('@media (hover: hover) and (pointer: fine) { .a:hover { color: red; } }')).toEqual([])
    expect(unguardedHovers('@layer components { .a:hover { color: red; } }')).toEqual(['.a:hover'])
  })
})
