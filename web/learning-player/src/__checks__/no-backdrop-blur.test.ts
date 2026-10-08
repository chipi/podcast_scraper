import { describe, expect, it } from 'vitest'
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join, relative, resolve } from 'node:path'

/**
 * NO BACKDROP BLUR anywhere in the app (2026-10-08).
 *
 * A beta tester's Pixel 8 (build 1.0.2) drew Episode notes blank: after toggling Key points, and
 * while scrolling the "More like this" rail, everything went dark except the blurred action buttons.
 * Reproduced on an 8 GB Pixel 8 emulator against prod: on that screen 38 elements carried
 * `backdrop-filter` (3 per related tile, plus the player's zone D underneath the sheet), the page had
 * 74 composited layers including ~20 full-screen ones that never painted, and Chromium's tile memory
 * sat at 200 MB. With every blur switched off — nothing else changed — it was 20 layers, 95 MB, and
 * the sheet drew.
 *
 * Each `backdrop-filter` is its own render surface that reads back everything beneath it, so the
 * cost grows with every tile in a rail and every sheet stacked over the player. iOS showed nothing,
 * which is why this reached testers: it only fails on the WebView we do not develop on. The plates
 * behind the buttons (`bg-black/55`, `bg-canvas/95`) carry the contrast; the blur was decoration.
 */

const SRC = resolve(process.cwd(), 'src')

function sourceFiles(dir: string, acc: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const full = join(dir, name)
    if (statSync(full).isDirectory()) {
      if (name === 'node_modules' || name === '__checks__') continue
      sourceFiles(full, acc)
      continue
    }
    if (!/\.(ts|vue|css)$/.test(name) || /\.test\.ts$/.test(name)) continue
    acc.push(full)
  }
  return acc
}

// A class or a declaration, not prose: `backdrop-blur-sm`, `backdrop-blur]`, `backdrop-filter:`.
const BLUR = /backdrop-blur(?:-[a-z0-9]+)?["'\s\]]|backdrop-filter\s*:/

/** Blank out comments but keep their newlines, so a reported line number is the real one. */
function stripComments(text: string): string {
  const blank = (m: string): string => m.replace(/[^\n]/g, '')
  return text
    .replace(/<!--[\s\S]*?-->/g, blank)
    .replace(/\/\*[\s\S]*?\*\//g, blank)
    .replace(/^[ \t]*\/\/.*$/gm, '')
}

describe('no backdrop blur (Android WebView blank regions, 2026-10-08)', () => {
  it('no source file applies a backdrop blur', () => {
    const offenders = sourceFiles(SRC).flatMap((file) =>
      stripComments(readFileSync(file, 'utf8'))
        .split('\n')
        .map((line, i) => ({ line, i }))
        .filter(({ line }) => BLUR.test(line))
        .map(({ line, i }) => `${relative(SRC, file)}:${i + 1}  ${line.trim().slice(0, 100)}`),
    )
    expect(
      offenders,
      'backdrop-filter made Android WebView stop drawing whole regions. Use an opaque or ' +
        'translucent plate (`bg-black/55`, `bg-canvas/95`) instead.',
    ).toEqual([])
  })

  it('the rule catches the forms it exists for', () => {
    for (const bad of ['class="a backdrop-blur-sm"', "'h-7 backdrop-blur'", '[&>button]:backdrop-blur]', 'backdrop-filter: blur(4px);'])
      expect(BLUR.test(bad), bad).toBe(true)
    // The notes that explain WHY (comments) are not offenders.
    expect(BLUR.test(stripComments('<!-- this was `backdrop-blur-md` -->\n/* backdrop-filter: blur(4px) */'))).toBe(false)
  })
})
