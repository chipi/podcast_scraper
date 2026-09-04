import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

/**
 * Every path a README points at still exists.
 *
 * These READMEs exist so a fresh agent or contributor can bootstrap from a directory without
 * reading the whole app. That only works while their pointers are true, and a pointer rots
 * silently: nothing fails when a file is renamed out from under a link, and the reader — human or
 * agent — follows it, finds nothing, and concludes the thing does not exist.
 *
 * That failure is not hypothetical here. `build_app_validation_corpus.py` carried a comment
 * asserting "the player never decodes it in e2e"; it had become false, it read as justification,
 * and it corroborated a wrong premise that cost most of a day. A stale doc is worse than a missing
 * one, because it is trusted.
 *
 * So the deal is: these files may point outward freely, and the build breaks the moment a pointer
 * goes bad. What this CANNOT check is whether the prose is still true — keep the prose about
 * contracts and reasons (slow to change) rather than restating code (fast to change).
 */

const APP_ROOT = resolve(__dirname, '..', '..')

/** Markdown link targets, minus anchors, mail and absolute URLs. */
const LINK = /\[[^\]]*\]\(([^)\s]+)\)/g

/**
 * Dependency trees whose READMEs we do not own.
 *
 * `node_modules` and `dist` were always here. `vendor` and `Pods` had to be added the first time
 * anyone ran `make ios-fastlane-install`, which vendors ~90 Ruby gems into `ios/vendor/bundle`:
 * this walk happily descended into them and produced 14 failures about broken links in
 * third-party gem READMEs (`colored2` pointing at a screenshot it does not ship, and similar).
 *
 * Those are not our documents and not our links. A guard that fails on someone else's README
 * teaches people to ignore it, which costs more than the guard is worth.
 */
const VENDORED = new Set(['node_modules', 'dist', 'vendor', 'Pods'])

function markdownFiles(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (VENDORED.has(entry.name) || entry.name.startsWith('.')) {
      continue
    }
    const p = join(dir, entry.name)
    if (entry.isDirectory()) markdownFiles(p, out)
    else if (entry.name === 'README.md' || entry.name.endsWith('_SURFACE_MAP.md')) out.push(p)
  }
  return out
}

describe('README pointers resolve', () => {
  const files = markdownFiles(APP_ROOT)

  it('finds the docs it is meant to guard', () => {
    // A guard that silently matches nothing is worse than no guard. If the app's READMEs move,
    // this fails rather than passing vacuously.
    expect(files.length, 'expected several READMEs / surface maps under the app').toBeGreaterThan(3)
  })

  it.each(files.map((f) => [f.slice(APP_ROOT.length + 1), f] as const))(
    '%s points only at things that exist',
    (_label, file) => {
      const text = readFileSync(file, 'utf8')
      const broken: string[] = []

      for (const match of text.matchAll(LINK)) {
        const target = match[1]
        if (/^(https?:|mailto:|#)/.test(target)) continue
        const path = target.split('#')[0]
        if (!path) continue
        const resolved = resolve(dirname(file), path)
        if (!existsSync(resolved)) broken.push(target)
      }

      expect(
        broken,
        `${file.slice(APP_ROOT.length + 1)} links to paths that no longer exist. Fix the link, or ` +
          `delete it — a pointer that goes nowhere sends the reader to the conclusion that the ` +
          `thing does not exist, which is exactly what these files are here to prevent.`,
      ).toEqual([])
    },
  )

  it('directory READMEs do not link to directories that vanished', () => {
    // Cheap extra: a link to a directory resolves, but a link to a directory that is now a file
    // (or vice versa) is a rename nobody noticed.
    const problems: string[] = []
    for (const file of files) {
      for (const match of readFileSync(file, 'utf8').matchAll(LINK)) {
        const target = match[1]
        if (/^(https?:|mailto:|#)/.test(target)) continue
        const path = target.split('#')[0]
        if (!path || !path.endsWith('/')) continue
        const resolved = resolve(dirname(file), path)
        if (existsSync(resolved) && !statSync(resolved).isDirectory()) {
          problems.push(`${file.slice(APP_ROOT.length + 1)} → ${target}`)
        }
      }
    }
    expect(problems).toEqual([])
  })
})
