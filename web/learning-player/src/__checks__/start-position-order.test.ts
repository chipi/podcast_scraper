import { existsSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

/**
 * Read a project file regardless of where vitest was invoked from.
 *
 * `process.cwd()` is the app directory for `npm test` but the REPO ROOT for
 * `vitest --root web/learning-player`, which is how it runs from a script at the top level. A
 * bare `resolve(process.cwd(), 'src/…')` therefore throws ENOENT in exactly the invocation a
 * CI wrapper is most likely to use — the check does not report a violation, it fails to run.
 */
function projectFile(rel: string): string {
  for (const base of ['.', 'web/learning-player']) {
    const candidate = resolve(process.cwd(), base, rel)
    if (existsSync(candidate)) return candidate
  }
  throw new Error(`cannot locate ${rel} from ${process.cwd()}`)
}

/**
 * The start position must be settled before the element is loaded (#1905/#1909 follow-up).
 *
 * ## The invariant
 *
 * `applyStartPosition` runs off a watcher on `[player.currentSlug, duration, startReady]`, and the
 * duration appears once the audio element has metadata — which loading it is what causes. So if
 * `player.load()` runs while `resumeSeconds` is still 0, the watcher can apply a position nobody
 * chose: a downloaded episode starts from the beginning instead of where the listener left it.
 *
 * For most of this file's life that could not happen, because `player.load()` and the line
 * computing `resumeSeconds` sat in one synchronous block with no await between them. Nothing named
 * that, and the disk-first path put an `await` inside the window.
 *
 * ## Why a SOURCE check
 *
 * `startReady` is the runtime guard, and no test fails when it is removed — in every current path
 * the position is assigned before the flag, so the hazard is unreachable and there is nothing for
 * a behavioural test to observe. That is exactly the situation where a diff can reintroduce a real
 * bug invisibly: move two statements, break nothing, and the guard silently becomes load-bearing
 * again. This asserts the ORDER, which is the thing that must not drift.
 *
 * Deliberately narrow: it checks the disk-first path in `PlayerView.load()`, the one place an
 * await was introduced into the window. It is not a general theory about player ordering.
 */

const source = readFileSync(
  projectFile('src/views/PlayerView.vue'),
  'utf8',
).replace(/\/\/[^\n]*/g, '') // strip comments — this file's own explanation quotes both calls

describe('the start position is settled before the element is loaded', () => {
  /** The disk-first branch: from its `if` to the end of that block. */
  const fastPath = (() => {
    const start = source.indexOf('if (diskSrc && diskDetail) {')
    expect(start, 'the disk-first fast path is gone — this check has nothing to guard').toBeGreaterThan(-1)
    return source.slice(start, start + 1200)
  })()

  it('assigns resumeSeconds before calling player.load()', () => {
    const assign = fastPath.indexOf('resumeSeconds =')
    const load = fastPath.indexOf('player.load(')
    expect(assign, 'the fast path no longer sets a resume position').toBeGreaterThan(-1)
    expect(load, 'the fast path no longer arms playback').toBeGreaterThan(-1)
    expect(
      assign < load,
      'player.load() runs before resumeSeconds is settled: the element can report a duration, ' +
        'and the start-position watcher will apply a position nobody chose. Move the assignment up.',
    ).toBe(true)
  })

  it('opens the start gate before loading, so the watcher never sees a half-set position', () => {
    const ready = fastPath.indexOf('startReady.value = true')
    const load = fastPath.indexOf('player.load(')
    expect(ready, 'the fast path no longer opens the start gate').toBeGreaterThan(-1)
    expect(ready < load, 'the gate opens after the element is loaded').toBe(true)
  })

  it('still HAS a gate for the paths this check does not cover', () => {
    // The critical path computes `resumeSeconds` after its own `player.load()`, in one synchronous
    // block. That holds today by the absence of an await — the gate is what survives someone
    // adding one.
    expect(source, 'the startReady gate is gone').toContain('startReady')
    expect(source, 'the watcher no longer depends on the gate').toMatch(
      /duration\.value,\s*startReady\.value/,
    )
  })
})
