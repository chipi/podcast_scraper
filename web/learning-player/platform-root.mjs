/**
 * Where the public platform checkout is, and where this app's own docs are.
 *
 * The app runs from two layouts (ADR-158): inside the public repo at `web/learning-player`, and
 * inside the private player repo mounted at `apps/player/web`. The platform root is two levels up
 * in the first and three in the second, so it is found by walking up to the directory that holds
 * the platform package rather than by counting `..`. `PLATFORM_ROOT` overrides the search for
 * layouts where the platform is checked out somewhere else (private CI).
 */
import { existsSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

export const appRoot = dirname(fileURLToPath(import.meta.url))

const MARKER = join('src', 'podcast_scraper', '__init__.py')

export function platformRoot() {
  const override = process.env.PLATFORM_ROOT
  if (override) {
    const root = resolve(override)
    if (!existsSync(join(root, MARKER))) {
      throw new Error(`PLATFORM_ROOT=${override} does not contain ${MARKER}`)
    }
    return root
  }
  let dir = appRoot
  while (dir !== dirname(dir)) {
    dir = dirname(dir)
    if (existsSync(join(dir, MARKER))) return dir
  }
  throw new Error(`no platform checkout (${MARKER}) above ${appRoot}; set PLATFORM_ROOT`)
}

/**
 * The directory holding this app's docs (UXS, PRD, RFC). In the private repo they sit beside the
 * app (`<repo>/docs`); in the public repo they are still in the platform's `docs/`.
 */
export function appDocsRoot() {
  const own = resolve(appRoot, '..', 'docs')
  return existsSync(own) ? own : join(platformRoot(), 'docs')
}
