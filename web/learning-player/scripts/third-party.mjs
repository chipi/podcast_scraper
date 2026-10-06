#!/usr/bin/env node
/**
 * Third-party software list for Settings › About & legal › Third-party software (2026-10-05).
 *
 * Regenerated on EVERY build (`npm run build` runs this first), so the page cannot go stale when a
 * dependency is added, bumped or dropped. Lists what SHIPS in the app — what the licences' notice
 * obligations attach to — and nothing that only runs on the build machine:
 *
 *  - npm: the resolved PRODUCTION tree, direct and transitive (a transitive package ships as much as
 *    a direct one), MINUS `BUILD_ONLY` roots. `@capacitor/cli` is the command-line tool that syncs
 *    the web build into the native projects; it never ships, and it alone pulled in more than half
 *    the list. It is a devDependency since 2026-10-06 (Snyk counted its tree as shipped while it sat
 *    in `dependencies`), so the production walk no longer reaches it; `BUILD_ONLY` stays as the
 *    guard should it ever move back. A package it shares with something that DOES ship is kept,
 *    because the walk only skips the CLI's own subtree.
 *  - native: `scripts/third-party-native.json`, the committed iOS pod and Android Gradle snapshot
 *    `scripts/third-party-native.mjs` writes on the Mac (see there for why it is a snapshot).
 *  - the self-hosted Google Sans font (SIL OFL 1.1).
 *
 * Each entry carries its licence text and, where the package has one, its NOTICE file — Apache-2.0
 * requires the NOTICE to travel with the software.
 *
 * Writes `public/third-party.json` (git-ignored, generated). The page fetches it when opened.
 * No dependencies: `npm ls` and the packages' own files are the source.
 */
import { execSync } from 'node:child_process'
import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const out = path.join(root, 'public', 'third-party.json')
const nativeSnapshot = path.join(root, 'scripts', 'third-party-native.json')
const MAX_TEXT_CHARS = 20_000
/** Production dependencies that only run on the build machine. Their subtrees are not walked. */
const BUILD_ONLY = new Set(['@capacitor/cli'])

function productionTree() {
  let stdout
  try {
    stdout = execSync('npm ls --omit=dev --all --json --long', {
      cwd: root,
      encoding: 'utf8',
      maxBuffer: 64 * 1024 * 1024,
      stdio: ['ignore', 'pipe', 'ignore'],
    })
  } catch (err) {
    // npm ls exits non-zero on peer-dependency warnings while still printing the full tree.
    stdout = err.stdout ?? '{}'
  }
  return JSON.parse(stdout || '{}')
}

/**
 * Install directories of everything that ships, skipping BUILD_ONLY subtrees.
 *
 * npm prints a package shared by several parents IN FULL under one of them and as a childless
 * "deduped" stub under the rest. So a package is recorded from any occurrence, but its children are
 * walked from whichever occurrence carries them — the first version marked a package visited at
 * its stub and then skipped the full copy, which silently dropped 22 shipping packages, Vue's own
 * runtime among them. Nodes without a path are optional peers that are not installed.
 */
function shippedDirs(tree) {
  const dirs = new Set()
  const walked = new Set()
  const walk = (deps) => {
    for (const [name, node] of Object.entries(deps ?? {})) {
      if (BUILD_ONLY.has(name) || !node?.path) continue
      dirs.add(node.path)
      const kids = node.dependencies ?? {}
      if (walked.has(node.path) || Object.keys(kids).length === 0) continue
      walked.add(node.path)
      walk(kids)
    }
  }
  walk(tree.dependencies)
  return [...dirs]
}

function licenseOf(pkg) {
  if (typeof pkg.license === 'string') return pkg.license
  if (pkg.license && typeof pkg.license.type === 'string') return pkg.license.type
  if (Array.isArray(pkg.licenses)) return pkg.licenses.map((l) => l.type ?? l).join(' OR ')
  return 'UNKNOWN'
}

function linkOf(pkg) {
  const repo = typeof pkg.repository === 'string' ? pkg.repository : pkg.repository?.url
  const raw = pkg.homepage || repo || `https://www.npmjs.com/package/${pkg.name}`
  return String(raw)
    .replace(/^git\+/, '')
    .replace(/^git:\/\//, 'https://')
    .replace(/\.git$/, '')
    .replace(/^github:/, 'https://github.com/')
}

function fileText(dir, pattern) {
  const file = readdirSync(dir).find((f) => pattern.test(f))
  if (!file) return null
  const text = readFileSync(path.join(dir, file), 'utf8').trim()
  return text.length > MAX_TEXT_CHARS ? `${text.slice(0, MAX_TEXT_CHARS)}\n…` : text
}

const seen = new Map()
for (const dir of shippedDirs(productionTree())) {
  const manifest = path.join(dir, 'package.json')
  if (!existsSync(manifest)) continue
  const pkg = JSON.parse(readFileSync(manifest, 'utf8'))
  if (!pkg.name || pkg.private) continue
  const key = `${pkg.name}@${pkg.version}`
  if (seen.has(key)) continue
  seen.set(key, {
    name: pkg.name,
    version: pkg.version,
    license: licenseOf(pkg),
    url: linkOf(pkg),
    text: fileText(dir, /^(licen[cs]e|copying)(\.|$|-)/i),
    notice: fileText(dir, /^notice(\.|$)/i),
    platform: 'web',
  })
}

const native = existsSync(nativeSnapshot) ? JSON.parse(readFileSync(nativeSnapshot, 'utf8')).packages ?? [] : []

const fontLicense = path.join(root, 'public', 'fonts', 'GOOGLE-SANS-OFL.txt')
const assets = [
  {
    name: 'Google Sans (font)',
    version: '',
    license: 'OFL-1.1',
    url: 'https://fonts.google.com/specimen/Google+Sans',
    text: existsSync(fontLicense) ? readFileSync(fontLicense, 'utf8').trim() : null,
    notice: null,
    platform: 'web',
  },
]

const packages = [...seen.values(), ...native, ...assets].sort((a, b) =>
  a.name.localeCompare(b.name, 'en', { sensitivity: 'base' }),
)
writeFileSync(out, `${JSON.stringify({ generated: new Date().toISOString(), packages })}\n`)
const by = (p) => packages.filter((e) => e.platform === p).length
console.log(
  `third-party: ${packages.length} entries (${by('web')} web, ${by('ios')} iOS, ${by('android')} Android; ` +
    `${packages.filter((e) => e.notice).length} with a NOTICE) → ${path.relative(root, out)}`,
)
