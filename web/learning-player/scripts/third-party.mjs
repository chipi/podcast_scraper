#!/usr/bin/env node
/**
 * Third-party software list for Settings › About & legal › Third-party software (2026-10-05).
 *
 * Regenerated on EVERY build (`npm run build` runs this first), so the page cannot go stale when a
 * dependency is added, bumped or dropped. It walks npm's own resolved PRODUCTION tree — direct and
 * transitive, because a transitive package ships in the app as much as a direct one does — and
 * records each package's name, version, declared licence, link and licence text. Dev-only tooling
 * (test runners, linters, the bundler) is not distributed and is excluded by `--omit=dev`.
 *
 * Also lists the one non-npm asset the app ships, the self-hosted Google Sans font (SIL OFL 1.1).
 *
 * Writes `public/third-party.json` (git-ignored, generated). The page fetches it when opened, so
 * hundreds of licence texts never weigh on the app bundle.
 *
 * No dependencies: `npm ls` and the packages' own package.json / LICENSE files are the source.
 */
import { execSync } from 'node:child_process'
import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const out = path.join(root, 'public', 'third-party.json')
const MAX_LICENSE_CHARS = 20_000

function productionPackageDirs() {
  let stdout
  try {
    stdout = execSync('npm ls --omit=dev --all --parseable', { cwd: root, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] })
  } catch (err) {
    // npm ls exits non-zero on peer-dependency warnings while still printing the full tree.
    stdout = err.stdout ?? ''
  }
  return stdout
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => l && path.resolve(l) !== root)
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

function licenseText(dir) {
  const file = readdirSync(dir).find((f) => /^(licen[cs]e|copying)(\.|$|-)/i.test(f))
  if (!file) return null
  const text = readFileSync(path.join(dir, file), 'utf8').trim()
  return text.length > MAX_LICENSE_CHARS ? `${text.slice(0, MAX_LICENSE_CHARS)}\n…` : text
}

const seen = new Map()
for (const dir of productionPackageDirs()) {
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
    text: licenseText(dir),
  })
}

const fontLicense = path.join(root, 'public', 'fonts', 'GOOGLE-SANS-OFL.txt')
const assets = [
  {
    name: 'Google Sans (font)',
    version: '',
    license: 'OFL-1.1',
    url: 'https://fonts.google.com/specimen/Google+Sans',
    text: existsSync(fontLicense) ? readFileSync(fontLicense, 'utf8').trim() : null,
  },
]

const packages = [...seen.values(), ...assets].sort((a, b) =>
  a.name.localeCompare(b.name, 'en', { sensitivity: 'base' }),
)
writeFileSync(out, `${JSON.stringify({ generated: new Date().toISOString(), packages })}\n`)
console.log(`third-party: ${packages.length} entries → ${path.relative(root, out)}`)
