#!/usr/bin/env node
/**
 * Native third-party libraries — the iOS and Android code the app binaries ship that does NOT come
 * from an npm package (2026-10-05). Writes `scripts/third-party-native.json`, which IS committed:
 * `scripts/third-party.mjs` merges it into the Third-party software page on every web build.
 *
 * Why a committed snapshot rather than resolving at build time: the production web build runs in a
 * Linux container with no Xcode or Android SDK, so it cannot ask CocoaPods or Gradle anything. This
 * runs on the Mac (`make third-party-native`) where both exist. To keep the snapshot from going
 * stale silently, it records a hash of every file the native dependency set is resolved from, and
 * `src/__checks__/third-party-native.test.ts` fails when any of them changes without a regenerate.
 *
 * iOS: the pods in `ios/App/Podfile.lock`, minus those built from `node_modules` (Capacitor and its
 * plugins — already listed as their npm packages). Licence text from the installed pod's LICENSE.
 *
 * Android: Gradle's `releaseRuntimeClasspath` — exactly what the release APK links — minus local
 * `project(...)` modules (Capacitor and plugins again). Licence name and URL from each artifact's POM
 * in the Gradle cache, and the licence text / NOTICE from inside the .jar/.aar when the artifact
 * carries one; otherwise the page links the licence, as Google's own oss-licenses plugin does.
 */
import { createHash } from 'node:crypto'
import { execSync } from 'node:child_process'
import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { homedir } from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const out = path.join(root, 'scripts', 'third-party-native.json')

/** Every file the native dependency set is resolved from. A change to any of them means regenerate. */
export const NATIVE_INPUTS = [
  'ios/App/Podfile.lock',
  'android/build.gradle',
  'android/variables.gradle',
  'android/app/build.gradle',
  'android/app/capacitor.build.gradle',
  'android/capacitor.settings.gradle',
]

export function hashInputs(base = root) {
  const h = {}
  for (const f of NATIVE_INPUTS) {
    const p = path.join(base, f)
    h[f] = existsSync(p) ? createHash('sha256').update(readFileSync(p)).digest('hex') : null
  }
  return h
}

function licenseTextIn(dir) {
  if (!existsSync(dir)) return null
  const file = readdirSync(dir).find((f) => /^(licen[cs]e|copying)(\.|$|-)/i.test(f))
  return file ? readFileSync(path.join(dir, file), 'utf8').trim() : null
}

function guessSpdx(text) {
  if (!text) return 'UNKNOWN'
  if (/MIT License|Permission is hereby granted, free of charge/i.test(text)) return 'MIT'
  if (/Apache License,?\s+Version 2\.0/i.test(text)) return 'Apache-2.0'
  if (/Redistribution and use in source and binary forms/i.test(text)) return 'BSD'
  return 'See licence text'
}

/** One name per licence: POMs spell Apache five different ways ("The Apache Software License, …"). */
function normalise(name) {
  if (/apache/i.test(name) && /2/.test(name)) return 'Apache-2.0'
  if (/^mit( license)?$/i.test(name.trim())) return 'MIT'
  if (/^license$/i.test(name.trim())) return 'See licence'
  return name
}

function iosPods() {
  const lock = readFileSync(path.join(root, 'ios/App/Podfile.lock'), 'utf8')
  const fromNodeModules = new Set(
    [...lock.matchAll(/^\s+- "?([A-Za-z0-9_-]+) \(from `[^`]*node_modules/gm)].map((m) => m[1]),
  )
  const podsSection = lock.split('\nDEPENDENCIES:')[0]
  const pods = [...podsSection.matchAll(/^ {2}- "?([A-Za-z0-9_-]+) \(([^)]+)\)/gm)]
  return pods
    .filter(([, name]) => !fromNodeModules.has(name))
    .map(([, name, version]) => {
      const text = licenseTextIn(path.join(root, 'ios/App/Pods', name))
      return {
        name,
        version,
        license: guessSpdx(text),
        url: `https://cocoapods.org/pods/${name}`,
        text,
        notice: null,
        platform: 'ios',
      }
    })
}

function androidArtifacts() {
  const env = {
    ...process.env,
    ANDROID_HOME: process.env.ANDROID_HOME || path.join(homedir(), 'Library/Android/sdk'),
  }
  const report = execSync(
    './gradlew --no-daemon -q :app:dependencies --configuration releaseRuntimeClasspath',
    { cwd: path.join(root, 'android'), env, encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'] },
  )
  const coords = new Map()
  for (const line of report.split('\n')) {
    const m = line.match(/--- ([A-Za-z0-9_.-]+):([A-Za-z0-9_.-]+):([^\s]+)(?: -> ([^\s]+))?/)
    if (!m) continue // `project :capacitor-…` lines and the tree's own decoration
    const [, group, artifact, requested, resolved] = m
    const version = resolved || requested
    coords.set(`${group}:${artifact}`, { group, artifact, version })
  }
  const cache = path.join(homedir(), '.gradle/caches/modules-2/files-2.1')
  const findPom = (g, a, v) => {
    const dir = path.join(cache, g, a, v)
    if (!existsSync(dir)) return null
    for (const h of readdirSync(dir)) {
      const p = path.join(dir, h, `${a}-${v}.pom`)
      if (existsSync(p)) return readFileSync(p, 'utf8')
    }
    return null
  }
  const licenceFrom = (pom, depth = 0) => {
    if (!pom || depth > 4) return null
    const name = pom.match(/<license>\s*<name>([^<]+)<\/name>/)?.[1]?.trim()
    const url = pom.match(/<license>[\s\S]*?<url>([^<]+)<\/url>/)?.[1]?.trim()
    if (name) return { name, url: url ?? null }
    // Licences are often declared once on a parent POM.
    const parent = pom.match(/<parent>\s*<groupId>([^<]+)<\/groupId>\s*<artifactId>([^<]+)<\/artifactId>\s*<version>([^<]+)<\/version>/)
    return parent ? licenceFrom(findPom(parent[1], parent[2], parent[3]), depth + 1) : null
  }
  // Some artifacts carry their licence (and could carry a NOTICE) inside the .jar/.aar; read them
  // straight out of the archive so the page shows the text, not only a link.
  const fromArchive = (g, a, v, pattern) => {
    const dir = path.join(cache, g, a, v)
    if (!existsSync(dir)) return null
    for (const h of readdirSync(dir)) {
      for (const f of readdirSync(path.join(dir, h))) {
        if (!/\.(aar|jar)$/.test(f) || /sources|javadoc/.test(f)) continue
        const archive = path.join(dir, h, f)
        const entry = execSync(`unzip -Z1 "${archive}"`, { encoding: 'utf8' })
          .split('\n')
          .find((l) => pattern.test(l))
        if (entry) return execSync(`unzip -p "${archive}" "${entry}"`, { encoding: 'utf8' }).trim()
      }
    }
    return null
  }
  return [...coords.values()].map(({ group, artifact, version }) => {
    const pom = findPom(group, artifact, version)
    const lic = licenceFrom(pom)
    const site = pom?.match(/<project[\s\S]*?<url>([^<]+)<\/url>/)?.[1]?.trim()
    return {
      name: `${group}:${artifact}`,
      version,
      license: lic?.name ? normalise(lic.name) : 'UNKNOWN',
      url: lic?.url ?? site ?? `https://mvnrepository.com/artifact/${group}/${artifact}`,
      text: fromArchive(group, artifact, version, /(^|\/)LICEN[CS]E[^/]*$/i),
      notice: fromArchive(group, artifact, version, /(^|\/)NOTICE[^/]*$/i),
      platform: 'android',
    }
  })
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const packages = [...iosPods(), ...androidArtifacts()].sort((a, b) => a.name.localeCompare(b.name))
  const unknown = packages.filter((p) => p.license === 'UNKNOWN').map((p) => p.name)
  writeFileSync(
    out,
    `${JSON.stringify({ generated: new Date().toISOString(), inputs: hashInputs(), packages }, null, 2)}\n`,
  )
  console.log(
    `third-party-native: ${packages.length} entries (${packages.filter((p) => p.platform === 'ios').length} iOS, ` +
      `${packages.filter((p) => p.platform === 'android').length} Android) → ${path.relative(root, out)}` +
      (unknown.length ? `\n  licence not found for: ${unknown.join(', ')}` : ''),
  )
}
