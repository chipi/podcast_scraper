import { createHash } from "node:crypto"
import { existsSync, readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

/**
 * The native third-party list cannot be resolved at build time — the production web build has no
 * Xcode or Android SDK — so `scripts/third-party-native.json` is a committed snapshot, written on the
 * Mac by `make third-party-native`. A snapshot can go stale silently; this is what stops that.
 *
 * It records a hash of every file the iOS pods and the Android release classpath are resolved from.
 * Change one of them (a pod bump, a new Gradle dependency, a Capacitor plugin added) without
 * regenerating, and the Third-party software page would list libraries the app no longer ships — or
 * miss ones it now does, which is the case the licences care about.
 */
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..")
const snapshot = JSON.parse(readFileSync(path.join(root, "scripts", "third-party-native.json"), "utf8")) as {
  inputs: Record<string, string | null>
  packages: Array<{ name: string; license: string; platform: string }>
}

describe("native third-party snapshot", () => {
  it("was generated from the native dependency files as they are now", () => {
    const stale = Object.entries(snapshot.inputs).filter(([file, hash]) => {
      const p = path.join(root, file)
      const now = existsSync(p) ? createHash("sha256").update(readFileSync(p)).digest("hex") : null
      return now !== hash
    })
    expect(
      stale.map(([f]) => f),
      "native dependencies changed — run `make third-party-native` and commit scripts/third-party-native.json",
    ).toEqual([])
  })

  it("covers both platforms, and every library has a licence", () => {
    expect(snapshot.packages.some((p) => p.platform === "ios")).toBe(true)
    expect(snapshot.packages.some((p) => p.platform === "android")).toBe(true)
    expect(snapshot.packages.filter((p) => p.license === "UNKNOWN").map((p) => p.name)).toEqual([])
  })
})
