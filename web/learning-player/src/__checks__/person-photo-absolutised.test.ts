import { describe, expect, it } from "vitest"
import apiSrc from "../services/api.ts?raw"
import typesSrc from "../services/types.ts?raw"

/**
 * Every endpoint that can carry a PERSON PHOTO must absolutise it.
 *
 * This defect has now shipped SIX times — episode artwork, audio, the profile avatar, the key-voices
 * rail, topic perspectives, and the trending people lists (2026-09-30: Discover → Trends → People
 * showed initials for every person, on device, while the same person's card showed the photo).
 * Every instance was silent.
 *
 * The mechanism is always the same. The server returns the photo route RELATIVE
 * (`/api/app/persons/<id>/photo`). On the web that is correct: the app and the API share an origin.
 * Inside the Capacitor WebView the document origin is `capacitor://localhost`, so the same string
 * resolves THERE, 404s, and `ProfileAvatar` falls back to initials. `fetch` is never affected —
 * `apiFetch` prefixes an absolute base itself — but `<img src>` has nothing doing that for it.
 *
 * Why it keeps happening: nothing about the failure looks like a failure. A 404 on an `<img>` is
 * not an error any human sees, and initials are a legitimate state ("we have no photo for this
 * person"), so the surface reads as working-but-sparse. It cannot be caught by the browser tier
 * either, because on the web the origins match and the bug does not exist.
 *
 * WHY THIS NO LONGER KEEPS A LIST OF FETCHERS. The previous version checked a hand-kept list of
 * five function names. `/trending` began returning person photos in #2034 and nobody added
 * `getTrending` to the list, so the guard that existed for exactly this bug passed while the bug
 * shipped a sixth time. A list of what to check is the thing that goes stale.
 *
 * So the fetchers are DERIVED: a fetcher is people-carrying when the type it asks `getJSON` for
 * reaches — directly or through nested types in types.ts — a type that holds a person photo. What
 * is still declared by hand is which TYPES hold a person photo, and that declaration is itself
 * checked: every type in types.ts with an `image_url` field must be classified as a person photo or
 * as artwork, so a new photo-carrying type fails here until someone decides which it is.
 */

/** Types whose `image_url` is a PERSON's photo (served relative, from our API). */
const PERSON_PHOTO_TYPES = ["Entity", "KeyVoice", "PersonWeb", "TopicPerspective", "TrendingEntity"]
/** Types whose `image_url` is episode/show ARTWORK — a different field family, covered elsewhere. */
const ARTWORK_TYPES = ["EpisodeDetail", "EpisodeSummary", "Podcast", "YourWeekItem"]

const ABSOLUTISERS = /resolveMediaUrl|withAbsoluteEntityImages|withAbsoluteCovers/

/** `{ typeName: body }` for every exported interface / type alias in types.ts. */
function declaredTypes(): Map<string, string> {
  const out = new Map<string, string>()
  const re = /^export (?:interface|type) (\w+)/gm
  const starts = [...typesSrc.matchAll(re)]
  starts.forEach((m, i) => {
    const end = i + 1 < starts.length ? starts[i + 1].index : typesSrc.length
    out.set(m[1], typesSrc.slice(m.index, end))
  })
  return out
}

/** Every type name reachable from `roots` through field references in types.ts. */
function reachable(roots: string[], types: Map<string, string>): Set<string> {
  const seen = new Set<string>()
  const stack = [...roots]
  while (stack.length) {
    const name = stack.pop()!
    if (seen.has(name) || !types.has(name)) continue
    seen.add(name)
    // Comments stripped: a doc comment that merely MENTIONS a type ("unlike Entity, …") is not a
    // reference, and counting it made /highlights look people-carrying.
    const body = types
      .get(name)!
      .replace(/^export (?:interface|type) \w+/, "")
      .replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, "")
    for (const [, ref] of body.matchAll(/\b([A-Z]\w+)\b/g)) if (types.has(ref)) stack.push(ref)
  }
  return seen
}

/** `[name, body]` for every exported fetcher in api.ts, split at the next top-level export. */
function fetchers(): [string, string][] {
  const re = /^export (?:async )?function (\w+)\(/gm
  const starts = [...apiSrc.matchAll(re)]
  return starts.map((m, i) => {
    const next = apiSrc.indexOf("\nexport ", (m.index ?? 0) + 1)
    return [m[1], apiSrc.slice(m.index, next < 0 ? undefined : next)]
  })
}

/** Type names a fetcher asks `getJSON` for (`getJSON<{ items: TrendingEntity[] }>` → TrendingEntity). */
function requestedTypes(body: string): string[] {
  const names: string[] = []
  for (const [, generic] of body.matchAll(/getJSON<([^(]*?)>\s*\(/g)) {
    for (const [, n] of generic.matchAll(/\b([A-Z]\w+)\b/g)) names.push(n)
  }
  return names
}

const types = declaredTypes()

function peopleFetchers(): string[] {
  return fetchers()
    .filter(([, body]) => {
      const hit = reachable(requestedTypes(body), types)
      return PERSON_PHOTO_TYPES.some((t) => hit.has(t))
    })
    .map(([name]) => name)
}

describe("person photos survive the native origin", () => {
  it("every type with an image_url is classified as a person photo or as artwork", () => {
    const withImage = [...types].filter(([, body]) => /\bimage_url\??:/.test(body)).map(([n]) => n)
    const unclassified = withImage.filter(
      (n) => !PERSON_PHOTO_TYPES.includes(n) && !ARTWORK_TYPES.includes(n),
    )
    expect(
      unclassified,
      `types.ts gained an image_url on ${unclassified.join(", ")}. Decide whether it is a PERSON ` +
        `photo (add to PERSON_PHOTO_TYPES — every fetcher returning it must then absolutise) or ` +
        `artwork (ARTWORK_TYPES).`,
    ).toEqual([])
    // And the declared lists must name real types, or a rename would silently empty them.
    for (const t of [...PERSON_PHOTO_TYPES, ...ARTWORK_TYPES]) expect(types.has(t), t).toBe(true)
  })

  it("the derivation finds the known people fetchers (not vacuous)", () => {
    const found = peopleFetchers()
    for (const name of ["getPersonCard", "getTopicCard", "getKeyVoices", "getTrending"]) {
      expect(found, `${name} was not derived as people-carrying`).toContain(name)
    }
  })

  it("every people-carrying fetcher absolutises the photo URL", () => {
    const raw = fetchers()
      .filter(([name]) => peopleFetchers().includes(name))
      .filter(([, body]) => !ABSOLUTISERS.test(body))
      .map(([name]) => name)
    expect(
      raw,
      `these fetchers return person data with a RELATIVE image_url. In the Capacitor WebView that ` +
        `resolves against capacitor://localhost, 404s, and ProfileAvatar silently falls back to ` +
        `initials — the surface looks like "no photo" rather than broken. Pass the response ` +
        `through resolveMediaUrl (or withAbsoluteEntityImages).`,
    ).toEqual([])
  })
})
