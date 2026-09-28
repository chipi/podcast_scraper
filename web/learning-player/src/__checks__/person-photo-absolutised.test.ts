import { describe, expect, it } from "vitest"
import apiSrc from "../services/api.ts?raw"

/**
 * Every endpoint that can carry a PERSON PHOTO must absolutise it.
 *
 * This defect has now shipped FIVE times — episode artwork, audio, the profile avatar, the
 * key-voices rail, and topic perspectives — and every instance was silent.
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
 * So the check is structural: any fetcher whose response carries `image_url` on a PERSON must pass
 * it through `resolveMediaUrl` (directly, or via `withAbsoluteEntityImages` / `withAbsoluteCovers`).
 * Asserted on the source, because the behaviour is only observable against a non-matching document
 * origin — which jsdom does not model.
 */

/** Fetchers whose payloads carry a person photo. Add to this when a new people endpoint lands. */
const PEOPLE_FETCHERS = [
  "getPersonCard",
  "getTopicCard",
  "getOrgCard",
  "getKeyVoices",
  "getTopicPerspectives",
]

const ABSOLUTISERS = /resolveMediaUrl|withAbsoluteEntityImages|withAbsoluteCovers/

/** The body of `export [async] function <name>(` up to the next top-level `export`. */
function bodyOf(name: string): string {
  const start = apiSrc.indexOf(`export async function ${name}(`) >= 0
    ? apiSrc.indexOf(`export async function ${name}(`)
    : apiSrc.indexOf(`export function ${name}(`)
  if (start < 0) return ""
  const next = apiSrc.indexOf("\nexport ", start + 1)
  return apiSrc.slice(start, next < 0 ? undefined : next)
}

describe("person photos survive the native origin", () => {
  it.each(PEOPLE_FETCHERS)("%s absolutises the photo URL", (name) => {
    const body = bodyOf(name)
    expect(body, `${name} not found in api.ts — was it renamed? Update PEOPLE_FETCHERS.`).not.toBe(
      ""
    )
    expect(
      body,
      `${name} returns person data with a RELATIVE image_url. In the Capacitor WebView that ` +
        `resolves against capacitor://localhost, 404s, and ProfileAvatar silently falls back to ` +
        `initials — the surface looks like "no photo" rather than broken. Pass the response ` +
        `through resolveMediaUrl (or withAbsoluteEntityImages).`
    ).toMatch(ABSOLUTISERS)
  })

  it("the guard is not vacuous — bodyOf actually isolates a function", () => {
    // If `bodyOf` returned the whole file, every case above would pass on someone else's
    // absolutiser. Pin that it returns one function and not its neighbours.
    const body = bodyOf("getKeyVoices")
    expect(body).toContain("key-voices")
    expect(body, "bodyOf leaked into the next export").not.toContain("getTopicPerspectives")
  })
})
