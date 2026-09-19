/**
 * Entity image URLs must come back ABSOLUTE from the API boundary.
 *
 * THE BUG THIS PINS (2026-09-19). The person_web enricher had downloaded 652 photos and the API
 * returned each as a relative `/api/app/persons/<id>/photo`. On the web that is fine. Inside the
 * Capacitor WebView the document origin is `capacitor://localhost`, so the relative path resolved
 * THERE, 404'd, and `ProfileAvatar` fell back to initials — every person card showed a bio and no
 * picture, which read as "the images are broken".
 *
 * It is the SAME defect the avatar hit on 2026-09-16 (`withAbsoluteAvatar`), and the same one
 * behind the artwork/audio device tier. The person photo was the one image in the app still
 * rendered from a raw `:src` without `resolveMediaUrl`.
 *
 * These tests assert the BOUNDARY resolves it, not the component, so a new render site cannot
 * reintroduce it.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const API_BASE = 'https://closelistening.app/api/app'

// `resolveMediaUrl` is mocked rather than driven through a fake base, because mocking
// `resolveApiBase` would NOT work: the real `resolveMediaUrl` calls its own module-scope
// `resolveApiBase`, not the mocked export, so the spread-actual version silently returned the
// input unchanged and these tests passed against broken code until the mock was fixed.
//
// Mocking the collaborator is also the sharper assertion. The invariant is "the API BOUNDARY
// routes every entity image field through the shared resolver" — the bug was that it did not
// call it at all. `resolveMediaUrl`'s own behaviour is covered by tier.mediaUrl.test.ts.
vi.mock('./tier', async () => {
  const actual = await vi.importActual<typeof import('./tier')>('./tier')
  return {
    ...actual,
    resolveMediaUrl: (u: string | null | undefined) =>
      !u ? null : /^[a-z][a-z0-9+.-]*:/i.test(u) ? u : `${API_BASE.replace(/\/api\/app$/, '')}${u}`,
  }
})

function mockFetch(body: unknown): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({ ok: true, status: 200, json: async () => body })),
  )
}

beforeEach(() => {
  vi.resetModules()
})
afterEach(() => {
  vi.unstubAllGlobals()
})

describe('person card photo', () => {
  it('absolutises web.image_url against the API base', async () => {
    const { getPersonCard } = await import('./api')
    mockFetch({
      id: 'person:elon-musk',
      label: 'Elon Musk',
      episode_count: 3,
      episodes: [],
      related_people: [],
      related_topics: [],
      web: { bio: 'b', source: 'wikipedia', image_url: '/api/app/persons/person%3Aelon-musk/photo' },
    })
    const card = await getPersonCard('person:elon-musk')
    expect(card.web?.image_url).toBe(
      'https://closelistening.app/api/app/persons/person%3Aelon-musk/photo',
    )
  })

  it('leaves an absolute url untouched', async () => {
    const { getPersonCard } = await import('./api')
    const abs = 'https://cdn.example.com/p.jpg'
    mockFetch({
      id: 'person:x',
      label: 'X',
      episode_count: 0,
      episodes: [],
      related_people: [],
      related_topics: [],
      web: { bio: 'b', source: 's', image_url: abs },
    })
    expect((await getPersonCard('person:x')).web?.image_url).toBe(abs)
  })

  it('leaves a null photo null — absent must not become a broken src', async () => {
    const { getPersonCard } = await import('./api')
    mockFetch({
      id: 'person:y',
      label: 'Y',
      episode_count: 0,
      episodes: [],
      related_people: [],
      related_topics: [],
      web: { bio: 'b', source: 's', image_url: null },
    })
    expect((await getPersonCard('person:y')).web?.image_url).toBeNull()
  })

  it('absolutises related_people avatars too — the same route backs those chips', async () => {
    const { getPersonCard } = await import('./api')
    mockFetch({
      id: 'person:a',
      label: 'A',
      episode_count: 0,
      episodes: [],
      related_topics: [],
      related_people: [
        { id: 'person:b', label: 'B', image_url: '/api/app/persons/person%3Ab/photo' },
        { id: 'person:c', label: 'C' },
      ],
      web: null,
    })
    const card = await getPersonCard('person:a')
    const people = card.related_people as Array<{ image_url?: string | null }>
    expect(people[0].image_url).toBe('https://closelistening.app/api/app/persons/person%3Ab/photo')
    expect(people[1].image_url).toBeUndefined()
  })
})

describe('org card logo', () => {
  it('absolutises web.logo_url', async () => {
    const { getOrgCard } = await import('./api')
    mockFetch({
      id: 'org:acme',
      label: 'Acme',
      episode_count: 0,
      episodes: [],
      web: { description: 'd', source: 'wikipedia', logo_url: '/api/app/organizations/org%3Aacme/logo' },
    })
    const card = (await getOrgCard('org:acme')) as { web?: { logo_url?: string | null } }
    expect(card.web?.logo_url).toBe(
      'https://closelistening.app/api/app/organizations/org%3Aacme/logo',
    )
  })
})

describe('topic card top voices', () => {
  it('absolutises the people chips a topic card carries', async () => {
    const { getTopicCard } = await import('./api')
    mockFetch({
      id: 'topic:ai',
      label: 'AI',
      episode_count: 0,
      episodes: [],
      related_people: [{ id: 'person:d', label: 'D', image_url: '/api/app/persons/person%3Ad/photo' }],
      related_topics: [],
    })
    const card = (await getTopicCard('topic:ai')) as {
      related_people: Array<{ image_url?: string | null }>
    }
    expect(card.related_people[0].image_url).toBe(
      'https://closelistening.app/api/app/persons/person%3Ad/photo',
    )
  })
})
