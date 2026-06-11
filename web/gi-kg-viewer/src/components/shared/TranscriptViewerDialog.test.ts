// @vitest-environment happy-dom
import { mount, type VueWrapper } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// TranscriptViewerDialog fetches the transcript text, an optional audio HEAD
// probe, and an optional segments sidecar — all through fetchWithTimeout. We
// mock that single entry point and route by URL + method.
const fetchWithTimeout = vi.fn()
vi.mock('../../api/httpClient', () => ({
  fetchWithTimeout: (...args: unknown[]) => fetchWithTimeout(...args),
  DEFAULT_VIEWER_FETCH_TIMEOUT_MS: 120_000,
}))

import TranscriptViewerDialog, {
  type TranscriptViewerOpenPayload,
} from './TranscriptViewerDialog.vue'

type Exposed = {
  open: (p: TranscriptViewerOpenPayload) => Promise<void>
  close: () => void
  seekToMs: (ms: number) => void
}

const mounted: VueWrapper[] = []
afterEach(() => {
  while (mounted.length) mounted.pop()!.unmount()
  fetchWithTimeout.mockReset()
})

function mockResponse(opts: {
  ok?: boolean
  status?: number
  body?: string
  contentLength?: string | null
}): Response {
  const { ok = true, status = 200, body = '', contentLength = null } = opts
  const headers = new Map<string, string>()
  if (contentLength != null) {
    headers.set('content-length', contentLength)
  }
  return {
    ok,
    status,
    headers: { get: (k: string) => headers.get(k.toLowerCase()) ?? null },
    text: async () => body,
  } as unknown as Response
}

/**
 * Default router: transcript GET returns `transcriptBody`, segments sidecar GET
 * returns 404 (none), audio HEAD probe returns 404 (hidden player). Tests can
 * override via `routeOverride`.
 */
function installFetchRouter(opts: {
  transcript?: Response | (() => Response | Promise<Response>)
  segments?: Response
  audioHead?: Response
} = {}) {
  fetchWithTimeout.mockImplementation(
    async (url: string, init?: { method?: string }) => {
      const method = init?.method ?? 'GET'
      if (method === 'HEAD' || url.includes('/api/corpus/media')) {
        return opts.audioHead ?? mockResponse({ ok: false, status: 404 })
      }
      if (url.includes('.segments.json')) {
        return opts.segments ?? mockResponse({ ok: false, status: 404 })
      }
      // transcript text-file route
      if (typeof opts.transcript === 'function') {
        return opts.transcript()
      }
      return opts.transcript ?? mockResponse({ ok: true, body: '' })
    },
  )
}

function mountDialog() {
  const w = mount(TranscriptViewerDialog, { attachTo: document.body })
  mounted.push(w)
  return w
}

function exposed(w: VueWrapper): Exposed {
  return w.vm as unknown as Exposed
}

function basePayload(
  over: Partial<TranscriptViewerOpenPayload> = {},
): TranscriptViewerOpenPayload {
  return {
    corpusRoot: '/corpus',
    transcriptRelpath: 'transcripts/ep1.txt',
    rawTabUrl: 'https://example.test/raw',
    ...over,
  }
}

const dialogEl = (w: VueWrapper) =>
  w.get('[data-testid="transcript-viewer-dialog"]').element as HTMLDialogElement

// Flush microtasks so the awaited fetch chain inside open() settles.
const settle = async (w: VueWrapper) => {
  await Promise.resolve()
  await Promise.resolve()
  await w.vm.$nextTick()
  await Promise.resolve()
  await w.vm.$nextTick()
}

describe('TranscriptViewerDialog', () => {
  beforeEach(() => {
    installFetchRouter()
  })

  it('renders a closed dialog with the static title before open()', () => {
    const w = mountDialog()
    expect(dialogEl(w).open).toBe(false)
    expect(w.get('#transcript-viewer-title').text()).toBe('Transcript')
  })

  it('opens the modal and shows loading then the transcript body', async () => {
    installFetchRouter({
      transcript: mockResponse({ ok: true, body: 'hello transcript world' }),
    })
    const w = mountDialog()
    await exposed(w).open(basePayload())
    await settle(w)
    expect(dialogEl(w).open).toBe(true)
    const body = w.get('[data-testid="transcript-viewer-body"]')
    expect(body.text()).toContain('hello transcript world')
  })

  it('renders header subtitle, audio label, and char-range label from payload', async () => {
    installFetchRouter({ transcript: mockResponse({ ok: true, body: 'x' }) })
    const w = mountDialog()
    await exposed(w).open(
      basePayload({
        subtitle: 'Episode 42',
        audioTimingLabel: '00:10 – 00:20',
        charPositionLabel: 'Characters 0–60',
      }),
    )
    await settle(w)
    expect(w.text()).toContain('Episode 42')
    expect(w.text()).toContain('00:10 – 00:20')
    const charRange = w.get('[data-testid="transcript-viewer-char-range"]')
    expect(charRange.text()).toContain('Characters 0–60')
  })

  it('renders the raw-tab link from rawTabUrl', async () => {
    installFetchRouter({ transcript: mockResponse({ ok: true, body: 'x' }) })
    const w = mountDialog()
    await exposed(w).open(basePayload({ rawTabUrl: 'https://raw.test/t.txt' }))
    await settle(w)
    const link = w.get('[data-testid="transcript-viewer-open-raw"]')
    expect(link.attributes('href')).toBe('https://raw.test/t.txt')
    expect(link.attributes('target')).toBe('_blank')
  })

  it('highlights a single GI char range via charStart/charEnd', async () => {
    installFetchRouter({
      transcript: mockResponse({ ok: true, body: 'ABCDEFGHIJ' }),
    })
    const w = mountDialog()
    await exposed(w).open(basePayload({ charStart: 2, charEnd: 5 }))
    await settle(w)
    const mark = w.get('[data-testid="transcript-viewer-highlight"]')
    expect(mark.text()).toBe('CDE')
  })

  it('highlights merged multi-span ranges via charRanges', async () => {
    installFetchRouter({
      transcript: mockResponse({ ok: true, body: '0123456789' }),
    })
    const w = mountDialog()
    await exposed(w).open(
      basePayload({
        charRanges: [
          { charStart: 1, charEnd: 3 },
          { charStart: 6, charEnd: 8 },
        ],
      }),
    )
    await settle(w)
    const marks = w.findAll('[data-testid="transcript-viewer-highlight"]')
    expect(marks).toHaveLength(2)
    expect(marks[0].text()).toBe('12')
    expect(marks[1].text()).toBe('67')
  })

  it('renders plain full text when no highlight range is given', async () => {
    installFetchRouter({
      transcript: mockResponse({ ok: true, body: 'just plain text' }),
    })
    const w = mountDialog()
    await exposed(w).open(basePayload())
    await settle(w)
    expect(w.get('[data-testid="transcript-viewer-body"]').text()).toContain(
      'just plain text',
    )
    expect(w.find('[data-testid="transcript-viewer-highlight"]').exists()).toBe(false)
  })

  it('shows an error message when the transcript fetch rejects', async () => {
    installFetchRouter({
      transcript: () => {
        throw new Error('network boom')
      },
    })
    const w = mountDialog()
    await exposed(w).open(basePayload())
    await settle(w)
    expect(w.text()).toContain('network boom')
    expect(w.find('[data-testid="transcript-viewer-body"]').exists()).toBe(false)
  })

  it('shows an HTTP error message on a non-ok transcript response', async () => {
    installFetchRouter({
      transcript: mockResponse({ ok: false, status: 503 }),
    })
    const w = mountDialog()
    await exposed(w).open(basePayload())
    await settle(w)
    expect(w.text()).toContain('HTTP 503 loading transcript')
  })

  it('shows the oversized message when Content-Length exceeds maxBytes', async () => {
    installFetchRouter({
      transcript: mockResponse({
        ok: true,
        body: 'small',
        contentLength: String(10 * 1024 * 1024),
      }),
    })
    const w = mountDialog()
    await exposed(w).open(basePayload({ maxBytes: 1024 }))
    await settle(w)
    expect(w.text()).toContain('too large to load in the viewer')
    expect(w.find('[data-testid="transcript-viewer-body"]').exists()).toBe(false)
  })

  it('renders a timeline disclosure when the segments sidecar loads', async () => {
    installFetchRouter({
      transcript: mockResponse({ ok: true, body: 'body text' }),
      segments: mockResponse({
        ok: true,
        body: JSON.stringify([
          { start: 0, end: 1.5, text: 'first' },
          { start: 1.5, end: 3, text: 'second' },
        ]),
      }),
    })
    const w = mountDialog()
    await exposed(w).open(basePayload())
    await settle(w)
    const timeline = w.find('[data-testid="transcript-viewer-timeline"]')
    expect(timeline.exists()).toBe(true)
    expect(w.text()).toContain('Timeline (2 segments)')
    expect(timeline.text()).toContain('first')
  })

  it('omits the timeline when no segments sidecar is available', async () => {
    installFetchRouter({
      transcript: mockResponse({ ok: true, body: 'body text' }),
      segments: mockResponse({ ok: false, status: 404 }),
    })
    const w = mountDialog()
    await exposed(w).open(basePayload())
    await settle(w)
    expect(w.find('[data-testid="transcript-viewer-timeline"]').exists()).toBe(false)
  })

  it('renders the audio player when the HEAD probe succeeds', async () => {
    installFetchRouter({
      transcript: mockResponse({ ok: true, body: 'body' }),
      audioHead: mockResponse({ ok: true, status: 200 }),
    })
    const w = mountDialog()
    await exposed(w).open(basePayload())
    await settle(w)
    expect(w.find('[data-testid="transcript-viewer-audio"]').exists()).toBe(true)
  })

  it('hides the audio player when the HEAD probe 404s', async () => {
    installFetchRouter({
      transcript: mockResponse({ ok: true, body: 'body' }),
      audioHead: mockResponse({ ok: false, status: 404 }),
    })
    const w = mountDialog()
    await exposed(w).open(basePayload())
    await settle(w)
    expect(w.find('[data-testid="transcript-viewer-audio"]').exists()).toBe(false)
  })

  it('closes via the Close button and stops dialog.open', async () => {
    installFetchRouter({ transcript: mockResponse({ ok: true, body: 'b' }) })
    const w = mountDialog()
    await exposed(w).open(basePayload())
    await settle(w)
    expect(dialogEl(w).open).toBe(true)
    await w.get('button').trigger('click') // first button in header is Close
    expect(dialogEl(w).open).toBe(false)
  })

  it('closes via the exposed close() method', async () => {
    installFetchRouter({ transcript: mockResponse({ ok: true, body: 'b' }) })
    const w = mountDialog()
    await exposed(w).open(basePayload())
    await settle(w)
    exposed(w).close()
    await w.vm.$nextTick()
    expect(dialogEl(w).open).toBe(false)
  })

  it('closes when a backdrop click hits the dialog element itself', async () => {
    installFetchRouter({ transcript: mockResponse({ ok: true, body: 'b' }) })
    const w = mountDialog()
    await exposed(w).open(basePayload())
    await settle(w)
    const el = dialogEl(w)
    // target === dialog element → treated as a backdrop click.
    await w.get('[data-testid="transcript-viewer-dialog"]').trigger('click')
    expect(el.open).toBe(false)
  })

  it('does not close when a click originates from inside the dialog content', async () => {
    installFetchRouter({ transcript: mockResponse({ ok: true, body: 'b' }) })
    const w = mountDialog()
    await exposed(w).open(basePayload())
    await settle(w)
    // Click the title (a child) — target !== dialog element → stays open.
    await w.get('#transcript-viewer-title').trigger('click')
    expect(dialogEl(w).open).toBe(true)
  })

  it('reopening resets prior error state', async () => {
    installFetchRouter({
      transcript: () => {
        throw new Error('first boom')
      },
    })
    const w = mountDialog()
    await exposed(w).open(basePayload())
    await settle(w)
    expect(w.text()).toContain('first boom')

    installFetchRouter({
      transcript: mockResponse({ ok: true, body: 'recovered body' }),
    })
    await exposed(w).open(basePayload())
    await settle(w)
    expect(w.text()).not.toContain('first boom')
    expect(w.get('[data-testid="transcript-viewer-body"]').text()).toContain(
      'recovered body',
    )
  })

  it('exposes seekToMs without throwing when no audio element is mounted', async () => {
    installFetchRouter({ transcript: mockResponse({ ok: true, body: 'b' }) })
    const w = mountDialog()
    await exposed(w).open(basePayload())
    await settle(w)
    // No audio player rendered (HEAD 404 by default) → seek is queued, no throw.
    expect(() => exposed(w).seekToMs(2000)).not.toThrow()
    expect(() => exposed(w).seekToMs(-5)).not.toThrow()
  })
})
