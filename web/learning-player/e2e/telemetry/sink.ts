import type { Page, Request } from '@playwright/test'

/**
 * Observes the two telemetry sinks WITHOUT intercepting them (#2263).
 *
 * `page.route` is deliberately not used. The point of this tier is that events reach the real Umami
 * and the real GlitchTip, so the surface can be read back afterwards and shown to agree with what
 * the app sent. A route handler that fulfilled these requests would prove only that the app called
 * a function — which the unit suite already proves, and which is exactly the kind of evidence that
 * let a website id that does not exist look wired for weeks.
 *
 * So: observe the wire, let it fly, and verify twice — once here on what left the browser, once on
 * the surface on what arrived.
 */

/** One Umami beacon, parsed. `name` is undefined for a pageview (`type: 'event'` with no name). */
export interface UmamiBeacon {
  type: string
  name?: string
  url?: string
  referrer?: string
  title?: string
  hostname?: string
  website?: string
  /** Custom event properties — the typed `EventProps` payload. */
  data?: Record<string, unknown>
  /** `window.umami.identify` sends the analytics id here. */
  id?: string
  raw: unknown
}

/** One Sentry/GlitchTip envelope, split into its newline-delimited items. */
export interface SentryEnvelope {
  header: Record<string, unknown>
  items: Record<string, unknown>[]
  rawText: string
}

export class TelemetrySink {
  readonly umami: UmamiBeacon[] = []
  readonly sentry: SentryEnvelope[] = []
  /** Requests that looked like telemetry but could not be parsed — never silently dropped. */
  readonly unparsed: { url: string; body: string | null; error: string }[] = []

  /** Every custom event name seen, in order, duplicates included. */
  names(): string[] {
    return this.umami.filter((b) => b.name).map((b) => b.name as string)
  }

  /** Every beacon carrying this event name. */
  byName(name: string): UmamiBeacon[] {
    return this.umami.filter((b) => b.name === name)
  }

  /** The first beacon with this name, or undefined. */
  first(name: string): UmamiBeacon | undefined {
    return this.byName(name)[0]
  }

  /** Every URL any beacon reported — the surface the `?q=` scrub has to hold on. */
  reportedUrls(): string[] {
    return this.umami.map((b) => b.url ?? '').filter(Boolean)
  }

  /** All exception/message values across every captured envelope. */
  sentryTitles(): string[] {
    const out: string[] = []
    for (const env of this.sentry) {
      for (const item of env.items) {
        const ex = (item as { exception?: { values?: { type?: string; value?: string }[] } }).exception
        for (const v of ex?.values ?? []) out.push(`${v.type ?? ''}: ${v.value ?? ''}`)
        const msg = (item as { message?: unknown }).message
        if (typeof msg === 'string') out.push(msg)
      }
    }
    return out
  }

  /**
   * Every URL-ish string anywhere inside the captured envelopes.
   *
   * Flattens the whole structure rather than checking the few fields a scrub is known to touch:
   * the search term leaked through `breadcrumbs[].data.to` specifically BECAUSE nobody was looking
   * at that field. A test that only inspects the fields the fix touches cannot catch the next one.
   */
  sentryUrlStrings(): string[] {
    const out: string[] = []
    const walk = (v: unknown): void => {
      if (typeof v === 'string') {
        if (v.includes('/') || v.includes('?') || v.includes('=')) out.push(v)
        return
      }
      if (Array.isArray(v)) {
        v.forEach(walk)
        return
      }
      if (v && typeof v === 'object') Object.values(v as Record<string, unknown>).forEach(walk)
    }
    for (const env of this.sentry) {
      walk(env.header)
      env.items.forEach(walk)
    }
    return out
  }

  /** Poll until an event with `name` has been seen, else throw with what WAS seen. */
  async waitForEvent(name: string, timeoutMs = 15_000): Promise<UmamiBeacon> {
    const deadline = Date.now() + timeoutMs
    for (;;) {
      const hit = this.first(name)
      if (hit) return hit
      if (Date.now() > deadline) {
        throw new Error(
          `timed out waiting for Umami event '${name}'. Seen: [${[...new Set(this.names())].join(', ')}]`,
        )
      }
      await new Promise((r) => setTimeout(r, 150))
    }
  }

  /** Poll until at least `n` envelopes have arrived. */
  async waitForSentry(n = 1, timeoutMs = 20_000): Promise<void> {
    const deadline = Date.now() + timeoutMs
    while (this.sentry.length < n) {
      if (Date.now() > deadline) {
        throw new Error(
          `timed out waiting for ${n} GlitchTip envelope(s); got ${this.sentry.length}. ` +
            `Titles: [${this.sentryTitles().join(' | ')}]`,
        )
      }
      await new Promise((r) => setTimeout(r, 200))
    }
  }
}

function recordUmami(sink: TelemetrySink, req: Request): void {
  const body = req.postData()
  try {
    const parsed = JSON.parse(body ?? '') as { type?: string; payload?: Record<string, unknown> }
    const p = parsed.payload ?? {}
    sink.umami.push({
      type: String(parsed.type ?? 'event'),
      name: typeof p.name === 'string' ? p.name : undefined,
      url: typeof p.url === 'string' ? p.url : undefined,
      referrer: typeof p.referrer === 'string' ? p.referrer : undefined,
      title: typeof p.title === 'string' ? p.title : undefined,
      hostname: typeof p.hostname === 'string' ? p.hostname : undefined,
      website: typeof p.website === 'string' ? p.website : undefined,
      data: (p.data as Record<string, unknown>) ?? undefined,
      id: typeof p.id === 'string' ? p.id : undefined,
      raw: parsed,
    })
  } catch (err) {
    sink.unparsed.push({ url: req.url(), body, error: String(err) })
  }
}

function recordSentry(sink: TelemetrySink, req: Request): void {
  const body = req.postData()
  if (!body) {
    sink.unparsed.push({ url: req.url(), body, error: 'empty envelope body' })
    return
  }
  // An envelope is newline-delimited JSON: a header line, then (item-header, item-payload) pairs.
  // Payloads can be non-JSON (attachments), so parse leniently and keep what decodes.
  const lines = body.split('\n').filter((l) => l.trim().length > 0)
  const decoded: Record<string, unknown>[] = []
  let header: Record<string, unknown> = {}
  lines.forEach((line, i) => {
    try {
      const obj = JSON.parse(line) as Record<string, unknown>
      if (i === 0) header = obj
      else decoded.push(obj)
    } catch {
      /* attachment or gzipped item — not a URL carrier, so not interesting here */
    }
  })
  sink.sentry.push({ header, items: decoded, rawText: body })
}

/**
 * Start observing. Call before the first `page.goto`, since `landing_view` and the first
 * `screen_view` fire during bootstrap and would otherwise be missed.
 */
export function attachSink(page: Page): TelemetrySink {
  const sink = new TelemetrySink()
  page.on('request', (req) => {
    if (req.method() !== 'POST') return
    const url = req.url()
    // Umami's collector. Matched on the path so a change of host (loopback vs tailnet) does not
    // quietly stop the recorder while the test keeps passing on an empty list.
    if (url.includes('/api/send')) {
      recordUmami(sink, req)
      return
    }
    // GlitchTip/Sentry ingest: modern SDKs use /envelope/, older paths use /store/.
    if (url.includes('/envelope') || /\/api\/\d+\/store/.test(url)) {
      recordSentry(sink, req)
    }
  })
  return sink
}

/** The dev Umami website id this tier must report into — never the prod site. */
export const DEV_UMAMI_WEBSITE_ID = '3ccaa1bc-fcc5-444d-9252-d20d60d58eed'

/** The dev GlitchTip project this tier must report into — never prod project 5. */
export const DEV_GLITCHTIP_PROJECT_ID = '20'
