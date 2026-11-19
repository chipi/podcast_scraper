import { describe, expect, it } from 'vitest'
import { formatFastApiDetail, readApiErrorMessage } from './readApiErrorMessage'

describe('formatFastApiDetail', () => {
  it('formats forbidden_operator_feed_keys', () => {
    expect(
      formatFastApiDetail({ error: 'forbidden_operator_feed_keys', keys: ['rss_urls'] }, 400),
    ).toContain('Feeds tab')
    expect(
      formatFastApiDetail({ error: 'forbidden_operator_feed_keys', keys: ['rss_urls'] }, 400),
    ).toContain('rss_urls')
  })

  it('joins multiple forbidden_operator_feed_keys with a comma', () => {
    expect(
      formatFastApiDetail({ error: 'forbidden_operator_feed_keys', keys: ['rss_urls', 'feeds'] }, 400),
    ).toBe('Remove feed keys from operator YAML (use the Feeds tab): rss_urls, feeds')
  })

  it('coerces non-string forbidden_operator_feed_keys entries via String()', () => {
    expect(
      formatFastApiDetail({ error: 'forbidden_operator_feed_keys', keys: [1, true, null] }, 400),
    ).toBe('Remove feed keys from operator YAML (use the Feeds tab): 1, true, null')
  })

  it('formats forbidden_operator_keys', () => {
    expect(
      formatFastApiDetail({ error: 'forbidden_operator_keys', keys: ['openai_api_key'] }, 400),
    ).toBe('Forbidden operator keys: openai_api_key')
  })

  it('joins multiple forbidden_operator_keys with a comma', () => {
    expect(
      formatFastApiDetail({ error: 'forbidden_operator_keys', keys: ['a', 'b'] }, 400),
    ).toBe('Forbidden operator keys: a, b')
  })

  it('falls through to JSON for forbidden_operator_feed_keys when keys is not an array', () => {
    // ``keys`` non-array → the dedicated branch is skipped → object is JSON.stringified.
    const detail = { error: 'forbidden_operator_feed_keys', keys: 'oops' }
    expect(formatFastApiDetail(detail, 400)).toBe(JSON.stringify(detail))
  })

  it('falls through to JSON for forbidden_operator_keys when keys is not an array', () => {
    const detail = { error: 'forbidden_operator_keys', keys: 42 }
    expect(formatFastApiDetail(detail, 400)).toBe(JSON.stringify(detail))
  })

  it('JSON-stringifies an unknown error object', () => {
    const detail = { error: 'something_else', keys: ['x'] }
    expect(formatFastApiDetail(detail, 400)).toBe(JSON.stringify(detail))
  })

  it('JSON-stringifies a plain object with no error field', () => {
    const detail = { msg: 'boom', code: 7 }
    expect(formatFastApiDetail(detail, 422)).toBe(JSON.stringify(detail))
  })

  it('appends hint for 409 forbidden string', () => {
    const s = formatFastApiDetail('Existing operator YAML contains forbidden keys', 409)
    expect(s).toContain('Feeds tab')
  })

  it('matches the 409-forbidden hint case-insensitively', () => {
    const s = formatFastApiDetail('Request FORBIDDEN by policy', 409)
    expect(s).toContain('Use the Feeds tab for RSS URLs')
  })

  it('does not append the hint for a 409 string without "forbidden"', () => {
    expect(formatFastApiDetail('Conflict detected', 409)).toBe('Conflict detected')
  })

  it('does not append the hint for a forbidden string on a non-409 status', () => {
    expect(formatFastApiDetail('forbidden keys present', 400)).toBe('forbidden keys present')
  })

  it('returns plain string for 200-style errors', () => {
    expect(formatFastApiDetail('Not found', 404)).toBe('Not found')
  })

  it('JSON-stringifies an array detail (FastAPI validation errors)', () => {
    const detail = [{ loc: ['body', 'x'], msg: 'field required' }]
    expect(formatFastApiDetail(detail, 422)).toBe(JSON.stringify(detail))
  })

  it('returns HTTP <status> when array JSON.stringify throws (circular)', () => {
    const arr: unknown[] = []
    arr.push(arr) // circular → JSON.stringify throws
    expect(formatFastApiDetail(arr, 400)).toBe('HTTP 400')
  })

  it('returns HTTP <status> when object JSON.stringify throws (circular)', () => {
    // Object with no recognised error key, but circular → final stringify throws.
    const obj: Record<string, unknown> = { error: 'unknown' }
    obj.self = obj
    expect(formatFastApiDetail(obj, 500)).toBe('HTTP 500')
  })

  it('returns HTTP <status> for null detail', () => {
    expect(formatFastApiDetail(null, 503)).toBe('HTTP 503')
  })

  it('returns HTTP <status> for numeric / boolean / undefined detail', () => {
    expect(formatFastApiDetail(42, 500)).toBe('HTTP 500')
    expect(formatFastApiDetail(true, 500)).toBe('HTTP 500')
    expect(formatFastApiDetail(undefined, 500)).toBe('HTTP 500')
  })
})

describe('readApiErrorMessage', () => {
  it('parses JSON body with object detail', async () => {
    const res = new Response(
      JSON.stringify({ detail: { error: 'forbidden_operator_feed_keys', keys: ['feeds'] } }),
      {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      },
    )
    const msg = await readApiErrorMessage(res)
    expect(msg).toContain('feeds')
  })

  it('parses JSON body with a plain string detail', async () => {
    const res = new Response(JSON.stringify({ detail: 'Not authorised' }), {
      status: 403,
      headers: { 'Content-Type': 'application/json' },
    })
    expect(await readApiErrorMessage(res)).toBe('Not authorised')
  })

  it('handles leading/trailing whitespace around a JSON body', async () => {
    const res = new Response(`\n  ${JSON.stringify({ detail: 'spaced' })}  \n`, {
      status: 400,
      headers: { 'Content-Type': 'application/json' },
    })
    expect(await readApiErrorMessage(res)).toBe('spaced')
  })

  it('matches content-type case-insensitively', async () => {
    const res = new Response(JSON.stringify({ detail: 'upper ct' }), {
      status: 400,
      headers: { 'Content-Type': 'APPLICATION/JSON; charset=utf-8' },
    })
    expect(await readApiErrorMessage(res)).toBe('upper ct')
  })

  it('falls back to raw trimmed body when JSON has no detail key', async () => {
    const body = JSON.stringify({ message: 'no detail here' })
    const res = new Response(body, {
      status: 400,
      headers: { 'Content-Type': 'application/json' },
    })
    expect(await readApiErrorMessage(res)).toBe(body.trim())
  })

  it('falls back to raw body when JSON content-type but body is malformed', async () => {
    const res = new Response('{not valid json', {
      status: 400,
      headers: { 'Content-Type': 'application/json' },
    })
    expect(await readApiErrorMessage(res)).toBe('{not valid json')
  })

  it('does not attempt JSON parse when body does not start with {', async () => {
    // JSON content-type but a non-object body (array / scalar) → raw fallback.
    const res = new Response('[1,2,3]', {
      status: 400,
      headers: { 'Content-Type': 'application/json' },
    })
    expect(await readApiErrorMessage(res)).toBe('[1,2,3]')
  })

  it('falls back to raw body when not JSON', async () => {
    const res = new Response('plain error', { status: 500 })
    expect(await readApiErrorMessage(res)).toBe('plain error')
  })

  it('trims the raw fallback body', async () => {
    const res = new Response('  padded error  ', { status: 500 })
    expect(await readApiErrorMessage(res)).toBe('padded error')
  })

  it('uses HTTP status when body empty', async () => {
    const res = new Response('', { status: 502 })
    expect(await readApiErrorMessage(res)).toBe('HTTP 502')
  })

  it('uses HTTP status when body is whitespace-only', async () => {
    const res = new Response('   \n\t ', { status: 504 })
    expect(await readApiErrorMessage(res)).toBe('HTTP 504')
  })

  it('uses HTTP status when content-type header is absent (null header → || "")', async () => {
    // A null body produces a Response with no content-type header at all,
    // exercising the ``headers.get('content-type') || ''`` fallback.
    const res = new Response(null, { status: 500 })
    expect(res.headers.get('content-type')).toBe(null)
    expect(await readApiErrorMessage(res)).toBe('HTTP 500')
  })

  it('parses JSON even when content-type is missing? — no: requires the json CT', async () => {
    // Body is JSON but no content-type header → the JSON branch is skipped,
    // raw trimmed body is returned verbatim.
    const body = JSON.stringify({ detail: 'ignored without ct' })
    const res = new Response(body, { status: 400 })
    // jsdom/undici default a string body to text/plain, so the JSON branch
    // is not taken; the raw body is returned.
    expect(await readApiErrorMessage(res)).toBe(body)
  })
})
