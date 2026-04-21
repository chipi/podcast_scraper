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

  it('formats forbidden_operator_keys', () => {
    expect(
      formatFastApiDetail({ error: 'forbidden_operator_keys', keys: ['openai_api_key'] }, 400),
    ).toContain('openai_api_key')
  })

  it('appends hint for 409 forbidden string', () => {
    const s = formatFastApiDetail('Existing operator YAML contains forbidden keys', 409)
    expect(s).toContain('Feeds tab')
  })

  it('returns plain string for 200-style errors', () => {
    expect(formatFastApiDetail('Not found', 404)).toBe('Not found')
  })
})

describe('readApiErrorMessage', () => {
  it('parses JSON body with object detail', async () => {
    const res = new Response(JSON.stringify({ detail: { error: 'forbidden_operator_feed_keys', keys: ['feeds'] } }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' },
    })
    const msg = await readApiErrorMessage(res)
    expect(msg).toContain('feeds')
  })

  it('falls back to raw body when not JSON', async () => {
    const res = new Response('plain error', { status: 500 })
    expect(await readApiErrorMessage(res)).toBe('plain error')
  })

  it('uses HTTP status when body empty', async () => {
    const res = new Response('', { status: 502 })
    expect(await readApiErrorMessage(res)).toBe('HTTP 502')
  })
})
