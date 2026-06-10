/**
 * Vitest unit suite for ``operatorConfigApi`` parsing — focused on the
 * RFC-081 §Layer 1 ``default_profile`` field added in #692.
 *
 * The logic that uses ``default_profile`` lives in
 * ``components/shell/StatusBar.vue`` (operator-config dropdown
 * preselection). Testing it via the api boundary keeps the assertion
 * targeted at the contract: server sends ``default_profile`` either as
 * a string (current allowlist's preferred profile) or null/undefined
 * (no preselection), and the client passes it through unchanged so
 * downstream code can use ``payload.default_profile`` directly.
 */

import { afterEach, describe, expect, it, vi } from 'vitest'
import { getOperatorConfig, putOperatorConfig } from './operatorConfigApi'

/**
 * Stub ``fetch`` returning one response. ``contentType`` drives how
 * ``readApiErrorMessage`` parses error bodies (JSON ``detail`` vs raw text).
 */
function mockFetchResponse(opts: {
  ok: boolean
  status?: number
  json?: unknown
  text?: string
  contentType?: string
}): void {
  const status = opts.status ?? (opts.ok ? 200 : 500)
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({
      ok: opts.ok,
      status,
      headers: {
        get: (name: string) =>
          name.toLowerCase() === 'content-type' ? opts.contentType ?? '' : null,
      },
      json: async () => opts.json,
      text: async () => opts.text ?? '',
    })) as unknown as typeof fetch,
  )
}

describe('operatorConfigApi.getOperatorConfig', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('GETs /api/operator-config with trimmed path and AbortSignal', async () => {
    mockFetchResponse({
      ok: true,
      json: {
        corpus_path: '/app/output',
        operator_config_path: '/app/output/viewer_operator.yaml',
        content: 'x',
        available_profiles: [],
      },
    })
    await getOperatorConfig('  /app/output  ')
    const [url, init] = vi.mocked(fetch).mock.calls[0]
    expect(url).toBe('/api/operator-config?path=%2Fapp%2Foutput')
    // fetchWithTimeout injects an AbortSignal even when init is undefined.
    expect((init as RequestInit).signal).toBeInstanceOf(AbortSignal)
    expect((init as RequestInit).method).toBeUndefined()
  })

  it('parses corpus_path, operator_config_path and content', async () => {
    mockFetchResponse({
      ok: true,
      json: {
        corpus_path: '/c',
        operator_config_path: '/c/viewer_operator.yaml',
        content: 'profile: x\n',
        available_profiles: ['x'],
      },
    })
    const payload = await getOperatorConfig('/c')
    expect(payload.corpus_path).toBe('/c')
    expect(payload.operator_config_path).toBe('/c/viewer_operator.yaml')
    expect(payload.content).toBe('profile: x\n')
  })

  it('throws plain-text error body when not ok', async () => {
    mockFetchResponse({ ok: false, status: 500, text: 'boom' })
    await expect(getOperatorConfig('/c')).rejects.toThrow('boom')
  })

  it('throws HTTP status when not ok and empty body', async () => {
    mockFetchResponse({ ok: false, status: 422, text: '' })
    await expect(getOperatorConfig('/c')).rejects.toThrow('HTTP 422')
  })

  it('extracts FastAPI JSON detail string on error', async () => {
    mockFetchResponse({
      ok: false,
      status: 400,
      contentType: 'application/json',
      text: '{"detail":"corpus path is not allowed"}',
    })
    await expect(getOperatorConfig('/c')).rejects.toThrow('corpus path is not allowed')
  })

  it('propagates network errors from fetch', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new Error('offline')
      }) as unknown as typeof fetch,
    )
    await expect(getOperatorConfig('/c')).rejects.toThrow('offline')
  })

  it('passes default_profile through when server sets it (preprod cloud_thin)', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        json: async () => ({
          corpus_path: '/app/output',
          operator_config_path: '/app/output/viewer_operator.yaml',
          content: '',
          available_profiles: ['cloud_thin'],
          default_profile: 'cloud_thin',
        }),
      })) as unknown as typeof fetch,
    )

    const payload = await getOperatorConfig('/app/output')
    expect(payload.default_profile).toBe('cloud_thin')
    expect(payload.available_profiles).toEqual(['cloud_thin'])
  })

  it('default_profile is undefined when server omits it (dev / CI default)', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        json: async () => ({
          corpus_path: '/tmp/corpus',
          operator_config_path: '/tmp/corpus/viewer_operator.yaml',
          content: 'profile: cloud_balanced\n',
          available_profiles: ['cloud_balanced', 'cloud_thin'],
          // server omits default_profile when PODCAST_DEFAULT_PROFILE is unset
        }),
      })) as unknown as typeof fetch,
    )

    const payload = await getOperatorConfig('/tmp/corpus')
    expect(payload.default_profile).toBeUndefined()
  })

  it('default_profile null is preserved as null (not coerced to undefined)', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        json: async () => ({
          corpus_path: '/tmp/corpus',
          operator_config_path: '/tmp/corpus/viewer_operator.yaml',
          content: '',
          available_profiles: ['cloud_thin'],
          default_profile: null,
        }),
      })) as unknown as typeof fetch,
    )

    const payload = await getOperatorConfig('/tmp/corpus')
    expect(payload.default_profile).toBeNull()
  })
})

describe('operatorConfigApi.putOperatorConfig', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('PUTs JSON content to /api/operator-config with trimmed path', async () => {
    mockFetchResponse({
      ok: true,
      json: {
        corpus_path: '/c',
        operator_config_path: '/c/viewer_operator.yaml',
        content: 'profile: cloud_thin\n',
        available_profiles: ['cloud_thin'],
      },
    })
    const payload = await putOperatorConfig('  /c  ', 'profile: cloud_thin\n')
    expect(payload.content).toBe('profile: cloud_thin\n')
    const [url, init] = vi.mocked(fetch).mock.calls[0]
    expect(url).toBe('/api/operator-config?path=%2Fc')
    const ri = init as RequestInit
    expect(ri.method).toBe('PUT')
    expect(ri.headers).toEqual({ 'Content-Type': 'application/json' })
    expect(ri.body).toBe(JSON.stringify({ content: 'profile: cloud_thin\n' }))
    expect(ri.signal).toBeInstanceOf(AbortSignal)
  })

  it('serializes empty content into the request body', async () => {
    mockFetchResponse({
      ok: true,
      json: {
        corpus_path: '/c',
        operator_config_path: '/c/viewer_operator.yaml',
        content: '',
        available_profiles: [],
      },
    })
    await putOperatorConfig('/c', '')
    const init = vi.mocked(fetch).mock.calls[0][1] as RequestInit
    expect(JSON.parse(init.body as string)).toEqual({ content: '' })
  })

  it('throws plain-text error body when not ok', async () => {
    mockFetchResponse({ ok: false, status: 500, text: 'write failed' })
    await expect(putOperatorConfig('/c', 'x')).rejects.toThrow('write failed')
  })

  it('throws HTTP status when not ok and empty body', async () => {
    mockFetchResponse({ ok: false, status: 409, text: '' })
    await expect(putOperatorConfig('/c', 'x')).rejects.toThrow('HTTP 409')
  })

  it('surfaces forbidden-operator-keys FastAPI detail on 409', async () => {
    mockFetchResponse({
      ok: false,
      status: 409,
      contentType: 'application/json',
      text: '{"detail":{"error":"forbidden_operator_keys","keys":["feeds","secret"]}}',
    })
    await expect(putOperatorConfig('/c', 'x')).rejects.toThrow(
      'Forbidden operator keys: feeds, secret',
    )
  })

  it('propagates network errors from fetch', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new Error('socket closed')
      }) as unknown as typeof fetch,
    )
    await expect(putOperatorConfig('/c', 'x')).rejects.toThrow('socket closed')
  })
})

/**
 * Regression-test the dropdown preselection rule used in
 * ``StatusBar.vue:loadSourcesTab``:
 *
 *   operatorProfileSelected.value = sp.profile.trim() || defaultFromServer
 *
 * Saved-on-disk profile takes priority over env default; env default is
 * the fallback when the corpus has no profile saved yet.
 */
describe('dropdown preselection rule (mirrors StatusBar.vue logic)', () => {
  function effectiveProfile(savedFromYaml: string, defaultFromServer: string): string {
    return savedFromYaml.trim() || defaultFromServer
  }

  it('saved-on-disk profile wins over env default', () => {
    expect(effectiveProfile('cloud_balanced', 'cloud_thin')).toBe('cloud_balanced')
  })

  it('env default fills in when on-disk is empty (fresh corpus)', () => {
    expect(effectiveProfile('', 'cloud_thin')).toBe('cloud_thin')
  })

  it('env default fills in when on-disk is whitespace-only', () => {
    expect(effectiveProfile('   ', 'cloud_thin')).toBe('cloud_thin')
  })

  it('both empty yields empty (legacy "None" preselection)', () => {
    expect(effectiveProfile('', '')).toBe('')
  })

  it('on-disk only with no env default still wins', () => {
    expect(effectiveProfile('cloud_balanced', '')).toBe('cloud_balanced')
  })
})
