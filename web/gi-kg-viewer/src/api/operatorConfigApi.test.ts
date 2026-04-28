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
import { getOperatorConfig } from './operatorConfigApi'

describe('operatorConfigApi.getOperatorConfig', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
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
