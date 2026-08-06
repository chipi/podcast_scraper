import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { McpConnectionConfig, McpTokenMeta } from '../services/types'
import ConnectedAgents from './ConnectedAgents.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

function cfg(over: Partial<McpConnectionConfig> = {}): McpConnectionConfig {
  return {
    connector_url: 'https://mcp.closelistening.app',
    authorization_server: 'https://closelistening.app',
    oauth_enabled: true,
    ...over,
  }
}

function tok(over: Partial<McpTokenMeta> = {}): McpTokenMeta {
  return { id: 't1', label: 'Claude Code', created_at: 1_700_000_000, last_used_at: null, ...over }
}

function mountAgents() {
  return mount(ConnectedAgents, { global: { plugins: [i18n] } })
}

beforeEach(() => {
  vi.spyOn(api, 'getMcpConfig').mockResolvedValue(cfg())
  vi.spyOn(api, 'getMcpTokens').mockResolvedValue([tok()])
  vi.spyOn(api, 'getMcpConnections').mockResolvedValue([])
})
afterEach(() => vi.restoreAllMocks())

describe('ConnectedAgents', () => {
  it('shows the connector URL and existing tokens', async () => {
    const w = mountAgents()
    await flushPromises()
    expect(w.text()).toContain('https://mcp.closelistening.app')
    expect(w.text()).toContain('Claude Code')
    expect(w.text()).toContain('never used')
  })

  it('hides the connector block and hints when the URL is unconfigured', async () => {
    vi.spyOn(api, 'getMcpConfig').mockResolvedValue(cfg({ connector_url: null, oauth_enabled: false }))
    const w = mountAgents()
    await flushPromises()
    expect(w.text()).not.toContain('https://mcp.closelistening.app')
    expect(w.text()).toContain(en.agents.connectorUnset)
  })

  it('creates a token and surfaces the plaintext once', async () => {
    const create = vi
      .spyOn(api, 'createMcpToken')
      .mockResolvedValue({ token: 'clp_mcp_SECRET', meta: tok({ id: 't2', label: 'Cursor' }) })
    const w = mountAgents()
    await flushPromises()

    await w.find('input[type="text"]').setValue('Cursor')
    await w.findAll('button').find((b) => b.text() === en.agents.create)!.trigger('click')
    await flushPromises()

    expect(create).toHaveBeenCalledWith('Cursor')
    expect(w.text()).toContain('clp_mcp_SECRET')
    expect(w.text()).toContain(en.agents.createdOnce)
  })

  it('lists connected OAuth apps and disconnects one', async () => {
    vi.spyOn(api, 'getMcpConnections').mockResolvedValue([
      { client_id: 'mcpc_1', client_name: 'claude.ai', scopes: ['mcp:read'], connected_at: 1_700_000_000 },
    ])
    const revoke = vi.spyOn(api, 'revokeMcpConnection').mockResolvedValue([])
    const w = mountAgents()
    await flushPromises()
    expect(w.text()).toContain('claude.ai')

    await w.findAll('button').find((b) => b.text() === en.agents.disconnect)!.trigger('click')
    await flushPromises()
    expect(revoke).toHaveBeenCalledWith('mcpc_1')
  })

  it('revokes a token', async () => {
    const revoke = vi.spyOn(api, 'revokeMcpToken').mockResolvedValue([])
    const w = mountAgents()
    await flushPromises()

    await w.findAll('button').find((b) => b.text() === en.agents.revoke)!.trigger('click')
    await flushPromises()

    expect(revoke).toHaveBeenCalledWith('t1')
    expect(w.text()).toContain(en.agents.noTokens)
  })

  it('shows an error when loading fails', async () => {
    vi.spyOn(api, 'getMcpConfig').mockRejectedValue(new Error('boom'))
    const w = mountAgents()
    await flushPromises()
    expect(w.text()).toContain(en.agents.error)
  })
})
