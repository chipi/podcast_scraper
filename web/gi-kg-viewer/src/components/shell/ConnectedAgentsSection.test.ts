// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const getMcpConfig = vi.fn()
const listMcpTokens = vi.fn()
const createMcpToken = vi.fn()
const revokeMcpToken = vi.fn()
const listMcpConnections = vi.fn()
const revokeMcpConnection = vi.fn()
vi.mock('../../api/mcpApi', () => ({
  getMcpConfig: (...a: unknown[]) => getMcpConfig(...a),
  listMcpTokens: (...a: unknown[]) => listMcpTokens(...a),
  createMcpToken: (...a: unknown[]) => createMcpToken(...a),
  revokeMcpToken: (...a: unknown[]) => revokeMcpToken(...a),
  listMcpConnections: (...a: unknown[]) => listMcpConnections(...a),
  revokeMcpConnection: (...a: unknown[]) => revokeMcpConnection(...a),
}))

import ConnectedAgentsSection from './ConnectedAgentsSection.vue'

const CFG = {
  connector_url: 'https://mcp.closelistening.app',
  authorization_server: 'https://closelistening.app',
  oauth_enabled: true,
}
const TOK = { id: 't1', label: 'Claude Code', created_at: 1_700_000_000, last_used_at: null }

beforeEach(() => {
  listMcpConnections.mockResolvedValue([])
})
afterEach(() => {
  getMcpConfig.mockReset()
  listMcpTokens.mockReset()
  createMcpToken.mockReset()
  revokeMcpToken.mockReset()
  listMcpConnections.mockReset()
  revokeMcpConnection.mockReset()
})

function mountSection() {
  return mount(ConnectedAgentsSection, { props: { active: true }, attachTo: document.body })
}

describe('ConnectedAgentsSection', () => {
  it('shows the connector URL and existing tokens on activation', async () => {
    getMcpConfig.mockResolvedValue(CFG)
    listMcpTokens.mockResolvedValue([TOK])
    const w = mountSection()
    await flushPromises()
    expect(w.text()).toContain('https://mcp.closelistening.app')
    expect(w.text()).toContain('Claude Code')
    expect(w.text()).toContain('never used')
  })

  it('hides the connector block when unconfigured', async () => {
    getMcpConfig.mockResolvedValue({ ...CFG, connector_url: null, oauth_enabled: false })
    listMcpTokens.mockResolvedValue([])
    const w = mountSection()
    await flushPromises()
    expect(w.text()).not.toContain('https://mcp.closelistening.app')
    expect(w.text()).toContain("isn't configured")
    expect(w.text()).toContain('No tokens yet.')
  })

  it('creates a token and surfaces the plaintext once', async () => {
    getMcpConfig.mockResolvedValue(CFG)
    listMcpTokens.mockResolvedValue([])
    createMcpToken.mockResolvedValue({ token: 'clp_mcp_SECRET', meta: { ...TOK, id: 't2', label: 'Cursor' } })
    const w = mountSection()
    await flushPromises()

    await w.find('[data-testid="agents-token-label"]').setValue('Cursor')
    await w.find('[data-testid="agents-token-create"]').trigger('click')
    await flushPromises()

    expect(createMcpToken).toHaveBeenCalledWith('Cursor')
    expect(w.find('[data-testid="agents-fresh-secret"]').text()).toContain('clp_mcp_SECRET')
  })

  it('lists connected OAuth apps and disconnects one', async () => {
    getMcpConfig.mockResolvedValue(CFG)
    listMcpTokens.mockResolvedValue([])
    listMcpConnections.mockResolvedValue([
      { client_id: 'mcpc_1', client_name: 'claude.ai', scopes: ['mcp:read'], connected_at: 1_700_000_000 },
    ])
    revokeMcpConnection.mockResolvedValue([])
    const w = mountSection()
    await flushPromises()
    expect(w.text()).toContain('claude.ai')

    await w.find('[data-testid="agents-connection-revoke"]').trigger('click')
    await flushPromises()
    expect(revokeMcpConnection).toHaveBeenCalledWith('mcpc_1')
  })

  it('revokes a token', async () => {
    getMcpConfig.mockResolvedValue(CFG)
    listMcpTokens.mockResolvedValue([TOK])
    revokeMcpToken.mockResolvedValue([])
    const w = mountSection()
    await flushPromises()

    await w.find('[data-testid="agents-token-revoke"]').trigger('click')
    await flushPromises()

    expect(revokeMcpToken).toHaveBeenCalledWith('t1')
    expect(w.text()).toContain('No tokens yet.')
  })

  it('surfaces an error when loading fails', async () => {
    getMcpConfig.mockRejectedValue(new Error('boom'))
    listMcpTokens.mockResolvedValue([])
    const w = mountSection()
    await flushPromises()
    expect(w.find('[data-testid="agents-error"]').text()).toContain('boom')
  })
})
