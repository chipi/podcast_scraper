/**
 * MCP "Connected agents" API for the viewer (RFC-112 §5). Same `/api/app/mcp/*` routes the player
 * uses; every call is `mcp_access`-gated server-side (403 without the entitlement), so the viewer
 * only surfaces this section to an entitled operator.
 */
import { fetchWithTimeout } from './httpClient'

const BASE = '/api/app'

export interface McpConnectionConfig {
  connector_url: string | null
  authorization_server: string | null
  oauth_enabled: boolean
}

export interface McpTokenMeta {
  id: string
  label: string
  created_at: number
  last_used_at: number | null
}

export interface McpTokenCreated {
  token: string
  meta: McpTokenMeta
}

export interface McpConnection {
  client_id: string
  client_name: string
  scopes: string[]
  connected_at: number
}

/** Connector wiring the section shows (resource URL + OAuth status). */
export async function getMcpConfig(): Promise<McpConnectionConfig> {
  const res = await fetchWithTimeout(`${BASE}/mcp/config`)
  if (!res.ok) throw new Error(`GET /mcp/config failed: ${res.status}`)
  return (await res.json()) as McpConnectionConfig
}

/** List the operator's MCP tokens (metadata only — the secret is never returned after creation). */
export async function listMcpTokens(): Promise<McpTokenMeta[]> {
  const res = await fetchWithTimeout(`${BASE}/mcp/tokens`)
  if (!res.ok) throw new Error(`GET /mcp/tokens failed: ${res.status}`)
  return ((await res.json()).items ?? []) as McpTokenMeta[]
}

/** Mint a token; the plaintext is returned ONCE (copy-then-forget). */
export async function createMcpToken(label: string): Promise<McpTokenCreated> {
  const res = await fetchWithTimeout(`${BASE}/mcp/tokens`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ label }),
  })
  if (!res.ok) throw new Error(`POST /mcp/tokens failed: ${res.status}`)
  return (await res.json()) as McpTokenCreated
}

/** Revoke a token by id; returns the remaining tokens. */
export async function revokeMcpToken(id: string): Promise<McpTokenMeta[]> {
  const res = await fetchWithTimeout(`${BASE}/mcp/tokens/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  })
  if (!res.ok) throw new Error(`DELETE /mcp/tokens/${id} failed: ${res.status}`)
  return ((await res.json()).items ?? []) as McpTokenMeta[]
}

/** The OAuth agents the operator has connected. */
export async function listMcpConnections(): Promise<McpConnection[]> {
  const res = await fetchWithTimeout(`${BASE}/mcp/connections`)
  if (!res.ok) throw new Error(`GET /mcp/connections failed: ${res.status}`)
  return ((await res.json()).items ?? []) as McpConnection[]
}

/** Disconnect an OAuth agent (forget consent + drop its live tokens); returns the remaining. */
export async function revokeMcpConnection(clientId: string): Promise<McpConnection[]> {
  const res = await fetchWithTimeout(`${BASE}/mcp/connections/${encodeURIComponent(clientId)}`, {
    method: 'DELETE',
  })
  if (!res.ok) throw new Error(`DELETE /mcp/connections/${clientId} failed: ${res.status}`)
  return ((await res.json()).items ?? []) as McpConnection[]
}
