/**
 * RFC-088 v2 enrichment-config API client (viewer side).
 *
 * Wraps the server routes under ``server/routes/enrichment_config.py``:
 *   - GET  /api/enrichment/config?path=<corpus>
 *   - PUT  /api/enrichment/config?path=<corpus>
 *   - GET  /api/enrichment/config/schema
 *   - GET  /api/enrichment/provider-types
 */

import { fetchWithTimeout } from './httpClient'
import { readApiErrorMessage } from './readApiErrorMessage'

/** Raw JSON Schema fragment as fetched from the server. UI form generation reads this. */
export type JsonSchemaFragment = Record<string, unknown>

export interface EnrichmentConfigResponse {
  corpus_path: string
  profile: string | null
  profile_block: Record<string, unknown>
  operator_block: Record<string, unknown>
  resolved_block: Record<string, unknown>
}

export interface ProviderTypeInfo {
  name: string
  protocol: string
  description: string
  params_schema: JsonSchemaFragment
}

export interface ProviderTypesResponse {
  by_protocol: Record<string, ProviderTypeInfo[]>
}

const TIMEOUT_MS = 8_000

export async function getEnrichmentConfig(corpusPath: string): Promise<EnrichmentConfigResponse> {
  const url = `/api/enrichment/config?path=${encodeURIComponent(corpusPath)}`
  const res = await fetchWithTimeout(url, { method: 'GET' }, { timeoutMs: TIMEOUT_MS })
  if (!res.ok) {
    throw new Error((await readApiErrorMessage(res)) || 'GET /api/enrichment/config failed')
  }
  return (await res.json()) as EnrichmentConfigResponse
}

export async function putEnrichmentConfig(
  corpusPath: string,
  enrichmentBlock: Record<string, unknown>,
): Promise<EnrichmentConfigResponse> {
  const url = `/api/enrichment/config?path=${encodeURIComponent(corpusPath)}`
  const res = await fetchWithTimeout(
    url,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enrichment_block: enrichmentBlock }),
    },
    { timeoutMs: TIMEOUT_MS },
  )
  if (!res.ok) {
    throw new Error((await readApiErrorMessage(res)) || 'PUT /api/enrichment/config failed')
  }
  return (await res.json()) as EnrichmentConfigResponse
}

export async function getEnrichmentConfigSchema(): Promise<JsonSchemaFragment> {
  const res = await fetchWithTimeout(
    '/api/enrichment/config/schema',
    { method: 'GET' },
    { timeoutMs: TIMEOUT_MS },
  )
  if (!res.ok) {
    throw new Error((await readApiErrorMessage(res)) || 'GET /api/enrichment/config/schema failed')
  }
  return (await res.json()) as JsonSchemaFragment
}

export async function getEnrichmentProviderTypes(): Promise<ProviderTypesResponse> {
  const res = await fetchWithTimeout(
    '/api/enrichment/provider-types',
    { method: 'GET' },
    { timeoutMs: TIMEOUT_MS },
  )
  if (!res.ok) {
    throw new Error((await readApiErrorMessage(res)) || 'GET /api/enrichment/provider-types failed')
  }
  return (await res.json()) as ProviderTypesResponse
}
