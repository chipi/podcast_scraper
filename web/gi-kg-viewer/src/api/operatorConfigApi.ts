import { fetchWithTimeout } from './httpClient'
import { readApiErrorMessage } from './readApiErrorMessage'

export interface OperatorConfigPayload {
  corpus_path: string
  operator_config_path: string
  content: string
  /** Packaged preset names for the operator-config profile picker. */
  available_profiles?: string[]
}

export async function getOperatorConfig(corpusPath: string): Promise<OperatorConfigPayload> {
  const q = new URLSearchParams({ path: corpusPath.trim() })
  const res = await fetchWithTimeout(`/api/operator-config?${q}`, undefined, {
    timeoutDetail: 'operator-config',
  })
  if (!res.ok) {
    throw new Error(await readApiErrorMessage(res))
  }
  return (await res.json()) as OperatorConfigPayload
}

export async function putOperatorConfig(
  corpusPath: string,
  content: string,
): Promise<OperatorConfigPayload> {
  const q = new URLSearchParams({ path: corpusPath.trim() })
  const res = await fetchWithTimeout(
    `/api/operator-config?${q}`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content }),
    },
    { timeoutDetail: 'operator-config' },
  )
  if (!res.ok) {
    throw new Error(await readApiErrorMessage(res))
  }
  return (await res.json()) as OperatorConfigPayload
}
