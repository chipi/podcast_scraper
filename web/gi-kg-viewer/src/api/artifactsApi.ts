import type { ArtifactData } from '../types/artifact'
import { DEFAULT_VIEWER_FETCH_TIMEOUT_MS, fetchWithTimeout } from './httpClient'

/** Per-file artifact JSON timeout (alias for shared default). */
export const ARTIFACT_FETCH_TIMEOUT_MS = DEFAULT_VIEWER_FETCH_TIMEOUT_MS

function encodeArtifactPath(rel: string): string {
  return rel.split('/').map(encodeURIComponent).join('/')
}

export async function fetchArtifactJson(
  corpusPath: string,
  relativePath: string,
): Promise<ArtifactData> {
  const q = new URLSearchParams({ path: corpusPath.trim() })
  const pathSeg = encodeArtifactPath(relativePath)
  const url = `/api/artifacts/${pathSeg}?${q.toString()}`
  const res = await fetchWithTimeout(url, undefined, {
    timeoutMs: ARTIFACT_FETCH_TIMEOUT_MS,
    timeoutDetail: relativePath,
  })
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t || `HTTP ${res.status}`)
  }
  return (await res.json()) as ArtifactData
}
