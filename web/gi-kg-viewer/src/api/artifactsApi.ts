import type { ArtifactData } from '../types/artifact'

function encodeArtifactPath(rel: string): string {
  return rel.split('/').map(encodeURIComponent).join('/')
}

export async function fetchArtifactJson(
  corpusPath: string,
  relativePath: string,
): Promise<ArtifactData> {
  const q = new URLSearchParams({ path: corpusPath.trim() })
  const pathSeg = encodeArtifactPath(relativePath)
  const res = await fetch(`/api/artifacts/${pathSeg}?${q.toString()}`)
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t || `HTTP ${res.status}`)
  }
  return (await res.json()) as ArtifactData
}
