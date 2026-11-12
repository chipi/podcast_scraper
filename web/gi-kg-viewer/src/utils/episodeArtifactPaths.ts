/**
 * The artifact files to load for one episode, in one place: the viewer's own (GI, then KG), then
 * what extensions add (`episodeArtifactPaths`). Every "open this episode in the graph" path builds
 * its list here, so an extension's files follow it everywhere.
 */
import { episodeArtifactPaths } from '../extensions'

/** The fields of an episode payload the viewer reads; the rest go to extensions untouched. */
export interface EpisodeArtifactPayload {
  has_gi?: boolean | null
  gi_relative_path?: string | null
  has_kg?: boolean | null
  kg_relative_path?: string | null
}

function unique(paths: string[]): string[] {
  return [...new Set(paths.map((p) => p.trim()).filter(Boolean))]
}

/** A payload path counts when its `has_*` flag is true, or, with no such key, when it is set. */
function counts(
  episode: EpisodeArtifactPayload,
  flag: 'has_gi' | 'has_kg',
  path: string | null | undefined,
): path is string {
  if (!path) return false
  return flag in episode ? Boolean(episode[flag]) : true
}

/** From an episode payload (node-episodes rows carry the paths without the flags). */
export function episodeArtifactRelPaths(episode: EpisodeArtifactPayload): string[] {
  const own: string[] = []
  if (counts(episode, 'has_kg', episode.kg_relative_path)) own.push(episode.kg_relative_path)
  const extra = episodeArtifactPaths
    .list()
    .flatMap((x) => x.paths?.(episode as Record<string, unknown>) ?? [])
  return unique([...own, ...extra])
}

/** From a metadata file's stem. */
export function episodeArtifactRelPathsForStem(stem: string): string[] {
  const own = [`${stem}.kg.json`]
  const extra = episodeArtifactPaths.list().flatMap((x) => x.forStem?.(stem) ?? [])
  return unique([...own, ...extra])
}
