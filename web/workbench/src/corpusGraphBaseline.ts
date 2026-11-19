import type { InjectionKey } from 'vue'

/**
 * Switches to the Graph tab and runs the same default merged-graph load as a first visit
 * (corpus artifact list + episode cap), **before** callers append episode-specific GI/KG paths.
 * Provided by ``App.vue``; absent in isolated mounts (tests).
 */
export type CorpusGraphBaselineLoader = () => Promise<void>

export const corpusGraphBaselineLoaderKey: InjectionKey<CorpusGraphBaselineLoader> = Symbol(
  'corpusGraphBaselineLoader',
)
