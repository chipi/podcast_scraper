/**
 * Favorites store (Pinia ↔ GET/PUT/DELETE /api/app/favorites) — the polymorphic "saved things"
 * the user collects (episodes, insights, … later people/topics). Mirrors the queue store: auth-gated
 * (empty + no-op signed out), every mutation persists and refreshes from the server response.
 */
import { defineStore } from 'pinia'
import { addFavorite, getFavorites, removeFavorite } from '../services/api'
import { readCached, writeCached } from '../services/contentCache'
import type { EpisodeSummary, FavoriteAdd, FavoriteInsight } from '../services/types'

interface FavoritesState {
  episodes: EpisodeSummary[]
  insights: FavoriteInsight[]
  loaded: boolean
  /** Showing a cached copy not yet revalidated (#1909). */
  stale: boolean
}

export const useFavoritesStore = defineStore('favorites', {
  state: (): FavoritesState => ({ episodes: [], insights: [], loaded: false, stale: false }),
  getters: {
    /** Whether a given item is saved (drives the heart toggle state). */
    has:
      (s) =>
      (kind: string, ref: string): boolean =>
        kind === 'episode'
          ? s.episodes.some((e) => e.slug === ref)
          : kind === 'insight'
            ? s.insights.some((i) => i.ref === ref)
            : false,
    count: (s): number => s.episodes.length + s.insights.length,
  },
  actions: {
    /** Revalidate, falling back to the cached copy when the request never lands (#1909). */
    async load(): Promise<void> {
      try {
        const f = await getFavorites()
        this.episodes = f.episodes
        this.insights = f.insights
        this.loaded = true
        this.stale = false
        void writeCached('favorites', { episodes: f.episodes, insights: f.insights })
      } catch {
        const cached = await readCached<Pick<FavoritesState, 'episodes' | 'insights'>>('favorites')
        if (cached) {
          this.episodes = cached.episodes
          this.insights = cached.insights
          this.loaded = true
          this.stale = true
        }
      }
    },
    async ensureLoaded(): Promise<void> {
      if (!this.loaded) await this.load()
    },
    /** Toggle a favorite; the server response is authoritative (no optimistic drift). */
    async toggle(item: FavoriteAdd): Promise<void> {
      try {
        const f = this.has(item.kind, item.ref)
          ? await removeFavorite(item.kind, item.ref)
          : await addFavorite(item)
        this.episodes = f.episodes
        this.insights = f.insights
        this.loaded = true
      } catch {
        /* signed out / transient — leave state; next load reconciles with the server */
      }
    },
  },
})
