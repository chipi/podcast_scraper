/**
 * Recent activity store (#1259-3).
 *
 * Extends the Recent-queries pattern to cover TWO other user-visible
 * activity streams so the palette empty state (and future LeftPanel
 * surfaces) can offer "I was just looking at that…" jumps without a
 * fresh search:
 *
 *   - ``ui.recentSubjects``  — episodes / topics / people the user
 *     opened in the right rail (subject store transitions).
 *   - ``ui.recentHandoffs``  — cross-tab handoffs
 *     ("Library → Graph via topic:foo", "Search → Rail via
 *     person:bar", …).
 *
 * Both are ring buffers capped at 20, newest first, deduped on the
 * natural key ("kind + id" for subjects, the whole envelope for
 * handoffs). Persisted through the shipped USERPREFS-1 store
 * (``useUserPreferencesStore``) — no new server endpoints.
 *
 * Thin Pinia wrapper on top of ``useUserPreferencesStore`` — mirrors
 * ``useSavedQueriesStore`` so downstream surfaces can consume both
 * with the same shape.
 */

import { computed } from 'vue'
import { defineStore } from 'pinia'
import { useUserPreferencesStore } from './userPreferences'

export type RecentSubjectKind = 'episode' | 'topic' | 'person'

export interface RecentSubjectEntry {
  kind: RecentSubjectKind
  id: string
  label?: string
  ts?: number
}

export interface RecentHandoffEntry {
  /**
   * Short human-readable trail (e.g. "Library → Graph"). Consumers
   * render it verbatim so the store stays UI-agnostic.
   */
  trail: string
  /** Payload the caller wants preserved (subject id, metadata path). */
  target: string
  /** Optional label for the target (episode title, topic label, …). */
  label?: string
  ts?: number
}

const RECENT_SUBJECTS_KEY = 'ui.recentSubjects'
const RECENT_HANDOFFS_KEY = 'ui.recentHandoffs'
const RECENT_MAX = 20

export const useRecentActivityStore = defineStore('recentActivity', () => {
  const userPrefs = useUserPreferencesStore()

  const recentSubjects = computed<RecentSubjectEntry[]>(() => {
    const raw = userPrefs.get<RecentSubjectEntry[]>(RECENT_SUBJECTS_KEY)
    return Array.isArray(raw) ? raw : []
  })

  const recentHandoffs = computed<RecentHandoffEntry[]>(() => {
    const raw = userPrefs.get<RecentHandoffEntry[]>(RECENT_HANDOFFS_KEY)
    return Array.isArray(raw) ? raw : []
  })

  function listRecentSubjects(limit?: number): RecentSubjectEntry[] {
    if (limit == null) return recentSubjects.value
    return recentSubjects.value.slice(0, Math.max(0, limit))
  }

  function listRecentHandoffs(limit?: number): RecentHandoffEntry[] {
    if (limit == null) return recentHandoffs.value
    return recentHandoffs.value.slice(0, Math.max(0, limit))
  }

  /**
   * Push a subject onto the ring. De-dupes on ``kind + id`` — repeated
   * visits move to the front instead of stacking. Silent no-op on
   * blank id (protects against uninitialised subject-store watchers).
   */
  async function pushSubject(entry: RecentSubjectEntry): Promise<void> {
    const id = entry.id?.trim() ?? ''
    if (!id) return
    const label = entry.label?.trim() || undefined
    const clean: RecentSubjectEntry = {
      kind: entry.kind,
      id,
      label,
      ts: Date.now(),
    }
    const existing = listRecentSubjects()
    const filtered = existing.filter((e) => !(e.kind === clean.kind && e.id === clean.id))
    const next = [clean, ...filtered].slice(0, RECENT_MAX)
    await userPrefs.set(RECENT_SUBJECTS_KEY, next)
  }

  /**
   * Push a handoff envelope onto the ring. Dedupe key is
   * ``trail + target`` so re-doing the same jump doesn't duplicate.
   */
  async function pushHandoff(entry: RecentHandoffEntry): Promise<void> {
    const trail = entry.trail?.trim() ?? ''
    const target = entry.target?.trim() ?? ''
    if (!trail || !target) return
    const label = entry.label?.trim() || undefined
    const clean: RecentHandoffEntry = {
      trail,
      target,
      label,
      ts: Date.now(),
    }
    const existing = listRecentHandoffs()
    const filtered = existing.filter(
      (e) => !(e.trail === clean.trail && e.target === clean.target),
    )
    const next = [clean, ...filtered].slice(0, RECENT_MAX)
    await userPrefs.set(RECENT_HANDOFFS_KEY, next)
  }

  async function clearRecentSubjects(): Promise<void> {
    await userPrefs.set(RECENT_SUBJECTS_KEY, [])
  }

  async function clearRecentHandoffs(): Promise<void> {
    await userPrefs.set(RECENT_HANDOFFS_KEY, [])
  }

  return {
    recentSubjects,
    recentHandoffs,
    listRecentSubjects,
    listRecentHandoffs,
    pushSubject,
    pushHandoff,
    clearRecentSubjects,
    clearRecentHandoffs,
  }
})
