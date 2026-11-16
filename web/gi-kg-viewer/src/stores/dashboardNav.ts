import { defineStore } from 'pinia'
import { ref } from 'vue'

export type DashboardLibraryHandoff = {
  kind: 'library'
  feedId?: string
  since?: string
  until?: string
  /** Only episodes without a KG artifact (`has_kg=false`). */
  missingKgOnly?: boolean
  /** Extra `/api/corpus/episodes` parameters a build adds (e.g. `has_gi`), kept until the next handoff. */
  params?: Record<string, string>
}

export type DashboardDigestHandoff = {
  kind: 'digest'
}

export type DashboardGraphHandoff = {
  kind: 'graph'
}

export type DashboardTabHandoff = {
  kind: 'dashboard'
  tab: 'coverage' | 'intelligence' | 'pipeline'
}

export type DashboardHandoff =
  | DashboardLibraryHandoff
  | DashboardDigestHandoff
  | DashboardGraphHandoff
  | DashboardTabHandoff

export const useDashboardNavStore = defineStore('dashboardNav', () => {
  const pending = ref<DashboardHandoff | null>(null)

  function setHandoff(h: DashboardHandoff): void {
    pending.value = h
  }

  function consumeHandoff(): DashboardHandoff | null {
    const p = pending.value
    pending.value = null
    return p
  }

  /** Pop pending only when it is a Library handoff (digest/other kinds stay queued). */
  function consumeLibraryHandoffIfPending(): DashboardLibraryHandoff | null {
    const p = pending.value
    if (!p || p.kind !== 'library') {
      return null
    }
    pending.value = null
    return p
  }

  return { pending, setHandoff, consumeHandoff, consumeLibraryHandoffIfPending }
})
