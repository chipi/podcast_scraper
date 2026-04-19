import { defineStore } from 'pinia'
import { ref } from 'vue'

export type DashboardLibraryHandoff = {
  kind: 'library'
  feedId?: string
  since?: string
  until?: string
  missingGiOnly?: boolean
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

  return { pending, setHandoff, consumeHandoff }
})
