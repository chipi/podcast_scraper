import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'
import type { GraphFilterState, ParsedArtifact } from '../types/artifact'
import { useArtifactsStore } from './artifacts'
import {
  applyGraphFilters,
  defaultFilterState,
  filterArtifactEgoOneHop,
  filtersActive,
} from '../utils/parsing'
import { semanticTypeForLegendVisual } from '../utils/colors'

export const useGraphFilterStore = defineStore('graphFilters', () => {
  const artifacts = useArtifactsStore()
  const state = ref<GraphFilterState | null>(null)

  watch(
    () => artifacts.displayArtifact,
    (art) => {
      state.value = art ? defaultFilterState(art) : null
    },
    { immediate: true },
  )

  const fullArtifact = computed(() => artifacts.displayArtifact)

  const filteredArtifact = computed(() => {
    const full = fullArtifact.value
    const st = state.value
    if (!full || !st) return null
    return applyGraphFilters(full, st)
  })

  const filtersAreActive = computed(() =>
    filtersActive(fullArtifact.value, state.value),
  )

  function viewWithEgo(focusId: string | null): ParsedArtifact | null {
    const base = filteredArtifact.value
    if (!base) return null
    return filterArtifactEgoOneHop(base, focusId)
  }

  function setHideUngrounded(v: boolean): void {
    if (state.value) state.value.hideUngroundedInsights = v
  }

  function setShowGiLayer(v: boolean): void {
    if (state.value) state.value.showGiLayer = v
  }

  function setShowKgLayer(v: boolean): void {
    if (state.value) state.value.showKgLayer = v
  }

  function toggleAllowedType(t: string): void {
    if (!state.value) return
    if (!(t in state.value.allowedTypes)) return
    const cur = state.value.allowedTypes[t]
    state.value.allowedTypes = {
      ...state.value.allowedTypes,
      [t]: !cur,
    }
  }

  function selectAllTypes(): void {
    if (!state.value) return
    const next: Record<string, boolean> = { ...state.value.allowedTypes }
    for (const k of Object.keys(next)) {
      next[k] = true
    }
    state.value.allowedTypes = next
    state.value.legendSoloVisual = null
  }

  function deselectAllTypes(): void {
    if (!state.value) return
    const next: Record<string, boolean> = { ...state.value.allowedTypes }
    for (const k of Object.keys(next)) {
      next[k] = false
    }
    state.value.allowedTypes = next
    state.value.legendSoloVisual = null
  }

  function resetLegendSolo(): void {
    if (!state.value || !fullArtifact.value) return
    const fresh = defaultFilterState(fullArtifact.value)
    if (fresh) {
      state.value.allowedTypes = { ...fresh.allowedTypes }
      state.value.hideUngroundedInsights = fresh.hideUngroundedInsights
      state.value.legendSoloVisual = null
      state.value.showGiLayer = fresh.showGiLayer
      state.value.showKgLayer = fresh.showKgLayer
    }
  }

  /** Match v1 graph-legend + graph-cyto: solo one visual group or reset. */
  function onLegendClick(visualKey: string): void {
    if (!state.value || !fullArtifact.value) return
    const st = state.value
    if (visualKey === '__reset__' || st.legendSoloVisual === visualKey) {
      resetLegendSolo()
      return
    }
    const sem = semanticTypeForLegendVisual(visualKey)
    if (!(sem in st.allowedTypes)) return
    st.legendSoloVisual = visualKey
    const keys = Object.keys(st.allowedTypes)
    for (const t of keys) {
      st.allowedTypes[t] = t === sem
    }
  }

  return {
    state,
    fullArtifact,
    filteredArtifact,
    filtersAreActive,
    viewWithEgo,
    setHideUngrounded,
    setShowGiLayer,
    setShowKgLayer,
    toggleAllowedType,
    selectAllTypes,
    deselectAllTypes,
    onLegendClick,
    resetLegendSolo,
  }
})
