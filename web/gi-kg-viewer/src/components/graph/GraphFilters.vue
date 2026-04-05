<script setup lang="ts">
import { computed } from 'vue'
import { graphNodeLegendLabel } from '../../utils/colors'
import { useGraphFilterStore } from '../../stores/graphFilters'

const gf = useGraphFilterStore()

const typeKeys = computed(() => {
  const st = gf.state
  if (!st) return []
  return Object.keys(st.allowedTypes).sort()
})

const kind = computed(() => gf.fullArtifact?.kind)

const showGrounded = computed(
  () => kind.value === 'gi' || kind.value === 'both',
)

const showLayerToggles = computed(() => kind.value === 'both')

const filterSummary = computed(() => {
  if (!gf.state) return ''
  const enabled = typeKeys.value.filter((t) => gf.state!.allowedTypes[t])
  const parts: string[] = []
  if (enabled.length < typeKeys.value.length) {
    parts.push(`${enabled.length}/${typeKeys.value.length} types`)
  }
  if (gf.state.hideUngroundedInsights) parts.push('grounded only')
  if (!gf.state.showGiLayer) parts.push('GI hidden')
  if (!gf.state.showKgLayer) parts.push('KG hidden')
  return parts.length ? parts.join(' · ') : 'all visible'
})

defineExpose({ filterSummary })
</script>

<template>
  <div
    v-if="gf.state"
    class="text-surface-foreground"
  >
    <!-- Entity types: horizontal fluid wrap -->
    <div class="mb-2">
      <div class="mb-1 flex items-center gap-2">
        <span class="text-[10px] font-semibold uppercase tracking-wide text-muted">Types</span>
        <button
          type="button"
          class="text-[10px] text-muted underline hover:text-surface-foreground"
          @click="gf.selectAllTypes()"
        >
          all
        </button>
        <button
          type="button"
          class="text-[10px] text-muted underline hover:text-surface-foreground"
          @click="gf.deselectAllTypes()"
        >
          none
        </button>
      </div>
      <div class="flex flex-wrap gap-x-3 gap-y-1">
        <label
          v-for="t in typeKeys"
          :key="t"
          class="flex cursor-pointer items-center gap-1 text-xs"
        >
          <input
            type="checkbox"
            class="rounded border-border"
            :checked="gf.state!.allowedTypes[t]"
            @change="gf.toggleAllowedType(t)"
          >
          <span>{{ graphNodeLegendLabel(t) }}</span>
        </label>
      </div>
    </div>

    <!-- Sources: layer + grounded -->
    <div>
      <span class="mb-1 block text-[10px] font-semibold uppercase tracking-wide text-muted">Sources</span>
      <div class="flex flex-wrap items-center gap-x-4 gap-y-1">
      <template v-if="showLayerToggles">
        <label class="flex cursor-pointer items-center gap-1 text-xs">
          <input
            type="checkbox"
            class="rounded border-border"
            :checked="gf.state!.showGiLayer"
            @change="gf.setShowGiLayer(!gf.state!.showGiLayer)"
          >
          <span>GI</span>
        </label>
        <label class="flex cursor-pointer items-center gap-1 text-xs">
          <input
            type="checkbox"
            class="rounded border-border"
            :checked="gf.state!.showKgLayer"
            @change="gf.setShowKgLayer(!gf.state!.showKgLayer)"
          >
          <span>KG</span>
        </label>
      </template>
      <label
        v-if="showGrounded"
        class="flex cursor-pointer items-center gap-1 text-xs"
      >
        <input
          type="checkbox"
          class="rounded border-border"
          :checked="gf.state!.hideUngroundedInsights"
          @change="gf.setHideUngrounded(!gf.state!.hideUngroundedInsights)"
        >
        <span>Hide ungrounded</span>
      </label>
      <span
        v-if="gf.filtersAreActive"
        class="text-[10px] font-medium text-warning"
      >
        filters active
      </span>
      </div>
    </div>
  </div>
</template>
