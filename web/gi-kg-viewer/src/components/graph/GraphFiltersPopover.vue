<script setup lang="ts">
import { storeToRefs } from 'pinia'
import { computed, nextTick, onUnmounted, ref, watch } from 'vue'
import { useGraphExplorerStore } from '../../stores/graphExplorer'
import { useGraphFilterStore } from '../../stores/graphFilters'
import { filtersActiveExcludingNodeTypes } from '../../utils/parsing'
import { DEGREE_BUCKET_ORDER } from '../../utils/graphDegreeBuckets'

const props = defineProps<{
  degreeHistogramCounts: Record<string, number>
}>()

const gf = useGraphFilterStore()
const ge = useGraphExplorerStore()
const { activeDegreeBucket } = storeToRefs(ge)

const open = ref(false)
const anchorRef = ref<HTMLButtonElement | null>(null)
const panelRef = ref<HTMLDivElement | null>(null)

const graphKind = computed(() => gf.fullArtifact?.kind)
const showLayerToggles = computed(() => graphKind.value === 'both')
const showGroundedFilter = computed(
  () => graphKind.value === 'gi' || graphKind.value === 'both',
)

const edgeTypeKeys = computed(() => {
  const aet = gf.state?.allowedEdgeTypes
  if (!aet) return [] as string[]
  return Object.keys(aet).sort()
})

const showWarningDot = computed(() => {
  const st = gf.state
  if (filtersActiveExcludingNodeTypes(gf.fullArtifact, st)) return true
  return activeDegreeBucket.value != null && activeDegreeBucket.value !== ''
})

function close(): void {
  open.value = false
}

function toggle(): void {
  open.value = !open.value
}

function onDocPointerDown(ev: MouseEvent | PointerEvent): void {
  if (!open.value) return
  const t = ev.target
  if (!(t instanceof Node)) return
  if (anchorRef.value?.contains(t)) return
  if (panelRef.value?.contains(t)) return
  close()
}

function onDocKeydown(ev: KeyboardEvent): void {
  if (!open.value || ev.key !== 'Escape') return
  ev.preventDefault()
  close()
  anchorRef.value?.focus()
}

watch(open, async (v) => {
  if (!v) {
    document.removeEventListener('pointerdown', onDocPointerDown, true)
    document.removeEventListener('keydown', onDocKeydown, true)
    return
  }
  await nextTick()
  document.addEventListener('pointerdown', onDocPointerDown, true)
  document.addEventListener('keydown', onDocKeydown, true)
})

onUnmounted(() => {
  document.removeEventListener('pointerdown', onDocPointerDown, true)
  document.removeEventListener('keydown', onDocKeydown, true)
})

function deselectAllEdgeTypes(): void {
  const st = gf.state
  if (!st) return
  for (const k of Object.keys(st.allowedEdgeTypes)) {
    if (st.allowedEdgeTypes[k]) {
      gf.toggleAllowedEdgeType(k)
    }
  }
}
</script>

<template>
  <div class="relative inline-flex shrink-0 items-center self-center">
    <button
      ref="anchorRef"
      type="button"
      class="relative inline-flex h-5 min-w-[1.25rem] items-center justify-center rounded border border-border bg-surface px-1 text-[11px] leading-none text-muted hover:bg-overlay/40 hover:text-surface-foreground"
      data-testid="graph-toolbar-more-filters"
      :aria-expanded="open"
      aria-haspopup="dialog"
      aria-label="More graph filters"
      @click="toggle"
    >
      ⚙
      <span
        v-if="showWarningDot"
        class="pointer-events-none absolute right-0 top-0 h-1.5 w-1.5 translate-x-px -translate-y-px rounded-full bg-warning"
        aria-hidden="true"
      />
    </button>
    <div
      v-show="open"
      ref="panelRef"
      role="dialog"
      aria-label="Graph filters"
      data-testid="graph-filters-popover"
      class="absolute right-0 top-full z-[40] mt-1 w-56 rounded-sm border border-border bg-elevated p-3 shadow-md"
    >
      <div
        v-if="gf.state && (showLayerToggles || showGroundedFilter)"
        class="space-y-1.5"
        data-testid="graph-filters-sources"
      >
        <div class="text-[10px] font-semibold uppercase tracking-wider text-muted">
          Sources
        </div>
        <template v-if="showLayerToggles">
          <label class="flex cursor-pointer items-center gap-1.5 text-[10px] text-surface-foreground">
            <input
              type="checkbox"
              class="rounded border-border"
              :checked="gf.state.showGiLayer"
              @change="gf.setShowGiLayer(!gf.state.showGiLayer)"
            >
            <span>GI</span>
          </label>
          <label class="flex cursor-pointer items-center gap-1.5 text-[10px] text-surface-foreground">
            <input
              type="checkbox"
              class="rounded border-border"
              :checked="gf.state.showKgLayer"
              @change="gf.setShowKgLayer(!gf.state.showKgLayer)"
            >
            <span>KG</span>
          </label>
        </template>
        <label
          v-if="gf.state && showGroundedFilter"
          class="flex cursor-pointer items-center gap-1.5 text-[10px] text-surface-foreground"
        >
          <input
            type="checkbox"
            class="rounded border-border"
            :checked="gf.state.hideUngroundedInsights"
            @change="gf.setHideUngrounded(!gf.state.hideUngroundedInsights)"
          >
          <span>Hide ungrounded</span>
        </label>
      </div>

      <div
        v-if="edgeTypeKeys.length && gf.state"
        class="mt-2 border-t border-border pt-2"
        data-testid="graph-filters-edges"
      >
        <div class="text-[10px] font-semibold uppercase tracking-wider text-muted">
          Edges
        </div>
        <div class="mt-1 flex flex-wrap items-center gap-x-2 gap-y-0.5">
          <button
            type="button"
            class="text-[10px] text-primary underline"
            @click="gf.selectAllEdgeTypes()"
          >
            all
          </button>
          <button type="button" class="text-[10px] text-primary underline" @click="deselectAllEdgeTypes">
            none
          </button>
        </div>
        <label
          v-for="et in edgeTypeKeys"
          :key="et"
          class="mt-0.5 flex cursor-pointer items-center gap-1.5 text-[10px] text-surface-foreground"
        >
          <input
            type="checkbox"
            class="rounded border-border"
            :checked="gf.state!.allowedEdgeTypes[et]"
            @change="gf.toggleAllowedEdgeType(et)"
          >
          <span class="max-w-[10rem] truncate" :title="et">{{ et }}</span>
        </label>
      </div>

      <div class="mt-2 border-t border-border pt-2" data-testid="graph-filters-degree">
        <div class="text-[10px] font-semibold uppercase tracking-wider text-muted">
          Degree
        </div>
        <div class="mt-1 grid grid-cols-2 gap-0.5">
          <button
            v-for="bid in DEGREE_BUCKET_ORDER"
            :key="bid"
            type="button"
            class="rounded border px-0.5 py-px text-[10px] leading-tight hover:bg-overlay"
            :class="
              activeDegreeBucket === bid
                ? 'border-primary bg-primary/15 font-medium'
                : 'border-border'
            "
            :aria-pressed="activeDegreeBucket === bid"
            @click="ge.toggleDegreeBucket(bid)"
          >
            {{ bid }}
            <span class="text-muted">({{ props.degreeHistogramCounts[bid] ?? 0 }})</span>
          </button>
        </div>
        <button
          v-if="activeDegreeBucket"
          type="button"
          class="mt-1 w-full rounded border border-border px-0.5 py-px text-[10px] leading-tight text-primary hover:bg-overlay"
          aria-label="Clear degree filter"
          @click="ge.clearDegreeBucket()"
        >
          Clear
        </button>
      </div>
    </div>
  </div>
</template>
