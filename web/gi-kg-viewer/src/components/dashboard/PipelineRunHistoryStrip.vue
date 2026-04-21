<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { CorpusRunSummaryItem } from '../../api/corpusMetricsApi'
import { formatDashboardRunDurationSeconds } from '../../utils/formatDuration'

const props = withDefaults(
  defineProps<{
    runs: CorpusRunSummaryItem[]
    /** When true, omit outer card chrome and heading (parent provides tab + border). */
    embedded?: boolean
  }>(),
  { embedded: false },
)

const selectedIdx = ref(-1)

const strip = computed(() => {
  const chrono = [...props.runs].sort((a, b) => (a.created_at ?? '').localeCompare(b.created_at ?? ''))
  return chrono.slice(-10)
})

watch(
  strip,
  (s) => {
    if (s.length) {
      selectedIdx.value = s.length - 1
    } else {
      selectedIdx.value = -1
    }
  },
  { immediate: true },
)

function dotClass(r: CorpusRunSummaryItem): string {
  const failed = r.episode_outcomes?.failed ?? 0
  const ok = r.episode_outcomes?.ok ?? 0
  const skipped = r.episode_outcomes?.skipped ?? 0
  const tot = failed + ok + skipped
  if (tot > 0 && failed > 0 && ok + skipped > 0) {
    return 'text-warning'
  }
  if (failed > 0 || (tot === 0 && (r.errors_total ?? 0) > 0)) {
    return 'text-danger'
  }
  return 'text-success'
}

function toggle(i: number): void {
  selectedIdx.value = selectedIdx.value === i ? -1 : i
}

const selectedRun = computed(() => {
  const list = strip.value
  const i = selectedIdx.value
  if (i < 0 || i >= list.length) {
    return null
  }
  return list[i] ?? null
})

const insight = computed(() => {
  const list = strip.value
  if (list.length === 0) {
    return undefined
  }
  let ok = 0
  let partial = 0
  let failed = 0
  for (const r of list) {
    const f = r.episode_outcomes?.failed ?? 0
    const o = r.episode_outcomes?.ok ?? 0
    const s = r.episode_outcomes?.skipped ?? 0
    const t = f + o + s
    if (t > 0 && f > 0 && o + s > 0) {
      partial += 1
    } else if (f > 0) {
      failed += 1
    } else {
      ok += 1
    }
  }
  if (partial === 0 && failed === 0) {
    return `All ${list.length} runs succeeded`
  }
  return `Last ${list.length} runs: ${ok} success, ${partial} partial, ${failed} failed`
})
</script>

<template>
  <div
    :class="
      embedded
        ? 'text-surface-foreground'
        : 'rounded border border-border bg-surface p-3 text-surface-foreground'
    "
    data-testid="pipeline-run-history-strip"
  >
    <h3
      v-if="!embedded"
      class="mb-2 text-sm font-semibold"
    >
      Run history
    </h3>
    <div class="flex flex-wrap items-center gap-1">
      <button
        v-for="(r, i) in strip"
        :key="r.relative_path + String(i)"
        type="button"
        :title="`${r.created_at ?? ''} · ${(r.episodes_scraped_total ?? 0)} episodes`"
        :class="[
          dotClass(r),
          i === strip.length - 1 ? 'text-[12px] leading-none' : 'text-[10px] leading-none',
          selectedIdx === i ? 'ring-1 ring-primary ring-offset-1 rounded-full' : '',
        ]"
        data-testid="pipeline-run-dot"
        @click="toggle(i)"
      >
        ●
      </button>
    </div>
    <div
      v-if="selectedRun"
      class="mt-2 rounded border border-border bg-overlay p-2 text-[10px] text-muted"
    >
      <p><span class="text-surface-foreground">Path:</span> {{ selectedRun.relative_path }}</p>
      <p><span class="text-surface-foreground">Episodes:</span> {{ (selectedRun.episodes_scraped_total ?? 0).toLocaleString() }}</p>
      <p>
        <span class="text-surface-foreground">Duration:</span>
        {{ formatDashboardRunDurationSeconds(selectedRun.run_duration_seconds ?? null) || '—' }}
      </p>
    </div>
    <p
      v-if="insight"
      class="mt-2 text-[11px] text-muted"
    >
      {{ insight }}
    </p>
  </div>
</template>
