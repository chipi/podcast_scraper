<script setup lang="ts">
import { computed } from 'vue'

import type { CorpusRunSummaryItem } from '../../api/corpusMetricsApi'
import DiagnosticRow from '../shared/DiagnosticRow.vue'

/**
 * #656 Stage D: pre-extraction ad-region excision metrics panel.
 *
 * Surfaces the three #663 run totals — pre-roll chars cut, post-roll
 * chars cut, and the episode count that had any excision — from the
 * latest run's ``metrics.json``. Companion to PipelineCleanupMetrics:
 * that panel covers *post-extraction* filters; this one covers the
 * *pre-extraction* sponsor-block cut that runs before GI/KG see the
 * transcript.
 *
 * Display rules match PipelineCleanupMetrics:
 *   - ``null`` (legacy run, pre-#663) → ``—``; row exists for layout
 *     stability.
 *   - ``0`` → ``0``; the excision ran but found nothing to cut.
 *   - ``> 0`` → localised count with an info badge.
 */

const props = defineProps<{
  run: CorpusRunSummaryItem | null
}>()

interface ExcisionMetric {
  label: string
  value: number | null
  tooltip: string
  badge: string
}

const rows = computed<ExcisionMetric[]>(() => [
  {
    label: 'Pre-roll chars cut',
    value: props.run?.ad_chars_excised_preroll ?? null,
    tooltip:
      'Total characters cut from pre-rolls across all episodes in this run (sponsor reads before content).',
    badge: 'preroll',
  },
  {
    label: 'Post-roll chars cut',
    value: props.run?.ad_chars_excised_postroll ?? null,
    tooltip:
      'Total characters cut from post-rolls across all episodes in this run (credits / outro ads).',
    badge: 'postroll',
  },
  {
    label: 'Episodes with excision',
    value: props.run?.ad_episodes_with_excision_count ?? null,
    tooltip:
      'Episodes where the pre-extraction detector cut at least one ad region.',
    badge: 'episodes',
  },
])

function formatValue(v: number | null): string {
  return v == null ? '—' : v.toLocaleString('en-US')
}

function badgeKind(v: number | null): 'default' | 'info' {
  return typeof v === 'number' && v > 0 ? 'info' : 'default'
}

const anyActive = computed(() =>
  rows.value.some((r) => typeof r.value === 'number' && r.value > 0),
)
const allMissing = computed(() => rows.value.every((r) => r.value == null))
</script>

<template>
  <section
    class="rounded border border-border bg-surface p-3 text-surface-foreground"
    data-testid="pipeline-ad-excision-metrics"
    aria-labelledby="pipeline-ad-excision-metrics-heading"
  >
    <div class="mb-2 flex items-baseline justify-between">
      <h3
        id="pipeline-ad-excision-metrics-heading"
        class="text-sm font-semibold"
      >
        Ad-region excision
      </h3>
      <p class="text-[10px] text-muted">
        Latest run — pre-extraction
      </p>
    </div>
    <p
      v-if="!run"
      class="text-xs text-muted"
    >
      No run metrics available yet for this corpus.
    </p>
    <template v-else>
      <p
        v-if="allMissing"
        class="mb-2 text-[10px] text-muted"
      >
        This run predates the #663 ad-excision counters — no excision
        data to show.
      </p>
      <dl class="space-y-0">
        <DiagnosticRow
          v-for="row in rows"
          :key="row.label"
          :label="row.label"
          :value="formatValue(row.value)"
          :kind="badgeKind(row.value)"
          :badge="row.badge"
          :tooltip="row.tooltip"
          :data-testid="`pipeline-ad-excision-${row.badge}`"
        />
      </dl>
      <p
        v-if="!anyActive && !allMissing"
        class="mt-2 text-[10px] text-muted"
      >
        No ad regions were excised on this run.
      </p>
    </template>
  </section>
</template>
