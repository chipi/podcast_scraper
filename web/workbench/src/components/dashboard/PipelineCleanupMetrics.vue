<script setup lang="ts">
import { computed } from 'vue'

import type { CorpusRunSummaryItem } from '../../api/corpusMetricsApi'
import DiagnosticRow from '../shared/DiagnosticRow.vue'

/**
 * #656 Stage B: Pipeline cleanup metrics panel.
 *
 * Surfaces the four #652 Part B post-extraction filter counters from
 * the latest run's ``metrics.json``. The backend now returns them on
 * ``CorpusRunSummaryItem`` (``null`` when the run predates #652).
 *
 * Rendering:
 *   - ``null`` (legacy run) renders as ``—`` — operators see the row
 *     exists but nothing fired.
 *   - ``0`` renders as ``0`` — a real reading that the filter ran and
 *     matched nothing. Useful signal (e.g. a well-tuned extractor that
 *     produces clean insights on this corpus).
 *   - ``> 0`` renders the count with a subtle info badge so big
 *     cleanup numbers pop.
 *
 * All four rows render even on legacy runs so the layout stays stable
 * across corpora at different #652 adoption states.
 */

const props = defineProps<{
  run: CorpusRunSummaryItem | null
}>()

interface CleanupMetric {
  label: string
  value: number | null
  tooltip: string
  /** Human-readable noun for the badge ("ads", "dialogue", "topics", "entities"). */
  badge: string
}

const rows = computed<CleanupMetric[]>(() => [
  {
    label: 'Ads filtered',
    value: props.run?.ads_filtered_count ?? null,
    tooltip:
      'Insights dropped by the #652 ad filter (≥ 2 spoken-form ad patterns in the quote window).',
    badge: 'ads',
  },
  {
    label: 'Dialogue dropped',
    value: props.run?.dialogue_insights_dropped_count ?? null,
    tooltip:
      'Insights rejected for starting with filler ("yeah", "well"), first-person pronoun density, or heavy quote coverage.',
    badge: 'dialogue',
  },
  {
    label: 'Topics normalized',
    value: props.run?.topics_normalized_count ?? null,
    tooltip:
      'KG topic labels simplified (leading/medial stopwords stripped, tokens capped, near-duplicates merged).',
    badge: 'topics',
  },
  {
    label: 'Entity kinds repaired',
    value: props.run?.entity_kinds_repaired_count ?? null,
    tooltip:
      'Entities whose kind was overridden by the curated KNOWN_ORGS whitelist (e.g. ``Planet Money`` → org, not person).',
    badge: 'entities',
  },
])

function formatValue(v: number | null): string {
  return v == null ? '—' : v.toLocaleString('en-US')
}

function badgeKind(v: number | null): 'default' | 'info' {
  // Only render the status chip when something actually happened —
  // plain "0" or "—" rows stay quiet.
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
    data-testid="pipeline-cleanup-metrics"
    aria-labelledby="pipeline-cleanup-metrics-heading"
  >
    <div class="mb-2 flex items-baseline justify-between">
      <h3
        id="pipeline-cleanup-metrics-heading"
        class="text-sm font-semibold"
      >
        Pipeline cleanup
      </h3>
      <p class="text-[10px] text-muted">
        Latest run
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
        This run predates the #652 post-extraction filter counters — no
        cleanup data to show.
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
          :data-testid="`pipeline-cleanup-${row.badge}`"
        />
      </dl>
      <p
        v-if="!anyActive && !allMissing"
        class="mt-2 text-[10px] text-muted"
      >
        No cleanup filters fired on this run.
      </p>
    </template>
  </section>
</template>
