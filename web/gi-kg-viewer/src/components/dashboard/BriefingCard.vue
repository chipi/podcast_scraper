<script setup lang="ts">
import { computed } from 'vue'
import type { CorpusCoverageResponse } from '../../api/corpusCoverageApi'
import type { CorpusRunSummaryItem } from '../../api/corpusMetricsApi'
import type { IndexStatsEnvelope } from '../../api/indexStatsApi'
import { formatDashboardRunDurationSeconds } from '../../utils/formatDuration'
import { formatRelativeRunAge } from '../../utils/dashboardTime'
import { useDashboardNavStore } from '../../stores/dashboardNav'

const props = defineProps<{
  runs: CorpusRunSummaryItem[]
  coverage: CorpusCoverageResponse | null
  indexEnv: IndexStatsEnvelope | null
  catalogFeedCount: number
  corpusPath: string
  apiReady: boolean
}>()

const emit = defineEmits<{
  'select-tab': [tab: 'coverage' | 'intelligence' | 'pipeline']
  /** Open Pipeline tab focused on corpus run strip (matches “Last run” context). */
  'open-pipeline-run-history': []
  'rebuild-index': []
  'open-library': []
}>()

const dashNav = useDashboardNavStore()

const GI_WARN = 0.5
const INDEX_WARN = 0.6

const latestRun = computed(() => props.runs[0] ?? null)

const runStatus = computed((): 'none' | 'success' | 'partial' | 'failed' => {
  const r = latestRun.value
  if (!r) {
    return 'none'
  }
  const failed = r.episode_outcomes?.failed ?? 0
  const ok = r.episode_outcomes?.ok ?? 0
  const skipped = r.episode_outcomes?.skipped ?? 0
  const totalOut = failed + ok + skipped
  const err = r.errors_total ?? 0
  if (totalOut > 0 && failed > 0 && ok + skipped > 0) {
    return 'partial'
  }
  if (failed > 0 || (totalOut === 0 && err > 0)) {
    return 'failed'
  }
  return 'success'
})

const totalEpisodes = computed(() => props.coverage?.total_episodes ?? 0)
const giRate = computed(() => {
  const t = totalEpisodes.value
  if (!t || !props.coverage) {
    return null
  }
  return props.coverage.with_gi / t
})
/** FAISS row count (many per episode) — do not divide by episode count as a “%”. */
const indexVectorCount = computed(() => props.indexEnv?.stats?.total_vectors ?? null)

const indexVectorsPerEpisode = computed(() => {
  const t = totalEpisodes.value
  const v = indexVectorCount.value
  if (!t || v == null || t <= 0) {
    return null
  }
  return v / t
})

const indexVectorsLabel = computed(() => {
  const v = indexVectorCount.value
  if (v == null || !props.indexEnv?.available) {
    return '—'
  }
  if (v >= 1_000_000) {
    return `${(v / 1_000_000).toFixed(1)}M vectors`
  }
  if (v >= 10_000) {
    return `${Math.round(v / 1000)}k vectors`
  }
  if (v >= 1000) {
    return `${(v / 1000).toFixed(1)}k vectors`
  }
  return `${v} vectors`
})

const indexVectorsTitle = computed(() => {
  const v = indexVectorCount.value
  const t = totalEpisodes.value
  const r = indexVectorsPerEpisode.value
  if (v == null || !t) {
    return 'Vector index stats unavailable.'
  }
  return (
    `${v.toLocaleString()} vector rows in the semantic index (many chunks per episode). ` +
    `Roughly ${r != null ? r.toFixed(1) : '—'} rows per catalog episode — not a percent of episodes.`
  )
})

const indexDaysSince = computed((): number | null => {
  const iso = props.indexEnv?.stats?.last_updated?.trim()
  if (!iso) {
    return null
  }
  const t = Date.parse(iso)
  if (Number.isNaN(t)) {
    return null
  }
  return Math.floor((Date.now() - t) / 86_400_000)
})

const giPctDisplay = computed(() => {
  const r = giRate.value
  return r == null ? '—' : `${Math.round(r * 100)}%`
})

const giCoverageTitle = computed(
  () =>
    'Episodes that have at least one GI artifact file, as a share of episodes in the corpus catalog (Coverage tab).',
)

const feedsIndexedCount = computed(() => props.indexEnv?.stats?.feeds_indexed?.length ?? 0)

const indexFeedsCoverageRatio = computed(() => {
  const fc = props.catalogFeedCount
  if (!fc || !props.indexEnv?.available) {
    return null
  }
  return feedsIndexedCount.value / fc
})

const giWarn = computed(() => giRate.value != null && giRate.value < GI_WARN)
/** Warn when few catalog feeds appear in index stats — not vector/episodes ratio. */
const indexWarn = computed(
  () =>
    indexFeedsCoverageRatio.value != null &&
    indexFeedsCoverageRatio.value < INDEX_WARN &&
    props.catalogFeedCount > 0,
)

const actionItems = computed(() => {
  type Act = { text: string; primary: string; secondary?: string; onPrimary: () => void; onSecondary?: () => void }
  const out: Act[] = []
  const r = latestRun.value
  const failed = r?.episode_outcomes?.failed ?? 0
  if (failed > 0) {
    out.push({
      text: `${failed} episode${failed === 1 ? '' : 's'} failed in last run`,
      primary: 'View failures',
      onPrimary: () => {
        dashNav.setHandoff({
          kind: 'library',
          missingGiOnly: false,
          since: r?.created_at?.slice(0, 10),
        })
        emit('open-library')
      },
    })
  }
  const noGi = props.coverage?.with_neither ?? 0
  const wg = props.coverage?.with_gi ?? 0
  const t = props.coverage?.total_episodes ?? 0
  if (t > 0 && giRate.value != null && giRate.value < GI_WARN) {
    const n = t - wg
    out.push({
      text: `${n} episode${n === 1 ? '' : 's'} have no GI artifacts`,
      primary: 'View in Library',
      onPrimary: () => {
        dashNav.setHandoff({ kind: 'library', missingGiOnly: true })
        emit('open-library')
      },
    })
  } else if (noGi > 0 && out.length === 0) {
    out.push({
      text: `${noGi} episode${noGi === 1 ? '' : 's'} have no GI artifacts`,
      primary: 'View in Library',
      onPrimary: () => {
        dashNav.setHandoff({ kind: 'library', missingGiOnly: true })
        emit('open-library')
      },
    })
  }
  if (!props.indexEnv?.available) {
    out.push({
      text: 'Vector index has not been built',
      primary: 'Build index',
      onPrimary: () => emit('rebuild-index'),
    })
  } else {
    const days = indexDaysSince.value
    if (days != null && days > 7 && props.indexEnv.reindex_recommended) {
      out.push({
        text: `Index last rebuilt ${days} day${days === 1 ? '' : 's'} ago`,
        primary: 'Rebuild now',
        onPrimary: () => emit('rebuild-index'),
      })
    }
  }
  const runs = props.runs
  if (runs.length > 0) {
    const newest = runs[0]?.created_at
    const oldest = runs[runs.length - 1]?.created_at
    if (newest && oldest) {
      const span = Date.parse(newest) - Date.parse(oldest)
      if (!Number.isNaN(span) && span > 8 * 86_400_000) {
        const last = Date.parse(newest)
        if (!Number.isNaN(last) && Date.now() - last > 7 * 86_400_000) {
          out.push({
            text: 'No pipeline runs in over 7 days',
            primary: 'View in Pipeline',
            onPrimary: () => emit('open-pipeline-run-history'),
          })
        }
      }
    }
  }
  return out.slice(0, 3)
})

function openLastRunDetails(): void {
  emit('open-pipeline-run-history')
}

function goLibraryAll(): void {
  dashNav.setHandoff({ kind: 'library' })
  emit('open-library')
}
</script>

<template>
  <div
    class="rounded-sm border border-border bg-elevated p-4 text-surface-foreground"
    data-testid="briefing-card"
  >
    <template v-if="!corpusPath.trim()">
      <p
        class="flex min-h-[7.5rem] items-center justify-center text-center text-sm text-muted"
        data-testid="briefing-no-corpus"
      >
        Set a corpus path in the status bar below to begin.
      </p>
    </template>
    <template v-else>
      <div class="flex flex-wrap items-start gap-x-6 gap-y-2">
        <div data-testid="briefing-last-run">
          <div class="text-[10px] font-semibold uppercase tracking-wider text-muted">
            Last run
          </div>
          <template v-if="!apiReady">
            <p class="mt-0.5 text-sm text-muted">
              Waiting for API…
            </p>
          </template>
          <template v-else-if="!latestRun">
            <p class="mt-0.5 text-sm text-muted">
              No pipeline runs found. Run <code class="font-mono text-xs">podcast scrape</code> to begin.
            </p>
          </template>
          <template v-else>
            <p class="mt-1 text-sm">
              <span
                v-if="runStatus === 'success'"
                class="text-success"
              >●</span>
              <span
                v-else-if="runStatus === 'partial'"
                class="text-warning"
              >●</span>
              <span
                v-else
                class="text-danger"
              >●</span>
              <span class="ml-1 capitalize">{{ runStatus === 'partial' ? 'Partial' : runStatus === 'failed' ? 'Failed' : 'Success' }}</span>
              <span class="text-muted"> · </span>
              <span>{{ (latestRun.episodes_scraped_total ?? 0).toLocaleString() }} episodes</span>
              <span class="text-muted"> · </span>
              <span>{{ formatDashboardRunDurationSeconds(latestRun.run_duration_seconds ?? null) || '—' }}</span>
              <span class="text-muted"> · </span>
              <span>{{ formatRelativeRunAge(latestRun.created_at) || '—' }}</span>
              <button
                type="button"
                class="ml-2 text-xs font-medium text-primary hover:underline"
                data-testid="briefing-last-run-details"
                @click="openLastRunDetails"
              >
                Details →
              </button>
            </p>
          </template>
        </div>

        <div data-testid="briefing-corpus-health">
          <div class="text-[10px] font-semibold uppercase tracking-wider text-muted">
            Corpus
          </div>
          <p class="mt-1 text-sm">
            <button
              type="button"
              class="hover:underline"
              @click="goLibraryAll"
            >
              {{ totalEpisodes.toLocaleString() }} episodes
            </button>
            <span class="text-muted"> · </span>
            <button
              type="button"
              class="hover:underline"
              :class="giWarn ? 'text-warning' : ''"
              :title="giCoverageTitle"
              @click="emit('select-tab', 'coverage')"
            >
              {{ giPctDisplay }} with GI
            </button>
            <span class="text-muted"> · </span>
            <button
              type="button"
              class="hover:underline"
              :class="indexWarn ? 'text-warning' : ''"
              :title="indexVectorsTitle"
              @click="emit('select-tab', 'coverage')"
            >
              {{ indexVectorsLabel }}
            </button>
            <span class="text-muted"> · </span>
            <button
              type="button"
              class="hover:underline"
              @click="goLibraryAll"
            >
              {{ catalogFeedCount }} feeds
            </button>
          </p>
        </div>
      </div>

      <div
        class="mt-3 border-t border-border pt-3"
        data-testid="briefing-action-items"
      >
        <template v-if="actionItems.length === 0">
          <p
            class="text-sm text-success"
            data-testid="briefing-all-clear"
          >
            ● Everything looks good
          </p>
        </template>
        <ul
          v-else
          class="space-y-1.5"
        >
          <li
            v-for="(it, idx) in actionItems"
            :key="idx"
            class="flex flex-wrap items-center gap-x-2 text-sm"
            data-testid="briefing-action-item"
          >
            <span>→ {{ it.text }}</span>
            <button
              type="button"
              class="text-xs font-medium text-primary hover:underline"
              @click="it.onPrimary()"
            >
              {{ it.primary }}
            </button>
            <button
              v-if="it.secondary && it.onSecondary"
              type="button"
              class="text-xs text-muted hover:underline"
              @click="it.onSecondary()"
            >
              {{ it.secondary }}
            </button>
          </li>
        </ul>
      </div>
    </template>
  </div>
</template>
