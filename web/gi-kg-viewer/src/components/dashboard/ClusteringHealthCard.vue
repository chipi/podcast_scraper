<script setup lang="ts">
/**
 * Storylines and themes over time (RFC-128, #2349).
 *
 * What the clustering artifacts cannot say on their own, because every run rewrites them: how
 * many storylines and themes there are and how that moved, whether the storyline registry is
 * healthy, what the last run created / renamed / quieted / absorbed / lost, and the actions an
 * operator takes on them — re-run storylines, rebuild themes, the storyline history seed.
 */
import { computed, ref, watch } from 'vue'
import {
  getClusteringHealth,
  postStorylineSeed,
  type ClusteringChanges,
  type ClusteringHealth,
  type SeedMode,
  type SeedResult,
} from '../../api/clusteringHealthApi'
import { submitEnrichmentJob } from '../../api/enrichmentApi'
import { useArtifactsStore } from '../../stores/artifacts'
import { useShellStore } from '../../stores/shell'
import MultiSeriesLineChart from './MultiSeriesLineChart.vue'

const shell = useShellStore()
const artifacts = useArtifactsStore()

const health = ref<ClusteringHealth | null>(null)
const state = ref<'idle' | 'ok' | 'error'>('idle')
const busy = ref<'' | 'rerun' | 'themes' | SeedMode>('')
const notice = ref('')
const seedPreview = ref<SeedResult | null>(null)

const showBlock = computed(() => Boolean(shell.healthStatus) && shell.hasCorpusPath)

async function load(): Promise<void> {
  const root = shell.corpusPath.trim()
  if (!root) return
  try {
    health.value = await getClusteringHealth(root)
    state.value = 'ok'
  } catch {
    state.value = 'error'
  }
}

watch(
  () => [shell.corpusPath, shell.healthStatus] as const,
  () => {
    if (showBlock.value) void load()
  },
  { immediate: true },
)

function day(ts: number | undefined): string {
  return ts ? new Date(ts * 1000).toISOString().slice(0, 10) : '—'
}

const story = computed(() => health.value?.storylines ?? null)
const latest = computed(() => story.value?.latest ?? null)
const themes = computed(() => health.value?.themes ?? null)

const coverage = computed((): string => {
  const r = latest.value
  if (!r?.episodes_total) return '—'
  return `${Math.round((100 * (r.episodes_covered ?? 0)) / r.episodes_total)}%`
})

const healthy = computed(() => story.value?.health?.ok !== false)

const storyChart = computed(() => {
  const s = story.value?.series ?? []
  return {
    labels: s.map((r) => day(r.ts)),
    series: [
      { label: 'Storylines', values: s.map((r) => r.storylines ?? 0) },
      { label: 'Quiet', values: s.map((r) => r.quiet ?? 0) },
      { label: 'Candidates', values: s.map((r) => r.candidates ?? 0) },
    ],
  }
})

const themeChart = computed(() => {
  const s = themes.value?.series ?? []
  return {
    labels: s.map((r) => day(r.ts)),
    series: [
      { label: 'Themes', values: s.map((r) => r.themes ?? 0) },
      { label: 'Kept their id', values: s.map((r) => r.kept ?? 0) },
    ],
  }
})

const KINDS = ['created', 'renamed', 'quiet', 'woke', 'absorbed', 'gone'] as const
const KIND_LABEL: Record<(typeof KINDS)[number], string> = {
  created: 'Created',
  renamed: 'Renamed',
  quiet: 'Went quiet',
  woke: 'Woke',
  absorbed: 'Absorbed',
  gone: 'Gone',
}

function changeRows(changes: ClusteringChanges | undefined): { kind: string; count: number; names: string }[] {
  if (!changes) return []
  return KINDS.filter((k) => (changes[k]?.length ?? 0) > 0).map((k) => {
    const items = changes[k] ?? []
    const names = items
      .slice(0, 5)
      .map((c) =>
        k === 'renamed'
          ? `${c.from} → ${c.to}`
          : k === 'absorbed'
            ? `${c.name} → ${c.into_name}`
            : (c.name ?? c.id),
      )
      .join(', ')
    return { kind: KIND_LABEL[k], count: items.length, names: items.length > 5 ? `${names}, …` : names }
  })
}

async function rerunStorylines(): Promise<void> {
  busy.value = 'rerun'
  notice.value = ''
  try {
    const job = await submitEnrichmentJob(shell.corpusPath.trim(), {
      only: ['topic_theme_clusters'],
      corpus_only: true,
    })
    notice.value = `Storyline run ${job.status} (job ${job.job_id}).`
  } catch (e) {
    notice.value = e instanceof Error ? e.message : String(e)
  } finally {
    busy.value = ''
  }
}

async function rebuildThemes(): Promise<void> {
  busy.value = 'themes'
  notice.value = ''
  try {
    await artifacts.rebuildTopicClusters()
    await load()
    notice.value = 'Themes rebuilt.'
  } catch (e) {
    notice.value = e instanceof Error ? e.message : String(e)
  } finally {
    busy.value = ''
  }
}

async function seed(mode: SeedMode): Promise<void> {
  busy.value = mode
  notice.value = ''
  try {
    const result = await postStorylineSeed(shell.corpusPath.trim(), mode)
    if (mode === 'preview') {
      seedPreview.value = result
    } else {
      seedPreview.value = null
      notice.value =
        mode === 'apply'
          ? result.already_taken
            ? 'This seed was already taken; nothing to do.'
            : 'Seed written; the next storyline run takes it in.'
          : `Undo: ${result.undone}${result.reason ? ` (${result.reason})` : ''}.`
      await load()
    }
  } catch (e) {
    notice.value = e instanceof Error ? e.message : String(e)
  } finally {
    busy.value = ''
  }
}
</script>

<template>
  <div
    v-if="showBlock"
    class="rounded border border-border bg-surface p-3 text-surface-foreground"
    data-testid="clustering-health-card"
  >
    <h3 class="mb-2 text-[10px] font-semibold uppercase tracking-wider text-muted">
      Storylines &amp; themes
    </h3>
    <p v-if="state === 'error'" class="text-xs text-danger" data-testid="clustering-health-error">
      Could not load the clustering history.
    </p>
    <template v-else-if="state === 'ok' && story">
      <div class="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <section data-testid="clustering-storylines">
          <h4 class="text-xs font-semibold">
            Storylines
          </h4>
          <dl class="mt-1 space-y-0.5 text-[11px]">
            <div class="flex justify-between gap-2">
              <dt class="text-muted">
                Now
              </dt>
              <dd data-testid="clustering-storylines-count">
                {{ latest?.storylines ?? '—' }}
                <span class="text-muted">({{ latest?.active ?? 0 }} active, {{ latest?.quiet ?? 0 }} quiet)</span>
              </dd>
            </div>
            <div class="flex justify-between gap-2">
              <dt class="text-muted">
                Candidates
              </dt>
              <dd>{{ latest?.candidates ?? '—' }}</dd>
            </div>
            <div class="flex justify-between gap-2">
              <dt class="text-muted">
                Episodes in a storyline
              </dt>
              <dd>{{ coverage }}</dd>
            </div>
            <div class="flex justify-between gap-2">
              <dt class="text-muted">
                Registry
              </dt>
              <dd
                :class="healthy ? 'text-success' : 'text-danger'"
                data-testid="clustering-registry-health"
              >
                {{ healthy ? 'healthy' : 'needs attention' }}
                <span class="text-muted">· from {{ story.health?.source ?? '—' }}</span>
              </dd>
            </div>
            <div class="flex justify-between gap-2">
              <dt class="text-muted">
                History seed
              </dt>
              <dd data-testid="clustering-seed-state">
                {{ story.seed.state }}
              </dd>
            </div>
            <div class="flex justify-between gap-2">
              <dt class="text-muted">
                Last run
              </dt>
              <dd>{{ day(latest?.ts) }}</dd>
            </div>
          </dl>
        </section>
        <section data-testid="clustering-themes">
          <h4 class="text-xs font-semibold">
            Themes
          </h4>
          <dl class="mt-1 space-y-0.5 text-[11px]">
            <div class="flex justify-between gap-2">
              <dt class="text-muted">
                Now
              </dt>
              <dd data-testid="clustering-themes-count">
                {{ themes?.latest?.themes ?? '—' }}
                <span class="text-muted">over {{ themes?.latest?.topics ?? '—' }} topics</span>
              </dd>
            </div>
            <div class="flex justify-between gap-2">
              <dt class="text-muted">
                Kept their id last run
              </dt>
              <dd>{{ themes?.latest?.kept ?? '—' }}</dd>
            </div>
            <div class="flex justify-between gap-2">
              <dt class="text-muted">
                Last run
              </dt>
              <dd>{{ day(themes?.latest?.ts) }}</dd>
            </div>
          </dl>
        </section>
      </div>

      <div class="mt-3 grid grid-cols-1 gap-3 lg:grid-cols-2">
        <MultiSeriesLineChart
          v-if="storyChart.labels.length > 1"
          title="Storylines over time"
          help-text="One point per storyline run; the earliest come from the history seed's replay."
          :labels="storyChart.labels"
          :series="storyChart.series"
        />
        <MultiSeriesLineChart
          v-if="themeChart.labels.length > 1"
          title="Themes over time"
          help-text="One point per theme build."
          :labels="themeChart.labels"
          :series="themeChart.series"
        />
      </div>

      <div class="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2 text-[11px]">
        <section data-testid="clustering-storyline-changes">
          <h4 class="text-xs font-semibold">
            Last storyline run
          </h4>
          <p v-if="!changeRows(latest?.changes).length" class="text-muted">
            No changes.
          </p>
          <ul v-else class="mt-1 space-y-0.5">
            <li v-for="row in changeRows(latest?.changes)" :key="row.kind">
              <span class="font-medium">{{ row.kind }} {{ row.count }}</span>
              <span class="text-muted">: {{ row.names }}</span>
            </li>
          </ul>
        </section>
        <section data-testid="clustering-theme-changes">
          <h4 class="text-xs font-semibold">
            Last theme build
          </h4>
          <p v-if="!changeRows(themes?.latest?.changes).length" class="text-muted">
            No changes.
          </p>
          <ul v-else class="mt-1 space-y-0.5">
            <li v-for="row in changeRows(themes?.latest?.changes)" :key="row.kind">
              <span class="font-medium">{{ row.kind }} {{ row.count }}</span>
              <span class="text-muted">: {{ row.names }}</span>
            </li>
          </ul>
        </section>
      </div>

      <div class="mt-3 flex flex-wrap gap-2" data-testid="clustering-actions">
        <button
          type="button"
          class="rounded border border-border bg-overlay px-2 py-0.5 text-[10px] font-medium hover:bg-elevated disabled:opacity-50"
          data-testid="clustering-rerun-storylines"
          :disabled="busy !== ''"
          @click="rerunStorylines"
        >
          {{ busy === 'rerun' ? 'Queueing…' : 'Re-run storylines' }}
        </button>
        <button
          type="button"
          class="rounded border border-border bg-overlay px-2 py-0.5 text-[10px] font-medium hover:bg-elevated disabled:opacity-50"
          data-testid="clustering-rebuild-themes"
          :disabled="busy !== ''"
          @click="rebuildThemes"
        >
          {{ busy === 'themes' ? 'Rebuilding…' : 'Rebuild themes' }}
        </button>
        <button
          type="button"
          class="rounded border border-border bg-overlay px-2 py-0.5 text-[10px] font-medium hover:bg-elevated disabled:opacity-50"
          data-testid="clustering-seed-preview"
          :disabled="busy !== ''"
          @click="seed('preview')"
        >
          {{ busy === 'preview' ? 'Replaying…' : 'Preview history seed' }}
        </button>
        <button
          v-if="seedPreview && !seedPreview.already_taken"
          type="button"
          class="rounded border border-border bg-overlay px-2 py-0.5 text-[10px] font-medium hover:bg-elevated disabled:opacity-50"
          data-testid="clustering-seed-apply"
          :disabled="busy !== ''"
          @click="seed('apply')"
        >
          Apply seed
        </button>
        <button
          v-if="story.seed.state !== 'none'"
          type="button"
          class="rounded border border-border bg-overlay px-2 py-0.5 text-[10px] font-medium hover:bg-elevated disabled:opacity-50"
          data-testid="clustering-seed-undo"
          :disabled="busy !== ''"
          @click="seed('undo')"
        >
          Undo seed
        </button>
      </div>
      <p
        v-if="seedPreview"
        class="mt-2 text-[11px] text-muted"
        data-testid="clustering-seed-report"
      >
        Seed preview: {{ seedPreview.current }} storylines today keep their id, name and anchor;
        {{ seedPreview.with_history }} take history from the replay ({{ seedPreview.folded }} replayed
        storylines folded in, {{ seedPreview.added_from_history?.length ?? 0 }} added);
        {{ seedPreview.candidates }} candidates; {{ seedPreview.history_months }} months for the chart.
        {{ seedPreview.already_taken ? 'This exact seed was already taken.' : '' }}
      </p>
      <p v-if="notice" class="mt-2 text-[11px]" data-testid="clustering-notice">
        {{ notice }}
      </p>
    </template>
  </div>
</template>
