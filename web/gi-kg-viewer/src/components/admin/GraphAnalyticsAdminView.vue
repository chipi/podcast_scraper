<script setup lang="ts">
/**
 * Admin → Analytics (owned, self-hosted). Aggregate graph analysis (usage / size / breakage) plus
 * a per-session drill-down: pick a session to see its step-by-step timeline, or Replay it in the
 * graph. The Graph block is the first "Analytics" section — ranking / listen / search can slot in
 * as siblings when we aggregate those logs too.
 */
import { computed, onMounted, ref } from 'vue'
import {
  fetchGraphAnalyticsSummary,
  fetchGraphSession,
  fetchGraphSessions,
  type GraphAnalyticsSummary,
  type GraphSessionSummary,
} from '../../api/authApi'

const emit = defineEmits<{ (e: 'replay', sessionId: string): void }>()

const summary = ref<GraphAnalyticsSummary | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)

const sessions = ref<GraphSessionSummary[]>([])
const selectedId = ref<string | null>(null)
const timeline = ref<Array<Record<string, unknown>>>([])
const timelineLoading = ref(false)

async function load(): Promise<void> {
  loading.value = true
  error.value = null
  try {
    ;[summary.value, sessions.value] = await Promise.all([
      fetchGraphAnalyticsSummary(),
      fetchGraphSessions(),
    ])
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to load graph analytics'
  } finally {
    loading.value = false
  }
}
onMounted(load)

async function selectSession(id: string): Promise<void> {
  selectedId.value = id
  timelineLoading.value = true
  try {
    timeline.value = await fetchGraphSession(id)
  } catch {
    timeline.value = []
  } finally {
    timelineLoading.value = false
  }
}

/** Human-readable one-line summary of a captured event, for the step-by-step timeline. */
function stepLabel(e: Record<string, unknown>): string {
  const a = String(e.action ?? '')
  if (a === 'graph_node_tap') return `tapped ${e.kind ?? 'node'}`
  if (a === 'graph_rail_nav') return `navigated → ${e.to_kind ?? 'node'} (trail ${e.trail_size ?? '?'})`
  if (a === 'graph_redraw') return `redraw · ${e.nodes ?? '?'} nodes / ${e.edges ?? '?'} edges`
  if (a === 'graph_recenter') return `re-centre (${e.source ?? '?'})`
  if (a === 'graph_broke') return `broke: ${e.reason ?? '?'}`
  return a
}

const byCountDesc = (rec: Record<string, number> | undefined): Array<[string, number]> =>
  Object.entries(rec ?? {}).sort((a, b) => b[1] - a[1])
const actionRows = computed(() => byCountDesc(summary.value?.by_action))
const tapRows = computed(() => byCountDesc(summary.value?.node_taps_by_kind))
const breakRows = computed(() => byCountDesc(summary.value?.breakage.by_reason))
const sizeMetrics = ['nodes', 'edges', 'trail'] as const
</script>

<template>
  <div class="mx-auto max-w-2xl" data-testid="graph-analytics-admin">
    <div class="mb-2 flex items-center justify-between gap-2">
      <h2 class="text-base font-semibold text-surface-foreground">Analytics — Graph</h2>
      <button
        type="button"
        class="rounded border border-border px-2 py-0.5 text-xs text-muted hover:bg-overlay"
        data-testid="graph-analytics-refresh"
        @click="load"
      >
        Refresh
      </button>
    </div>
    <p v-if="loading" class="text-xs text-muted" data-testid="graph-analytics-loading">Loading…</p>
    <p
      v-else-if="error"
      class="rounded border border-danger/40 bg-danger/10 px-2 py-1 text-xs text-danger"
      role="alert"
      data-testid="graph-analytics-error"
    >
      {{ error }}
    </p>
    <div v-else-if="summary" class="space-y-4" data-testid="graph-analytics-body">
      <p class="text-xs text-muted">
        <span data-testid="ga-total">{{ summary.total_events }}</span> events ·
        <span data-testid="ga-users">{{ summary.users }}</span> users
      </p>

      <!-- Size / dynamics -->
      <section class="rounded border border-border bg-elevated/40 p-2">
        <h3 class="mb-1 text-[11px] font-semibold uppercase tracking-wide text-muted">
          Graph size (per redraw · {{ summary.size.samples }} samples)
        </h3>
        <table class="w-full text-xs">
          <thead>
            <tr class="text-muted">
              <th class="text-left font-medium"></th>
              <th class="font-medium">min</th>
              <th class="font-medium">avg</th>
              <th class="font-medium">p50</th>
              <th class="font-medium">p95</th>
              <th class="font-medium">max</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="m in sizeMetrics" :key="m" :data-testid="`ga-size-${m}`">
              <td class="text-left font-medium capitalize">{{ m }}</td>
              <td class="text-center">{{ summary.size[m].min }}</td>
              <td class="text-center">{{ summary.size[m].avg }}</td>
              <td class="text-center">{{ summary.size[m].p50 }}</td>
              <td class="text-center">{{ summary.size[m].p95 }}</td>
              <td class="text-center">{{ summary.size[m].max }}</td>
            </tr>
          </tbody>
        </table>
      </section>

      <!-- Usage -->
      <section class="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div class="rounded border border-border bg-elevated/40 p-2">
          <h3 class="mb-1 text-[11px] font-semibold uppercase tracking-wide text-muted">Actions</h3>
          <ul class="space-y-0.5 text-xs" data-testid="ga-actions">
            <li v-for="[a, n] in actionRows" :key="a" class="flex justify-between">
              <span>{{ a }}</span><span class="font-semibold">{{ n }}</span>
            </li>
            <li v-if="!actionRows.length" class="text-muted">No events yet.</li>
          </ul>
        </div>
        <div class="rounded border border-border bg-elevated/40 p-2">
          <h3 class="mb-1 text-[11px] font-semibold uppercase tracking-wide text-muted">
            Node taps by kind
          </h3>
          <ul class="space-y-0.5 text-xs" data-testid="ga-taps">
            <li v-for="[k, n] in tapRows" :key="k" class="flex justify-between">
              <span>{{ k }}</span><span class="font-semibold">{{ n }}</span>
            </li>
            <li v-if="!tapRows.length" class="text-muted">No taps yet.</li>
          </ul>
        </div>
      </section>

      <!-- Breakage -->
      <section class="rounded border border-border bg-elevated/40 p-2">
        <h3 class="mb-1 text-[11px] font-semibold uppercase tracking-wide text-muted">
          Breakage — {{ summary.breakage.count }}
        </h3>
        <ul class="space-y-0.5 text-xs" data-testid="ga-breakage">
          <li v-for="[r, n] in breakRows" :key="r" class="flex justify-between">
            <span>{{ r }}</span><span class="font-semibold">{{ n }}</span>
          </li>
          <li v-if="!breakRows.length" class="text-grounded">None recorded.</li>
        </ul>
      </section>

      <!-- Sessions (step-by-step + replay) -->
      <section class="rounded border border-border bg-elevated/40 p-2">
        <h3 class="mb-1 text-[11px] font-semibold uppercase tracking-wide text-muted">Sessions</h3>
        <ul class="max-h-40 space-y-0.5 overflow-y-auto text-xs" data-testid="ga-sessions">
          <li
            v-for="s in sessions"
            :key="s.session_id"
            class="flex items-center gap-1"
          >
            <button
              type="button"
              class="min-w-0 flex-1 truncate rounded px-1 py-0.5 text-left hover:bg-overlay"
              :class="selectedId === s.session_id ? 'bg-overlay' : ''"
              :data-testid="`ga-session-${s.session_id}`"
              @click="selectSession(s.session_id)"
            >
              {{ s.user_id }} · {{ s.count }} ev · size {{ s.size_min }}–{{ s.size_max }}
            </button>
            <button
              type="button"
              class="shrink-0 rounded px-1 py-0.5 text-accent hover:underline"
              data-testid="ga-replay"
              @click="emit('replay', s.session_id)"
            >
              Replay ▶
            </button>
          </li>
          <li v-if="!sessions.length" class="text-muted">No sessions yet.</li>
        </ul>
        <div v-if="selectedId" class="mt-2 border-t border-border pt-2" data-testid="ga-timeline">
          <p v-if="timelineLoading" class="text-[11px] text-muted">Loading timeline…</p>
          <ol v-else class="space-y-0.5 text-[11px]">
            <li v-for="(e, i) in timeline" :key="i" class="flex gap-2">
              <span class="w-5 shrink-0 text-right text-muted">{{ i + 1 }}</span>
              <span>{{ stepLabel(e) }}</span>
            </li>
          </ol>
        </div>
      </section>
    </div>
  </div>
</template>
