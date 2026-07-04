<script setup lang="ts">
/**
 * Admin → Analytics (owned, self-hosted). A minimal first-cut analysis of the graph-usage log:
 * usage (event counts + node-tap kinds), size dynamics (node/edge/trail stats per redraw), and
 * breakage. The Graph block is the first section — ranking / listen / search can slot in as
 * siblings when we aggregate those logs too.
 */
import { computed, onMounted, ref } from 'vue'
import { fetchGraphAnalyticsSummary, type GraphAnalyticsSummary } from '../../api/authApi'

const summary = ref<GraphAnalyticsSummary | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)

async function load(): Promise<void> {
  loading.value = true
  error.value = null
  try {
    summary.value = await fetchGraphAnalyticsSummary()
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to load graph analytics'
  } finally {
    loading.value = false
  }
}
onMounted(load)

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
    </div>
  </div>
</template>
