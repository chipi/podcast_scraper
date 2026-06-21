<script setup lang="ts">
import { ref, watch } from 'vue'
import { getScheduledJobs, type ScheduledJobItem } from '../../api/scheduledJobsApi'
import { isValidCron, nextCronRuns } from '../../utils/cronPreview'
import { formatAbsoluteUtc } from '../../utils/relativeTime'
import { StaleGeneration } from '../../utils/staleGeneration'
import RelativeTime from '../shared/RelativeTime.vue'
import ToggleSwitch from '../shared/ToggleSwitch.vue'

/**
 * Scheduled feed-sweep section (#709) under Configuration. Reads
 * `GET /api/scheduled-jobs` (Name / Cron / Enabled / Next run) and emits a
 * `toggle` so the parent (StatusBar) can rewrite the `enabled:` field via
 * `PUT /api/operator-config`. Authoring of schedules stays in Job Configuration.
 */
const props = withDefaults(
  defineProps<{
    corpusPath: string
    active: boolean
    /** Bumped by the parent after a successful toggle to re-fetch. */
    reloadNonce?: number
    busy?: boolean
  }>(),
  { reloadNonce: 0, busy: false },
)

const emit = defineEmits<{ toggle: [name: string, enabled: boolean] }>()

const jobs = ref<ScheduledJobItem[]>([])
const schedulerRunning = ref(false)
const timezone = ref('UTC')
const loading = ref(false)
const error = ref<string | null>(null)
const gate = new StaleGeneration()

async function load(): Promise<void> {
  const p = props.corpusPath.trim()
  if (!p) {
    jobs.value = []
    return
  }
  const seq = gate.bump()
  loading.value = true
  error.value = null
  try {
    const res = await getScheduledJobs(p)
    if (gate.isStale(seq)) return
    jobs.value = res.jobs
    schedulerRunning.value = res.scheduler_running
    timezone.value = res.timezone
  } catch (e) {
    if (gate.isCurrent(seq)) {
      error.value = e instanceof Error ? e.message : String(e)
    }
  } finally {
    if (gate.isCurrent(seq)) loading.value = false
  }
}

watch(
  () => [props.active, props.corpusPath, props.reloadNonce] as const,
  ([active]) => {
    if (active) void load()
  },
  { immediate: true },
)

function cronPreviewTitle(cron: string): string {
  const runs = nextCronRuns(cron, 3, { tz: timezone.value })
  if (!runs) return 'Invalid cron expression'
  return `Next runs (${timezone.value}):\n${runs.map((r) => formatAbsoluteUtc(r)).join('\n')}`
}
</script>

<template>
  <div
    class="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto overscroll-contain"
    data-testid="scheduled-jobs-section"
  >
    <p class="shrink-0 text-[10px] text-muted leading-snug" data-testid="scheduled-jobs-status">
      Cron schedules from <code class="rounded bg-overlay px-0.5 font-mono text-[9px]">scheduled_jobs:</code>
      in <strong class="text-surface-foreground">viewer_operator.yaml</strong>.
      Scheduler:
      <span :class="schedulerRunning ? 'text-success' : 'text-muted'">{{ schedulerRunning ? 'running' : 'stopped' }}</span>
      · timezone {{ timezone }}. Add or edit schedules in <strong class="text-surface-foreground">Job Configuration</strong>.
    </p>

    <p
      v-if="error"
      class="shrink-0 rounded border border-danger/40 bg-danger/10 px-2 py-1 text-[10px] text-danger"
      data-testid="scheduled-jobs-error"
    >
      {{ error }}
    </p>

    <p v-if="loading && jobs.length === 0" class="shrink-0 text-[10px] text-muted">
      Loading…
    </p>

    <div
      v-else-if="jobs.length === 0"
      class="shrink-0 rounded border border-border/60 bg-overlay/40 px-2 py-2 text-[10px] text-muted"
      data-testid="scheduled-jobs-empty"
    >
      No scheduled jobs configured. Add a <code class="font-mono text-[9px]">scheduled_jobs:</code> list in Job Configuration.
    </div>

    <table v-else class="w-full border-collapse text-[11px]">
      <thead>
        <tr class="border-b border-border text-left text-[10px] text-muted">
          <th class="py-1 pr-2 font-medium">Name</th>
          <th class="py-1 pr-2 font-medium">Cron</th>
          <th class="py-1 pr-2 font-medium">Enabled</th>
          <th class="py-1 font-medium">Next run</th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="(job, idx) in jobs"
          :key="job.name"
          class="border-b border-border/60"
          :class="{ 'opacity-60': !job.enabled }"
          :data-testid="`scheduled-jobs-row-${idx}`"
        >
          <td class="py-1 pr-2 font-medium text-surface-foreground">
            {{ job.name }}
          </td>
          <td class="py-1 pr-2">
            <code
              class="rounded bg-overlay px-0.5 font-mono text-[10px]"
              :title="cronPreviewTitle(job.cron)"
            >{{ job.cron }}</code>
            <span
              v-if="!isValidCron(job.cron)"
              class="ml-1 rounded bg-danger/15 px-1 text-[9px] text-danger"
              data-testid="scheduled-jobs-invalid-cron"
            >invalid cron</span>
          </td>
          <td class="py-1 pr-2">
            <ToggleSwitch
              :model-value="job.enabled"
              :disabled="busy"
              :label="`Enable ${job.name}`"
              :testid="`scheduled-jobs-toggle-${idx}`"
              @update:model-value="emit('toggle', job.name, $event)"
            />
          </td>
          <td class="py-1" :data-testid="`scheduled-jobs-next-${idx}`">
            <span v-if="!job.enabled" class="text-muted">—</span>
            <span
              v-else-if="!isValidCron(job.cron)"
              class="text-danger"
            >invalid cron</span>
            <RelativeTime v-else :iso="job.next_run_at" />
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>
