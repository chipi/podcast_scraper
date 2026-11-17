<script setup lang="ts">
import { computed } from 'vue'
import { isValidCron, nextCronRuns } from '../../utils/cronPreview'
import { formatAbsoluteUtc } from '../../utils/relativeTime'
import { parseScheduledJobsFromYaml } from '../../utils/scheduledJobsYaml'

/**
 * Live preview + validation for `scheduled_jobs:` as the operator edits the Job
 * Configuration YAML (#709). For each parsed entry it flags an invalid cron and
 * previews the next 3 fire times (client-side via cron-parser) — so a bad cron
 * is caught before Save rather than silently never firing.
 */
const props = defineProps<{ yaml: string }>()

const rows = computed(() =>
  parseScheduledJobsFromYaml(props.yaml).map((job) => ({
    ...job,
    valid: isValidCron(job.cron),
    next: nextCronRuns(job.cron, 3) ?? [],
  })),
)

const invalidCount = computed(() => rows.value.filter((r) => !r.valid).length)
</script>

<template>
  <div
    v-if="rows.length"
    class="shrink-0 rounded border border-border bg-elevated/40 p-2 text-[10px]"
    data-testid="cron-schedule-preview"
  >
    <p class="mb-1 font-medium text-muted">
      Schedule preview
      <span
        v-if="invalidCount"
        class="ml-1 rounded bg-danger/15 px-1 text-danger"
        data-testid="cron-schedule-preview-invalid-summary"
      >{{ invalidCount }} invalid</span>
    </p>
    <ul class="space-y-1">
      <li
        v-for="(row, idx) in rows"
        :key="idx"
        :data-testid="`cron-schedule-preview-row-${idx}`"
      >
        <span class="font-medium text-surface-foreground">{{ row.name || '(unnamed)' }}</span>
        <code class="ml-1 rounded bg-overlay px-0.5 font-mono">{{ row.cron || '(no cron)' }}</code>
        <span
          v-if="!row.valid"
          class="ml-1 rounded bg-danger/15 px-1 text-danger"
          :data-testid="`cron-schedule-preview-invalid-${idx}`"
        >invalid cron</span>
        <span v-else-if="!row.enabled" class="ml-1 text-muted">(disabled)</span>
        <span
          v-else
          class="ml-1 text-muted"
          :title="row.next.map((r) => formatAbsoluteUtc(r)).join('\n')"
        >next: {{ formatAbsoluteUtc(row.next[0]) }}<span v-if="row.next.length > 1"> · +{{ row.next.length - 1 }} more</span></span>
      </li>
    </ul>
  </div>
</template>
