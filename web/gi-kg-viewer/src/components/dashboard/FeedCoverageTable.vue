<script setup lang="ts">
import { computed } from 'vue'
import type { CoverageFeedItem } from '../../api/corpusCoverageApi'

const props = defineProps<{
  rows: CoverageFeedItem[]
  feedsIndexed: string[]
}>()

const emit = defineEmits<{
  'select-feed': [feedId: string]
}>()

const indexedSet = computed(() => new Set(props.feedsIndexed.map((x) => x.trim()).filter(Boolean)))

function pct(n: number, d: number): number {
  if (!d) {
    return 0
  }
  return Math.round((n / d) * 100)
}

const insight = computed(() => {
  const r = props.rows[0]
  if (!r || !r.total) {
    return undefined
  }
  const gi = pct(r.with_gi, r.total)
  const missing = r.total - r.with_gi
  return `Feed ${r.display_title} has lowest GI coverage at ${gi}% — ${missing} episode${missing === 1 ? '' : 's'} without GI artifacts.`
})
</script>

<template>
  <div
    class="rounded border border-border bg-surface p-3 text-surface-foreground"
    data-testid="feed-coverage-table"
  >
    <h3 class="mb-2 text-sm font-semibold">
      Feed coverage (lowest GI first)
    </h3>
    <div
      v-if="rows.length === 0"
      class="text-xs text-muted"
    >
      No feeds in catalog.
    </div>
    <table
      v-else
      class="w-full border-collapse text-left text-xs"
    >
      <thead>
        <tr class="border-b border-border text-[10px] uppercase tracking-wide text-muted">
          <th class="py-1 pr-2">
            Feed
          </th>
          <th class="py-1 pr-2">
            Episodes
          </th>
          <th class="py-1 pr-2">
            GI
          </th>
          <th class="py-1 pr-2">
            KG
          </th>
          <th class="py-1">
            In index
          </th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="r in rows"
          :key="r.feed_id"
          class="border-b border-border/60 hover:bg-overlay/40"
          data-testid="feed-coverage-row"
        >
          <td
            class="cursor-pointer py-1.5 pr-2 font-medium"
            title="Open Corpus Library filtered to this feed"
            @click.stop="emit('select-feed', r.feed_id)"
          >
            {{ r.display_title }}
          </td>
          <td
            class="cursor-pointer py-1.5 pr-2 tabular-nums"
            title="Open Corpus Library filtered to this feed"
            @click.stop="emit('select-feed', r.feed_id)"
          >
            {{ r.total }}
          </td>
          <td
            class="cursor-pointer py-1.5 pr-2"
            :title="`GI coverage for this feed — click to open Library with this feed (${pct(r.with_gi, r.total)}%)`"
            @click.stop="emit('select-feed', r.feed_id)"
          >
            <div class="flex items-center gap-1">
              <div
                class="h-2 w-10 shrink-0 overflow-hidden rounded bg-overlay"
              >
                <div
                  class="h-full bg-gi"
                  :style="{ width: `${pct(r.with_gi, r.total)}%` }"
                />
              </div>
              <span class="tabular-nums text-muted">{{ pct(r.with_gi, r.total) }}%</span>
            </div>
          </td>
          <td
            class="cursor-pointer py-1.5 pr-2"
            :title="`KG coverage for this feed — click to open Library with this feed (${pct(r.with_kg, r.total)}%)`"
            @click.stop="emit('select-feed', r.feed_id)"
          >
            <div class="flex items-center gap-1">
              <div
                class="h-2 w-10 shrink-0 overflow-hidden rounded bg-overlay"
              >
                <div
                  class="h-full bg-kg"
                  :style="{ width: `${pct(r.with_kg, r.total)}%` }"
                />
              </div>
              <span class="tabular-nums text-muted">{{ pct(r.with_kg, r.total) }}%</span>
            </div>
          </td>
          <td
            class="cursor-pointer py-1.5 text-muted"
            title="Open Corpus Library filtered to this feed"
            @click.stop="emit('select-feed', r.feed_id)"
          >
            {{ indexedSet.has(r.feed_id) ? 'Yes' : '—' }}
          </td>
        </tr>
      </tbody>
    </table>
    <p
      v-if="insight"
      class="mt-2 text-[11px] text-muted"
    >
      {{ insight }}
    </p>
  </div>
</template>
