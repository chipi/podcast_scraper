<script setup lang="ts">
/**
 * Show activity — episodes published per month as a compact bar sparkline ("how active /
 * consistent is this show"). Built client-side from the loaded episodes' publish dates;
 * fills gaps as zero-height bars, caps the window so it stays small, and hides itself when
 * there aren't enough dated episodes to be meaningful.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import type { EpisodeSummary } from '../services/types'

const props = defineProps<{ episodes: EpisodeSummary[] }>()
const { t } = useI18n()

const MAX_MONTHS = 24

const bars = computed(() => {
  const byMonth = new Map<string, number>()
  for (const e of props.episodes) {
    const key = (e.publish_date ?? '').slice(0, 7) // YYYY-MM
    if (key.length === 7) byMonth.set(key, (byMonth.get(key) ?? 0) + 1)
  }
  if (byMonth.size < 2) return []
  const months = [...byMonth.keys()].sort()
  const [ey, em] = months[months.length - 1].split('-').map(Number)
  let [y, m] = months[0].split('-').map(Number)
  const out: Array<{ key: string; count: number }> = []
  while ((y < ey || (y === ey && m <= em)) && out.length < MAX_MONTHS) {
    const key = `${y}-${String(m).padStart(2, '0')}`
    out.push({ key, count: byMonth.get(key) ?? 0 })
    m += 1
    if (m > 12) {
      m = 1
      y += 1
    }
  }
  return out
})

const maxCount = computed(() => Math.max(1, ...bars.value.map((b) => b.count)))
const rangeLabel = computed(() => {
  if (!bars.value.length) return ''
  return `${bars.value[0].key} – ${bars.value[bars.value.length - 1].key}`
})
</script>

<template>
  <section v-if="bars.length" class="mb-6" data-testid="show-activity">
    <div class="mb-1.5 flex items-baseline justify-between gap-2">
      <h3 class="lp-kicker">{{ t('podcast.activity') }}</h3>
      <span class="text-[10px] text-muted">{{ rangeLabel }}</span>
    </div>
    <div class="flex items-end gap-px" style="height: 44px">
      <div
        v-for="b in bars"
        :key="b.key"
        class="min-w-[3px] flex-1 rounded-sm bg-accent/70"
        :style="{ height: Math.round((b.count / maxCount) * 40) + 4 + 'px' }"
        :title="`${b.key} · ${b.count}`"
        :data-testid="`show-activity-bar-${b.key}`"
      />
    </div>
  </section>
</template>
