<script setup lang="ts">
/**
 * Topic conversation arc — the aggregate-first view of how a topic's conversation evolved.
 *
 * A generic topic like "AI" can carry 1000s of insights across the corpus; a flat chronological
 * list is unusable. So we show the SHAPE first: a compact row of weekly stacked bars (bar height =
 * volume, colour = neg/neu/pos sentiment mix) from `GET /api/topics/{id}/conversation-arc`. Click a
 * week to drill into just that week's insights (dozens, not 1000s) — a tinted list from the existing
 * timeline endpoint (whose insights now carry `sentiment`). Aggregate → detail-on-demand.
 */
import { computed, ref, watch } from 'vue'
import {
  fetchTopicConversationArc,
  fetchTopicTimeline,
  type CilTopicConversationArcWeek,
} from '../../api/cilApi'
import { useShellStore } from '../../stores/shell'
import { primaryTextFromLooseGiNode } from '../../utils/parsing'
import { formatCalendarDateForDisplay } from '../../utils/formatting'

const props = defineProps<{ topicId: string }>()
const shell = useShellStore()

const weeks = ref<CilTopicConversationArcWeek[]>([])
const rows = ref<Array<{ text: string; date: string | null; week: string; label: string }>>([])
const loading = ref(false)
const error = ref<string | null>(null)
const selectedWeek = ref<string | null>(null)

const SENT_CLASS: Record<string, string> = {
  negative: 'bg-rose-500/70',
  neutral: 'bg-slate-400/50',
  positive: 'bg-emerald-500/70',
}
const ROW_CLASS: Record<string, string> = {
  negative: 'border-rose-700/40 bg-rose-900/10',
  neutral: 'border-default bg-overlay/40',
  positive: 'border-emerald-700/40 bg-emerald-900/10',
}

const maxVolume = computed(() => Math.max(1, ...weeks.value.map((w) => w.volume)))
const totalInsights = computed(() => weeks.value.reduce((n, w) => n + w.volume, 0))

/** ``YYYY-MM-DD`` → ISO ``YYYY-Www`` (client mirror of the server bucket key). */
function isoWeek(dateStr: string | null): string | null {
  if (!dateStr || dateStr.length < 10) return null
  const d = new Date(dateStr.slice(0, 10) + 'T00:00:00Z')
  if (Number.isNaN(d.getTime())) return null
  const target = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()))
  const day = target.getUTCDay() || 7
  target.setUTCDate(target.getUTCDate() + 4 - day)
  const yearStart = new Date(Date.UTC(target.getUTCFullYear(), 0, 1))
  const wk = Math.ceil(((target.getTime() - yearStart.getTime()) / 86400000 + 1) / 7)
  return `${target.getUTCFullYear()}-W${String(wk).padStart(2, '0')}`
}

const visibleRows = computed(() => {
  const list = selectedWeek.value
    ? rows.value.filter((r) => r.week === selectedWeek.value)
    : rows.value
  return list.slice(0, 300)
})

async function load(): Promise<void> {
  const root = shell.corpusPath?.trim()
  const tid = props.topicId?.trim()
  selectedWeek.value = null
  weeks.value = []
  rows.value = []
  error.value = null
  if (!root || !tid) return
  loading.value = true
  try {
    const [arc, tl] = await Promise.all([
      fetchTopicConversationArc(root, tid),
      fetchTopicTimeline(root, tid),
    ])
    weeks.value = arc.weeks
    const flat: typeof rows.value = []
    for (const block of tl.episodes) {
      const wk = isoWeek(block.publish_date)
      if (!wk) continue
      for (const n of block.insights) {
        const text = primaryTextFromLooseGiNode(n)
        if (!text) continue
        const sent = (n as { sentiment?: { label?: string } }).sentiment
        flat.push({ text, date: block.publish_date, week: wk, label: sent?.label ?? 'neutral' })
      }
    }
    rows.value = flat
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}

function toggleWeek(w: string): void {
  selectedWeek.value = selectedWeek.value === w ? null : w
}

watch(() => [shell.corpusPath, props.topicId], () => void load(), { immediate: true })
</script>

<template>
  <section
    v-if="weeks.length"
    class="rounded border border-default bg-overlay/40 p-2 text-[11px]"
    data-testid="topic-conversation-arc"
  >
    <div class="mb-1 flex items-center justify-between gap-2">
      <p class="text-[10px] font-semibold uppercase tracking-wider text-muted">
        Conversation over time
        <span class="font-normal normal-case text-muted/70">· {{ totalInsights }} insights</span>
      </p>
      <div class="flex items-center gap-2 text-[9px] text-muted">
        <span class="inline-flex items-center gap-1"><span class="inline-block h-2 w-2 rounded-sm bg-rose-500/70" />neg</span>
        <span class="inline-flex items-center gap-1"><span class="inline-block h-2 w-2 rounded-sm bg-slate-400/50" />neu</span>
        <span class="inline-flex items-center gap-1"><span class="inline-block h-2 w-2 rounded-sm bg-emerald-500/70" />pos</span>
      </div>
    </div>

    <!-- Weekly stacked bars: height = volume, segments = sentiment mix. Click to drill. -->
    <div class="flex items-end gap-px overflow-x-auto pb-1" style="height: 56px" data-testid="tca-bars">
      <button
        v-for="w in weeks"
        :key="w.week"
        type="button"
        class="flex shrink-0 flex-col justify-end rounded-sm hover:outline hover:outline-1 hover:outline-primary"
        :class="selectedWeek === w.week ? 'outline outline-1 outline-primary' : ''"
        style="width: 7px"
        :style="{ height: Math.round((w.volume / maxVolume) * 48) + 4 + 'px' }"
        :title="`${w.week} · ${w.volume} insights · ${w.negative}◂ ${w.neutral}◦ ${w.positive}▸ · avg ${w.avg_compound.toFixed(2)}`"
        :data-testid="`tca-bar-${w.week}`"
        @click="toggleWeek(w.week)"
      >
        <span
          v-if="w.positive"
          class="w-full"
          :style="{ height: (w.positive / w.volume) * 100 + '%' }"
          :class="SENT_CLASS.positive"
        />
        <span
          v-if="w.neutral"
          class="w-full"
          :style="{ height: (w.neutral / w.volume) * 100 + '%' }"
          :class="SENT_CLASS.neutral"
        />
        <span
          v-if="w.negative"
          class="w-full"
          :style="{ height: (w.negative / w.volume) * 100 + '%' }"
          :class="SENT_CLASS.negative"
        />
      </button>
    </div>

    <!-- Drill: the selected week's insights (or a capped recent slice), sentiment-tinted. -->
    <div class="mt-1 flex items-center justify-between text-[9px] text-muted">
      <span v-if="selectedWeek" data-testid="tca-week-filter">
        Week {{ selectedWeek }} · {{ visibleRows.length }} insights
        <button type="button" class="ml-1 underline hover:text-surface-foreground" @click="selectedWeek = null">clear</button>
      </span>
      <span v-else>{{ Math.min(rows.length, 300) }} of {{ rows.length }} — click a week to focus</span>
    </div>
    <ul class="mt-1 max-h-[40vh] space-y-1 overflow-y-auto pr-0.5" data-testid="tca-insights">
      <li
        v-for="(r, i) in visibleRows"
        :key="i"
        class="rounded border px-2 py-1 text-[10px] leading-snug"
        :class="ROW_CLASS[r.label] || ROW_CLASS.neutral"
      >
        <span class="line-clamp-3 text-surface-foreground">{{ r.text }}</span>
        <span v-if="r.date" class="mt-0.5 block text-[9px] text-muted">{{ formatCalendarDateForDisplay(r.date) }}</span>
      </li>
    </ul>
    <p v-if="error" class="mt-1 text-[9px] text-rose-300" data-testid="tca-error">{{ error }}</p>
  </section>
</template>
