<script setup lang="ts">
/**
 * A storyline's summary (#2344): one paragraph on the thread, written in the background for every
 * storyline. On the storyline page and at the top of its Learn path. Renders nothing until one has
 * been written, or when it cannot be read — it is never a blocking part of the page.
 */
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { getStorylineSummary, type StorylineSummary } from '../services/api'

const props = defineProps<{ storylineId: string }>()
const { t, locale } = useI18n()
const summary = ref<StorylineSummary | null>(null)

watch(
  () => props.storylineId,
  async (id) => {
    summary.value = null
    if (!id) return
    const got = await getStorylineSummary(id).catch(() => null)
    if (props.storylineId === id) summary.value = got
  },
  { immediate: true },
)

/** "Oct 13", the year only when it is not this one — the board Brief's date style. */
const when = computed(() => {
  const at = summary.value?.created_at
  if (!at) return ''
  const d = new Date(at * 1000)
  const sameYear = d.getFullYear() === new Date().getFullYear()
  return new Intl.DateTimeFormat(locale.value, {
    month: 'short',
    day: 'numeric',
    ...(sameYear ? {} : { year: 'numeric' }),
  }).format(d)
})
</script>

<template>
  <section v-if="summary" class="mb-5" data-testid="storyline-summary">
    <h3 class="lp-section mb-1">{{ t('kp.summary') }}</h3>
    <p class="text-sm leading-relaxed text-surface-foreground">{{ summary.text }}</p>
    <p class="mt-1 text-xs text-muted" data-testid="storyline-summary-when">
      {{ t('boardBrief.summarisedOn', { date: when }) }}<template v-if="summary.stale"> · {{ t('storylineSummary.stale') }}</template>
    </p>
  </section>
</template>
