<script setup lang="ts">
import { computed } from 'vue'
import type { TopPersonItem } from '../../api/corpusPersonsApi'

const props = defineProps<{
  persons: TopPersonItem[]
  loading: boolean
  error: string | null
}>()

const insight = computed(() => {
  const p = props.persons[0]
  if (!p) {
    return undefined
  }
  return `${props.persons.length} speakers with grounded insights — ${p.display_name} leads with ${p.insight_count} insights across ${p.episode_count} episodes.`
})
</script>

<template>
  <section
    class="rounded border border-border bg-surface p-3 text-surface-foreground"
    data-testid="intelligence-top-voices"
  >
    <h3 class="mb-2 text-sm font-semibold">
      Top voices
    </h3>
    <p
      v-if="loading"
      class="text-xs text-muted"
    >
      Loading…
    </p>
    <p
      v-else-if="error"
      class="text-xs text-danger"
    >
      {{ error }}
    </p>
    <p
      v-else-if="persons.length === 0"
      class="text-xs text-muted"
    >
      No speaker intelligence found for this corpus.
    </p>
    <div
      v-else
      class="flex flex-wrap gap-2"
    >
      <div
        v-for="p in persons"
        :key="p.person_id"
        class="min-w-[180px] flex-1 rounded border border-border bg-elevated p-2 text-sm"
      >
        <div class="font-semibold">
          {{ p.display_name }}
        </div>
        <div class="mt-1 text-[10px] text-muted">
          {{ p.episode_count }} episodes · <span class="text-gi">{{ p.insight_count }} insights</span>
        </div>
        <div class="mt-1 flex flex-wrap gap-1">
          <span
            v-for="t in p.top_topics"
            :key="t"
            class="rounded border border-kg px-1 py-0.5 text-[10px] text-surface-foreground"
          >{{ t }}</span>
        </div>
      </div>
    </div>
    <p
      v-if="insight"
      class="mt-2 text-[11px] text-muted"
    >
      {{ insight }}
    </p>
  </section>
</template>
