<script setup lang="ts">
/**
 * Topic perspectives (#1146) — multi-perspective synthesis on the topic card.
 * Each guest who spoke on the topic, with their grounded insights. Self-fetches
 * from GET /api/app/topics/{id}/perspectives; renders nothing when the topic has
 * none. Speaker names tap through to their person card (the same `open` contract
 * the card's other people rows use).
 */
import { ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'

import { getTopicPerspectives } from '../services/api'
import type { TopicPerspective } from '../services/types'

const props = defineProps<{ id: string; scope?: 'all' | 'mine' }>()
const emit = defineEmits<{ (e: 'open', payload: { kind: 'person' | 'topic'; id: string }): void }>()

const { t } = useI18n()

const perspectives = ref<TopicPerspective[]>([])
/**
 * Did the fetch fail? Distinct from "this topic has no perspectives".
 *
 * The old code caught every failure into an empty array, and the section is `v-if`'d on being
 * non-empty — so a failed request rendered as the section simply NOT EXISTING, indistinguishable
 * from a topic nobody discussed, with no retry and no way for the reader to tell. `serve` is a
 * single process, so one concurrent search is enough to make this request time out; the reader
 * then sees a topic card that quietly claims nobody had a perspective on it. A wrong answer
 * presented as a confident one.
 */
const failed = ref(false)
/** Guards against a slow reply for a topic the reader has already navigated away from. */
let requestSeq = 0

async function load(): Promise<void> {
  const mine = ++requestSeq
  perspectives.value = []
  failed.value = false
  // One retry before giving up: the failure this exists for is a transient timeout under load,
  // and re-asking costs one request where the alternative is silently withholding real content.
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      const r = await getTopicPerspectives(props.id, props.scope)
      if (mine !== requestSeq) return
      perspectives.value = r.perspectives
      return
    } catch {
      if (mine !== requestSeq) return
      if (attempt === 0) await new Promise((resolve) => setTimeout(resolve, 600))
    }
  }
  if (mine === requestSeq) failed.value = true
}

watch(() => [props.id, props.scope] as const, () => void load(), { immediate: true })

// Show up to PREVIEW insights per speaker; the rest sit behind a per-speaker toggle.
const PREVIEW = 3
const expanded = ref<Set<string>>(new Set())
function toggle(personId: string): void {
  const next = new Set(expanded.value)
  if (next.has(personId)) next.delete(personId)
  else next.add(personId)
  expanded.value = next
}
</script>

<template>
  <!-- A failed load says so and offers a retry, rather than vanishing and letting the card imply
       nobody spoke on this topic. Genuinely-empty still renders nothing, as before. -->
  <section v-if="failed" class="mb-4" data-testid="topic-perspectives-error">
    <p class="text-sm text-muted">
      {{ t('section.error') }}
      <button
        type="button"
        class="ml-1 font-bold text-accent underline"
        data-testid="topic-perspectives-retry"
        @click="load()"
      >
        {{ t('section.retry') }}
      </button>
    </p>
  </section>

  <section v-else-if="perspectives.length" class="mb-4" data-testid="topic-perspectives">
    <h3 class="lp-section mb-2">
      {{ t('ec.perspectives', perspectives.length, { named: { count: perspectives.length } }) }}
    </h3>
    <ul class="flex flex-col gap-2.5">
      <li
        v-for="p in perspectives"
        :key="p.person_id"
        class="rounded-lg border border-border bg-overlay p-3"
        data-testid="topic-perspective"
      >
        <div class="flex items-baseline gap-2">
          <button
            type="button"
            class="text-sm font-bold text-person hover:underline"
            @click="emit('open', { kind: 'person', id: p.person_id })"
          >
            {{ p.person_name }}
          </button>
          <span class="lp-kicker">{{
            t('ec.perspectiveInsights', p.insight_count, { named: { count: p.insight_count } })
          }}</span>
        </div>
        <ul class="mt-1.5 flex flex-col gap-1">
          <li
            v-for="ins in expanded.has(p.person_id) ? p.insights : p.insights.slice(0, PREVIEW)"
            :key="ins.id"
            class="flex gap-1.5 text-sm text-canvas-foreground"
          >
            <span aria-hidden="true" class="text-muted">•</span>
            <span>{{ ins.text }}</span>
          </li>
        </ul>
        <button
          v-if="p.insights.length > PREVIEW"
          type="button"
          class="mt-1 text-xs font-semibold text-accent hover:underline"
          @click="toggle(p.person_id)"
        >
          {{
            expanded.has(p.person_id)
              ? t('ec.perspectiveLess')
              : t('ec.perspectiveMore', { count: p.insights.length - PREVIEW })
          }}
        </button>
      </li>
    </ul>
  </section>
</template>
