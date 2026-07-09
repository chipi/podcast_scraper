<script setup lang="ts">
/**
 * Entity Signals (Plan B — RFC-088 enrichment on the consumer entity card).
 * Brings the viewer's "Signals" to the player: corpus-scope enrichment for the
 * focused person / topic, read once from `/api/app/corpus/enrichment` (shared,
 * memoized). Best-effort — every section hides when its enricher didn't run, and
 * the whole block renders nothing when there's no signal. Chips emit `open` so
 * the parent card can walk the graph (person↔topic), same as its other rows.
 *
 *   Person → grounding %, often-appears-with, where-they-agree (consensus).
 *   Topic  → momentum (velocity), similar topics, often-discussed-alongside.
 */
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { getCorpusEnrichment } from '../services/api'
import type { CorpusEnrichmentSignals } from '../services/types'

const props = defineProps<{ kind: 'person' | 'topic'; id: string }>()
const emit = defineEmits<{ (e: 'open', payload: { kind: 'person' | 'topic'; id: string }): void }>()

const { t } = useI18n()

const signals = ref<CorpusEnrichmentSignals | null>(null)
watch(
  () => props.id,
  () => {
    // Cached after the first card; subsequent cards resolve instantly.
    void getCorpusEnrichment()
      .then((s) => {
        signals.value = s
      })
      .catch(() => {
        signals.value = null
      })
  },
  { immediate: true },
)

const MAX = 8
const norm = (id: string): string => id.replace(/^(?:g:|k:|kg:)+/, '')
const self = computed(() => norm(props.id))
function shortId(id: string): string {
  return norm(id).replace(/^(?:person|topic|org):/, '').replace(/[-_]+/g, ' ').trim() || id
}
function titleCase(s: string): string {
  return s.replace(/(^|[^\p{L}\p{N}])(\p{L})/gu, (_m, sep, ch) => sep + ch.toUpperCase())
}
/** Real name from the envelope, else a prettified slug. */
function nameOf(name: string | undefined, id: string): string {
  return name?.trim() ? name.trim() : titleCase(shortId(id))
}

// ── Person signals ───────────────────────────────────────────────────────────
const grounding = computed(() => {
  if (props.kind !== 'person') return null
  const row = (signals.value?.grounding_rate?.persons ?? []).find((p) => norm(p.person_id) === self.value)
  if (!row || !row.total_insights) return null
  return { pct: Math.round((row.rate ?? 0) * 100), grounded: row.grounded_insights, total: row.total_insights }
})
const coappears = computed(() => {
  if (props.kind !== 'person') return []
  const out: Array<{ id: string; name: string; count: number }> = []
  for (const p of signals.value?.guest_coappearance?.pairs ?? []) {
    if (norm(p.person_a_id) === self.value) out.push({ id: p.person_b_id, name: nameOf(p.person_b_name, p.person_b_id), count: p.episode_count })
    else if (norm(p.person_b_id) === self.value) out.push({ id: p.person_a_id, name: nameOf(p.person_a_name, p.person_a_id), count: p.episode_count })
  }
  return out.sort((a, b) => b.count - a.count).slice(0, MAX)
})
// Cross-person corroboration on a topic (topic_consensus, ADR-108): who else makes
// the same point as this person, oriented so the focused person's claim is "self".
const consensus = computed(() => {
  if (props.kind !== 'person') return []
  const out: Array<{ otherId: string; otherName: string; topic: string; selfText: string; otherText: string }> = []
  for (const c of signals.value?.topic_consensus?.consensus ?? []) {
    const isA = norm(c.person_a_id) === self.value
    const isB = norm(c.person_b_id) === self.value
    if (!isA && !isB) continue
    const topic = shortId(c.topic_id)
    if (isA) out.push({ otherId: c.person_b_id, otherName: nameOf(c.person_b_name, c.person_b_id), topic, selfText: c.insight_a_text ?? '', otherText: c.insight_b_text ?? '' })
    else out.push({ otherId: c.person_a_id, otherName: nameOf(c.person_a_name, c.person_a_id), topic, selfText: c.insight_b_text ?? '', otherText: c.insight_a_text ?? '' })
  }
  return out.slice(0, MAX)
})

// ── Topic signals ────────────────────────────────────────────────────────────
const momentum = computed(() => {
  if (props.kind !== 'topic') return null
  const row = (signals.value?.temporal_velocity?.topics ?? []).find((x) => norm(x.topic_id) === self.value)
  const v = row?.velocity_last_over_6mo
  // Only surface genuine upward momentum ("heating up") — steady/cooling is noise
  // to a consumer and reads oddly on a sparse sample (e.g. "Cooling · 0×").
  if (v == null || v < 1.5) return null
  return { v: Math.round(v * 10) / 10, total: row?.total ?? 0 }
})
const similarTopics = computed(() => {
  if (props.kind !== 'topic') return []
  const row = (signals.value?.topic_similarity?.topics ?? []).find((x) => norm(x.topic_id) === self.value)
  return (row?.top_k ?? [])
    .map((k) => ({ id: k.topic_id, label: k.topic_label?.trim() || shortId(k.topic_id) }))
    .slice(0, MAX)
})
const alongside = computed(() => {
  if (props.kind !== 'topic') return []
  const out: Array<{ id: string; label: string; lift: number }> = []
  for (const p of signals.value?.topic_cooccurrence_corpus?.pairs ?? []) {
    if ((p.episode_count ?? 0) < 2 || (p.lift ?? 0) <= 1) continue
    if (norm(p.topic_a_id) === self.value) out.push({ id: p.topic_b_id, label: p.topic_b_label?.trim() || shortId(p.topic_b_id), lift: p.lift ?? 0 })
    else if (norm(p.topic_b_id) === self.value) out.push({ id: p.topic_a_id, label: p.topic_a_label?.trim() || shortId(p.topic_a_id), lift: p.lift ?? 0 })
  }
  return out.sort((a, b) => b.lift - a.lift).slice(0, MAX)
})

const hasAny = computed(() =>
  Boolean(
    grounding.value ||
      coappears.value.length ||
      consensus.value.length ||
      momentum.value ||
      similarTopics.value.length ||
      alongside.value.length,
  ),
)
</script>

<template>
  <div v-if="hasAny" data-testid="entity-signals">
    <!-- Person -->
    <section v-if="grounding" class="mb-4" data-testid="es-grounding">
      <h3 class="lp-section mb-2">{{ t('ec.sigGrounding') }}</h3>
      <p class="text-sm text-canvas-foreground">
        {{ t('ec.sigGroundedLine', { grounded: grounding.grounded, total: grounding.total, pct: grounding.pct }) }}
      </p>
    </section>

    <section v-if="coappears.length" class="mb-4" data-testid="es-coappears">
      <h3 class="lp-section mb-2">{{ t('ec.sigCoappears') }}</h3>
      <div class="flex flex-wrap gap-1.5">
        <button
          v-for="p in coappears"
          :key="p.id"
          type="button"
          class="rounded-full bg-overlay px-2.5 py-1 text-xs text-person transition hover:bg-elevated"
          @click="emit('open', { kind: 'person', id: p.id })"
        >{{ p.name }} <span class="text-muted">· {{ p.count }}</span></button>
      </div>
    </section>

    <section v-if="consensus.length" class="mb-4" data-testid="es-consensus">
      <h3 class="lp-section mb-2">{{ t('ec.sigConsensus') }}</h3>
      <ul class="flex flex-col gap-2">
        <li
          v-for="(c, i) in consensus"
          :key="i"
          class="rounded-md bg-overlay px-3 py-2"
          data-testid="es-consensus-row"
        >
          <p class="text-xs">
            <button
              type="button"
              class="font-semibold text-person hover:underline"
              @click="emit('open', { kind: 'person', id: c.otherId })"
            >{{ c.otherName }}</button>
            <span class="text-muted">{{ ' ' + t('ec.sigOn', { topic: c.topic }) }}</span>
          </p>
          <p v-if="c.selfText" class="mt-1 text-xs text-muted"><span class="text-canvas-foreground">“{{ c.selfText }}”</span></p>
          <p v-if="c.otherText" class="mt-0.5 text-xs text-muted">{{ c.otherName }}: “{{ c.otherText }}”</p>
        </li>
      </ul>
    </section>

    <!-- Topic -->
    <section v-if="momentum" class="mb-4" data-testid="es-momentum">
      <h3 class="lp-section mb-2">{{ t('ec.sigMomentum') }}</h3>
      <p class="text-sm">
        <span class="rounded-full bg-accent/20 px-2 py-0.5 text-xs font-semibold text-accent">
          {{ t('ec.sig_rising') }} · {{ momentum.v }}× {{ t('ec.sigVsAvg') }}
        </span>
      </p>
    </section>

    <section v-if="similarTopics.length" class="mb-4" data-testid="es-similar">
      <h3 class="lp-section mb-2">{{ t('ec.sigSimilar') }}</h3>
      <div class="flex flex-wrap gap-1.5">
        <button
          v-for="tp in similarTopics"
          :key="tp.id"
          type="button"
          class="rounded-full bg-overlay px-2.5 py-1 text-xs text-topic transition hover:bg-elevated"
          @click="emit('open', { kind: 'topic', id: tp.id })"
        >{{ tp.label }}</button>
      </div>
    </section>

    <section v-if="alongside.length" class="mb-4" data-testid="es-alongside">
      <h3 class="lp-section mb-2">{{ t('ec.sigAlongside') }}</h3>
      <div class="flex flex-wrap gap-1.5">
        <button
          v-for="tp in alongside"
          :key="tp.id"
          type="button"
          class="rounded-full bg-overlay px-2.5 py-1 text-xs text-topic transition hover:bg-elevated"
          @click="emit('open', { kind: 'topic', id: tp.id })"
        >{{ tp.label }}</button>
      </div>
    </section>
  </div>
</template>
