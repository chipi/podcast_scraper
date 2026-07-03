<script setup lang="ts">
/**
 * Enrichment signals for a graph node's Enrichment tab (#1128 follow-up). Topic → temporal velocity
 * + corpus co-occurrence; Person → grounding rate + guest co-appearance + contradictions. Best-effort:
 * missing envelopes are silently hidden. `nodeId` is the canonical prefixed id (topic:/person:).
 */
import { computed, ref, watch } from 'vue'
import { fetchCachedCorpusEnvelope } from '../../composables/useEnrichmentEnvelopeCache'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'
import PersonInitialAvatar from '../shared/PersonInitialAvatar.vue'

const props = defineProps<{ nodeId: string; nodeType: string }>()
// Reported to the parent card so it can hide the Enrichment tab entirely when
// this node has no signals (the common case for graph nodes on fresh corpora).
const emit = defineEmits<{ 'has-content': [boolean] }>()
const shell = useShellStore()
const subject = useSubjectStore()

const kind = () => props.nodeType.trim().toLowerCase()
const isTopic = () => kind() === 'topic'
const isPerson = () => kind() === 'person' || kind() === 'speaker'

const loaded = ref(false)

// --- topic signals ---
const velocity = ref<{ velocity: number; total: number } | null>(null)
const cooccurrence = ref<Array<{ topic_id: string; topic_label?: string; episode_count: number; lift: number }>>([])

// Rank co-occurrence by lift/PMI (co-occurs more than chance, given each topic's
// own frequency) — NOT raw count, which just surfaces the popular/obvious. Gated
// to real associations (≥2 episodes, above chance); simply empty on tiny corpora,
// which is itself the honest signal that co-occurrence hasn't earned its keep yet.
const cooccurByLift = computed(() =>
  [...cooccurrence.value]
    .filter((p) => p.episode_count >= 2 && p.lift > 1)
    .sort((a, b) => b.lift - a.lift)
    .slice(0, 8),
)

// --- person signals ---
const grounding = ref<{ grounded: number; total: number; rate: number } | null>(null)
const coappearances = ref<Array<{ person_id: string; person_name?: string; episode_count: number }>>([])
const contradictions = ref<Array<{ person_id: string; person_name?: string; topic_id: string }>>([])

function shortId(id: string): string {
  return id.replace(/^(podcast|person|topic|org):/, '').replace(/[-_]/g, ' ').trim() || id
}

function reset(): void {
  loaded.value = false
  velocity.value = null
  cooccurrence.value = []
  grounding.value = null
  coappearances.value = []
  contradictions.value = []
  emit('has-content', false)
}

function currentHasContent(): boolean {
  if (isTopic()) return velocity.value !== null || cooccurByLift.value.length > 0
  if (isPerson()) {
    return (
      grounding.value !== null ||
      coappearances.value.length > 0 ||
      contradictions.value.length > 0
    )
  }
  return false
}

async function load(): Promise<void> {
  reset()
  const root = shell.corpusPath?.trim()
  const id = props.nodeId?.trim()
  if (!root || !id) return

  if (isTopic()) {
    const [vel, co] = await Promise.all([
      fetchCachedCorpusEnvelope<{ topics: Array<{ topic_id: string; velocity_last_over_6mo: number; total: number }> }>(root, 'temporal_velocity').catch(() => null),
      fetchCachedCorpusEnvelope<{ pairs: Array<{ topic_a_id: string; topic_b_id: string; topic_a_label?: string; topic_b_label?: string; episode_count: number; lift?: number }> }>(root, 'topic_cooccurrence_corpus').catch(() => null),
    ])
    const vrow = vel?.data?.topics?.find((t) => t.topic_id === id) ?? null
    if (vrow) velocity.value = { velocity: vrow.velocity_last_over_6mo, total: vrow.total }
    if (co?.data?.pairs) {
      const partners: Array<{ topic_id: string; topic_label?: string; episode_count: number; lift: number }> = []
      for (const p of co.data.pairs) {
        if (p.topic_a_id === id) partners.push({ topic_id: p.topic_b_id, topic_label: p.topic_b_label, episode_count: p.episode_count, lift: p.lift ?? 0 })
        else if (p.topic_b_id === id) partners.push({ topic_id: p.topic_a_id, topic_label: p.topic_a_label, episode_count: p.episode_count, lift: p.lift ?? 0 })
      }
      // Store unranked; cooccurByCount (A) and cooccurByLift (B) do the ordering.
      cooccurrence.value = partners
    }
  } else if (isPerson()) {
    const [gr, co, ct] = await Promise.all([
      fetchCachedCorpusEnvelope<{ persons: Array<{ person_id: string; grounded_insights: number; total_insights: number; rate: number }> }>(root, 'grounding_rate').catch(() => null),
      fetchCachedCorpusEnvelope<{ pairs: Array<{ person_a_id: string; person_b_id: string; person_a_name?: string; person_b_name?: string; episode_count: number }> }>(root, 'guest_coappearance').catch(() => null),
      fetchCachedCorpusEnvelope<{ contradictions: Array<{ topic_id: string; person_a_id: string; person_b_id: string; person_a_name?: string; person_b_name?: string }> }>(root, 'nli_contradiction').catch(() => null),
    ])
    const grow = gr?.data?.persons?.find((p) => p.person_id === id) ?? null
    if (grow) grounding.value = { grounded: grow.grounded_insights, total: grow.total_insights, rate: grow.rate }
    if (co?.data?.pairs) {
      const out: Array<{ person_id: string; person_name?: string; episode_count: number }> = []
      for (const p of co.data.pairs) {
        if (p.person_a_id === id) out.push({ person_id: p.person_b_id, person_name: p.person_b_name, episode_count: p.episode_count })
        else if (p.person_b_id === id) out.push({ person_id: p.person_a_id, person_name: p.person_a_name, episode_count: p.episode_count })
      }
      coappearances.value = out.sort((a, b) => b.episode_count - a.episode_count).slice(0, 8)
    }
    if (ct?.data?.contradictions) {
      const out: Array<{ person_id: string; person_name?: string; topic_id: string }> = []
      for (const c of ct.data.contradictions) {
        if (c.person_a_id === id) out.push({ person_id: c.person_b_id, person_name: c.person_b_name, topic_id: c.topic_id })
        else if (c.person_b_id === id) out.push({ person_id: c.person_a_id, person_name: c.person_a_name, topic_id: c.topic_id })
      }
      contradictions.value = out.slice(0, 8)
    }
  }
  loaded.value = true
  emit('has-content', currentHasContent())
}

watch(() => props.nodeId, () => void load(), { immediate: true })
</script>

<template>
  <div class="space-y-3 text-[11px]" data-testid="node-enrichment-section">
    <p v-if="!isTopic() && !isPerson()" class="text-muted" data-testid="node-enrichment-unsupported">
      No enrichment signals for this node type.
    </p>

    <!-- Topic -->
    <template v-else-if="isTopic()">
      <div v-if="velocity" data-testid="node-enrichment-velocity">
        <p class="mb-0.5 text-[10px] font-semibold uppercase tracking-wider text-muted">Velocity (last / 6-mo avg)</p>
        <span
          class="rounded px-2 py-0.5 font-mono"
          :class="velocity.velocity > 1.5 ? 'bg-emerald-700/30 text-emerald-300' : velocity.velocity < 0.5 ? 'bg-rose-700/30 text-rose-300' : 'bg-overlay text-muted'"
        >{{ velocity.velocity.toFixed(2) }}×</span>
        <span class="ml-2 text-muted">· {{ velocity.total }} mentions / 12-mo</span>
      </div>
      <div v-if="cooccurByLift.length" data-testid="node-enrichment-cooccurrence-lift">
        <p class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">Co-occurs with · above chance</p>
        <div class="flex flex-wrap gap-1">
          <button
            v-for="r in cooccurByLift"
            :key="r.topic_id"
            type="button"
            class="rounded border border-default bg-overlay px-2 py-0.5 hover:bg-overlay-2"
            :title="`lift ${r.lift.toFixed(2)}× · ${r.episode_count} episodes`"
            @click="subject.focusTopic(r.topic_id)"
          >{{ r.topic_label || shortId(r.topic_id) }}<span class="ml-1 text-muted">·{{ r.lift.toFixed(1) }}×</span></button>
        </div>
      </div>
      <p v-if="loaded && !velocity && !cooccurByLift.length" class="text-muted">No enrichment signals for this topic.</p>
    </template>

    <!-- Person -->
    <template v-else>
      <div v-if="grounding" data-testid="node-enrichment-grounding">
        <p class="mb-0.5 text-[10px] font-semibold uppercase tracking-wider text-muted">Grounding rate</p>
        <span class="rounded bg-overlay px-2 py-0.5 font-mono">{{ Math.round(grounding.rate * 100) }}%</span>
        <span class="ml-2 text-muted">· {{ grounding.grounded }}/{{ grounding.total }} insights grounded</span>
      </div>
      <div v-if="coappearances.length" data-testid="node-enrichment-coappearance">
        <p class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">Co-appears with</p>
        <div class="flex flex-wrap gap-1">
          <button
            v-for="r in coappearances"
            :key="r.person_id"
            type="button"
            class="inline-flex items-center gap-1 rounded border border-default bg-overlay px-2 py-0.5 hover:bg-overlay-2"
            @click="subject.focusPerson(r.person_id)"
          ><PersonInitialAvatar :name="r.person_name || shortId(r.person_id)" />{{ r.person_name || shortId(r.person_id) }}<span class="ml-1 text-muted">·{{ r.episode_count }}</span></button>
        </div>
      </div>
      <div v-if="contradictions.length" data-testid="node-enrichment-contradictions">
        <p class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">Contradictions</p>
        <ul class="space-y-1">
          <li v-for="(r, i) in contradictions" :key="i" class="rounded border border-border bg-elevated/40 px-2 py-1">
            <button type="button" class="inline-flex items-center gap-1 align-middle font-semibold text-primary hover:underline" @click="subject.focusPerson(r.person_id)"><PersonInitialAvatar :name="r.person_name || shortId(r.person_id)" />{{ r.person_name || shortId(r.person_id) }}</button>
            <span class="text-muted"> on </span>
            <button type="button" class="text-surface-foreground hover:underline" @click="subject.focusTopic(r.topic_id)">{{ shortId(r.topic_id) }}</button>
          </li>
        </ul>
      </div>
      <p v-if="loaded && !grounding && !coappearances.length && !contradictions.length" class="text-muted">No enrichment signals for this person.</p>
    </template>
  </div>
</template>
