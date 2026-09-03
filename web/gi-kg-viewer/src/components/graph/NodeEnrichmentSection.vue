<script setup lang="ts">
/**
 * Enrichment signals for a graph node's Enrichment tab (#1128 follow-up). Topic → temporal velocity
 * + corpus co-occurrence; Person → guest co-appearance + consensus (grounding is
 * per-EPISODE since #1927 and shown on the Show rail, not on a person card)
 * (topic_consensus, ADR-108). Best-effort: missing envelopes are silently hidden.
 * `nodeId` is the canonical prefixed id (topic:/person:).
 */
import { computed, ref, watch } from 'vue'
import { getCorpusEntitySignals } from '../../api/enrichmentApi'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'
import { titleCaseWords } from '../../utils/nameCase'
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
const cooccurrence = ref<
  Array<{
    topic_id: string
    topic_label?: string
    episode_count: number
    lift: number
    npmi?: number | null
  }>
>([])

// Rank co-occurrence by association strength — NOT raw count, which just surfaces the
// popular/obvious. Gated to real associations (≥2 episodes, above chance); simply empty on tiny
// corpora, which is itself the honest signal that co-occurrence hasn't earned its keep yet.
//
// #1928 — rank on ``npmi`` where the envelope carries it, falling back to ``lift``. Both say
// "more than chance", but lift is UNBOUNDED and rewards rarity by construction: on the
// 1,066-episode corpus 99.4% of pairs co-occurred in exactly one episode, and lift's median, p90
// and max were all 1066 (what ``N / (1 x 1)`` evaluates to). Maximum-possible lift was also modal
// lift, so ranking by it put the thinnest evidence first. NPMI is bounded to [-1, 1] and
// compresses what lift exaggerates, so the values are comparable enough to mix with counts.
//
// NPMI does NOT, on its own, make the ordering reflect strength rather than rarity — an earlier
// version of this comment claimed it did. For two topics that appear only together
// (df_a === df_b === episode_count) NPMI is exactly 1.0, its maximum, so a coincidence still
// outranks a genuine link. That is fixed in the ENRICHER, not here:
// `require_independent_recurrence` drops those pairs before they are ever scored. This ranking
// is only correct because that filter runs upstream.
function assoc(p: { lift: number; npmi?: number | null }): number {
  return typeof p.npmi === 'number' ? p.npmi : p.lift
}
const cooccurByLift = computed(() =>
  [...cooccurrence.value]
    .filter((p) => p.episode_count >= 2 && p.lift > 1)
    .sort((a, b) => assoc(b) - assoc(a))
    .slice(0, 8),
)

// --- person signals ---
const coappearances = ref<Array<{ person_id: string; person_name?: string; episode_count: number }>>([])
// Consensus (ADR-108) — each row carries the two corroborating claims (oriented
// to the focused person: ``selfText`` is their statement, ``otherText`` the
// counterpart's) so the panel shows *what* they agree on, not just *who*.
const consensus = ref<
  Array<{
    person_id: string
    person_name?: string
    topic_id: string
    selfName?: string
    selfText?: string
    otherText?: string
  }>
>([])

function shortId(id: string): string {
  return id.replace(/^(podcast|person|topic|org):/, '').replace(/[-_]/g, ' ').trim() || id
}

function reset(): void {
  loaded.value = false
  velocity.value = null
  cooccurrence.value = []
  coappearances.value = []
  consensus.value = []
  emit('has-content', false)
}

function currentHasContent(): boolean {
  if (isTopic()) return velocity.value !== null || cooccurByLift.value.length > 0
  if (isPerson()) {
    return (
      coappearances.value.length > 0 ||
      consensus.value.length > 0
    )
  }
  return false
}

async function load(): Promise<void> {
  reset()
  const root = shell.corpusPath?.trim()
  const id = props.nodeId?.trim()
  if (!root || !id) return

  // One lean fetch: the server pre-filters every corpus enricher to the rows touching THIS entity,
  // so the node panel pulls a few KB instead of the whole (multi-MB) co-occurrence / co-appearance
  // envelope. The graph overlays (GraphCanvas) still read the full envelopes — they draw every edge.
  const entityKind = isTopic() ? 'topic' : isPerson() ? 'person' : null
  if (!entityKind) return
  const signals = await getCorpusEntitySignals(root, entityKind, id).catch(() => null)
  if (!signals) {
    loaded.value = true
    emit('has-content', false)
    return
  }

  if (isTopic()) {
    const vrow = signals.temporal_velocity?.topics?.find((t) => t.topic_id === id) ?? null
    if (vrow) velocity.value = { velocity: vrow.velocity_last_over_6mo, total: vrow.total }
    const pairs = signals.topic_cooccurrence_corpus?.pairs
    if (pairs) {
      const partners: Array<{
        topic_id: string
        topic_label?: string
        episode_count: number
        lift: number
        npmi?: number | null
      }> = []
      for (const p of pairs) {
        if (p.topic_a_id === id) partners.push({ topic_id: p.topic_b_id, topic_label: p.topic_b_label, episode_count: p.episode_count, lift: p.lift ?? 0, npmi: p.npmi })
        else if (p.topic_b_id === id) partners.push({ topic_id: p.topic_a_id, topic_label: p.topic_a_label, episode_count: p.episode_count, lift: p.lift ?? 0, npmi: p.npmi })
      }
      // Store unranked; cooccurByCount (A) and cooccurByLift (B) do the ordering.
      cooccurrence.value = partners
    }
  } else if (isPerson()) {
    // #1927: no per-person grounding. The metric was per-Person and scored exactly 1.0 for all
    // 689 people, because an insight is grounded exactly when a supporting quote exists and the
    // quote carries the speaker — so an ungrounded insight has no speaker to attribute it to and
    // the denominator could only ever equal the numerator. It is per-EPISODE now (see the Show
    // rail), and there is nothing meaningful to put on a person card.
    const coPairs = signals.guest_coappearance?.pairs
    if (coPairs) {
      const out: Array<{ person_id: string; person_name?: string; episode_count: number }> = []
      for (const p of coPairs) {
        if (p.person_a_id === id) out.push({ person_id: p.person_b_id, person_name: p.person_b_name, episode_count: p.episode_count })
        else if (p.person_b_id === id) out.push({ person_id: p.person_a_id, person_name: p.person_a_name, episode_count: p.episode_count })
      }
      coappearances.value = out.sort((a, b) => b.episode_count - a.episode_count).slice(0, 8)
    }
    const consensusRows = signals.topic_consensus?.consensus
    if (consensusRows) {
      const out: typeof consensus.value = []
      for (const c of consensusRows) {
        if (c.person_a_id === id) {
          out.push({
            person_id: c.person_b_id,
            person_name: c.person_b_name,
            topic_id: c.topic_id,
            selfName: c.person_a_name,
            selfText: c.insight_a_text,
            otherText: c.insight_b_text,
          })
        } else if (c.person_b_id === id) {
          out.push({
            person_id: c.person_a_id,
            person_name: c.person_a_name,
            topic_id: c.topic_id,
            selfName: c.person_b_name,
            selfText: c.insight_b_text,
            otherText: c.insight_a_text,
          })
        }
      }
      consensus.value = out.slice(0, 8)
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
            :title="
              (typeof r.npmi === 'number' ? `npmi ${r.npmi.toFixed(2)} · ` : '') +
              `lift ${r.lift.toFixed(2)}× · ${r.episode_count} episodes`
            "
            @click="subject.focusTopic(r.topic_id)"
          >{{ r.topic_label || shortId(r.topic_id) }}<span class="ml-1 text-muted">·{{ r.lift.toFixed(1) }}×</span></button>
        </div>
      </div>
      <p v-if="loaded && !velocity && !cooccurByLift.length" class="text-muted">No enrichment signals for this topic.</p>
    </template>

    <!-- Person -->
    <template v-else>
      <div v-if="coappearances.length" data-testid="node-enrichment-coappearance">
        <p class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">Co-appears with</p>
        <div class="flex flex-wrap gap-1">
          <button
            v-for="r in coappearances"
            :key="r.person_id"
            type="button"
            class="inline-flex items-center gap-1 rounded border border-default bg-overlay px-2 py-0.5 hover:bg-overlay-2"
            @click="subject.focusPerson(r.person_id)"
          ><PersonInitialAvatar :name="r.person_name || shortId(r.person_id)" />{{ titleCaseWords(r.person_name || shortId(r.person_id)) }}<span class="ml-1 text-muted">·{{ r.episode_count }}</span></button>
        </div>
      </div>
      <div v-if="consensus.length" data-testid="node-enrichment-consensus">
        <p class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">Consensus</p>
        <ul class="space-y-1">
          <li v-for="(r, i) in consensus" :key="i" class="rounded border border-emerald-700/40 bg-emerald-900/10 px-2 py-1">
            <button type="button" class="inline-flex items-center gap-1 align-middle font-semibold text-primary hover:underline" @click="subject.focusPerson(r.person_id)"><PersonInitialAvatar :name="r.person_name || shortId(r.person_id)" />{{ titleCaseWords(r.person_name || shortId(r.person_id)) }}</button>
            <span class="text-muted"> on </span>
            <button type="button" class="text-surface-foreground hover:underline" :title="`Open ${titleCaseWords(shortId(r.topic_id).replace(/[-_]+/g, ' '))} — see both takes under Key voices`" @click="subject.focusTopic(r.topic_id)">{{ titleCaseWords(shortId(r.topic_id).replace(/[-_]+/g, ' ')) }}</button>
            <!-- The two corroborating claims, so it's clear *what* they agree on. -->
            <div
              v-if="r.selfText || r.otherText"
              class="mt-1 space-y-1"
              data-testid="node-enrichment-consensus-claims"
            >
              <p v-if="r.selfText" class="text-[10px] leading-snug text-muted">
                <span class="font-medium text-surface-foreground">{{ titleCaseWords(r.selfName || shortId(props.nodeId)) }}:</span>
                <span class="line-clamp-3">“{{ r.selfText }}”</span>
              </p>
              <p v-if="r.otherText" class="text-[10px] leading-snug text-muted">
                <span class="font-medium text-surface-foreground">{{ titleCaseWords(r.person_name || shortId(r.person_id)) }}:</span>
                <span class="line-clamp-3">“{{ r.otherText }}”</span>
              </p>
            </div>
          </li>
        </ul>
      </div>
      <p v-if="loaded && !coappearances.length && !consensus.length" class="text-muted">No enrichment signals for this person.</p>
    </template>
  </div>
</template>
