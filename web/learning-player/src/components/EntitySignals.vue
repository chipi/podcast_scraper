<script setup lang="ts">
/**
 * Entity Signals (Plan B — RFC-088 enrichment on the consumer entity card).
 * Brings the viewer's "Signals" to the player: corpus-scope enrichment for the
 * focused person / topic, read once from `/api/app/corpus/enrichment` (shared,
 * memoized). Best-effort — every section hides when its enricher didn't run, and
 * the whole block renders nothing when there's no signal. Chips emit `open` so
 * the parent card can walk the graph (person↔topic), same as its other rows.
 *
 *   Person → often-appears-with, where-they-agree (consensus). NOT grounding: that metric was
 *            per-Person until #1927 and scored exactly 1.0 for all 689 people, because an insight
 *            is grounded exactly when a supporting quote exists and the quote carries the
 *            speaker — so an ungrounded insight has no person to attribute it to. It is
 *            per-EPISODE now, and per-episode QA is an operator concern, not a listener's.
 *   Topic  → momentum (velocity). (Similar / discussed-alongside topics are shown once, on the
 *            card itself, to avoid four near-identical related-topic chip rows.)
 */
import CollapsibleSection from "./CollapsibleSection.vue"
import { computed, ref, watch } from "vue"
import { useI18n } from "vue-i18n"
import { personName, personNameFromId } from "../utils/personName"
import { getEntitySignals } from "../services/api"
import { useCappedSections } from "../composables/useCappedSections"
import ShowAllToggle from "./ShowAllToggle.vue"
import type { CorpusEnrichmentSignals } from "../services/types"

const props = defineProps<{
  kind: "person" | "topic"
  id: string
  /**
   * Render one of the two sections only, so a host can put its own sections between them (the
   * person page: "Often appears with", then related people and topics, then "Where they agree" —
   * operator 2026-10-05). Two instances share ONE request: `getEntitySignals` caches by entity.
   */
  only?: "coappears" | "consensus"
  /** The person's display name, so the agreement heading says WHO "they" are. */
  name?: string
}>()
const emit = defineEmits<{ (e: "open", payload: { kind: "person" | "topic"; id: string }): void }>()

const { t } = useI18n()

const signals = ref<CorpusEnrichmentSignals | null>(null)
watch(
  () => [props.kind, props.id] as const,
  ([kind, id]) => {
    // Per-entity lean fetch (server pre-filters the corpus lists to this entity); cached per
    // kind:id so re-opening the same card resolves instantly. Guard against a stale response
    // landing after the user has already moved to another card.
    const requested = id
    void getEntitySignals(kind, id)
      .then((s) => {
        if (props.id === requested) signals.value = s
      })
      .catch(() => {
        if (props.id === requested) signals.value = null
      })
  },
  { immediate: true }
)

const norm = (id: string): string => id.replace(/^(?:g:|k:|kg:)+/, "")
const self = computed(() => norm(props.id))
function shortId(id: string): string {
  return (
    norm(id)
      .replace(/^(?:person|topic|org):/, "")
      .replace(/[-_]+/g, " ")
      .trim() || id
  )
}
/**
 * Real name from the envelope, else a prettified slug — cased either way.
 *
 * The local `titleCase` this replaced ran ONLY on the slug fallback, so a real name arrived from
 * the envelope exactly as the pipeline normalised it: "simon wilson" sat next to a de-slugged
 * "Simon Wilson" in the same list (operator 2026-10-01). One shared formatter now covers both
 * paths, and it only ever ADDS capitalisation — see `utils/personName`.
 */
function nameOf(name: string | undefined, id: string): string {
  return name?.trim() ? personName(name) : personNameFromId(shortId(id))
}

// ── Person signals ───────────────────────────────────────────────────────────
// #1927 — no grounding row here. The server stopped sending `grounding_rate.persons` on this
// route when the enricher pivoted to per-EPISODE, so this section had already become dead UI that
// could never render; it is removed rather than left to look like a feature that "has no data".
const coappears = computed(() => {
  if (props.kind !== "person") return []
  const out: Array<{ id: string; name: string; count: number }> = []
  for (const p of signals.value?.guest_coappearance?.pairs ?? []) {
    if (norm(p.person_a_id) === self.value)
      out.push({
        id: p.person_b_id,
        name: nameOf(p.person_b_name, p.person_b_id),
        count: p.episode_count,
      })
    else if (norm(p.person_b_id) === self.value)
      out.push({
        id: p.person_a_id,
        name: nameOf(p.person_a_name, p.person_a_id),
        count: p.episode_count,
      })
  }
  return out.sort((a, b) => b.count - a.count)
})
// Cross-person corroboration on a topic (topic_consensus, ADR-108): who else makes
// the same point as this person, oriented so the focused person's claim is "self".
const consensus = computed(() => {
  if (props.kind !== "person") return []
  const out: Array<{
    otherId: string
    otherName: string
    topic: string
    selfText: string
    otherText: string
  }> = []
  for (const c of signals.value?.topic_consensus?.consensus ?? []) {
    const isA = norm(c.person_a_id) === self.value
    const isB = norm(c.person_b_id) === self.value
    if (!isA && !isB) continue
    const topic = shortId(c.topic_id)
    if (isA)
      out.push({
        otherId: c.person_b_id,
        otherName: nameOf(c.person_b_name, c.person_b_id),
        topic,
        selfText: c.insight_a_text ?? "",
        otherText: c.insight_b_text ?? "",
      })
    else
      out.push({
        otherId: c.person_a_id,
        otherName: nameOf(c.person_a_name, c.person_a_id),
        topic,
        selfText: c.insight_b_text ?? "",
        otherText: c.insight_a_text ?? "",
      })
  }
  return out
})
// Both lists page five at a time with the app's section cap (operator 2026-10-05). Each was a hard
// cap of 8 that hid the rest with no way to reach them.
const caps = useCappedSections(5, 5)
const shownCoappears = computed(() => caps.visible("coappears", coappears.value))
const shownConsensus = computed(() => caps.visible("consensus", consensus.value))

// Topic momentum moved OUT of here to the top of the entity card, under the title (operator
// review): a topic's "↑ Rising" badge now leads the card, the same idiom as the storyline sheet,
// rather than sitting mid-body under a "Momentum" heading. Similar-topics + discussed-alongside
// were removed earlier for the same reason (the card owns those chips). So this block is now
// PERSON-ONLY — co-appearance + consensus — and renders nothing for a topic.
const showCoappears = computed(() => props.only !== "consensus" && coappears.value.length > 0)
const showConsensus = computed(() => props.only !== "coappears" && consensus.value.length > 0)
const hasAny = computed(() => showCoappears.value || showConsensus.value)
</script>

<template>
  <div v-if="hasAny" :data-testid="props.only ? `entity-signals-${props.only}` : 'entity-signals'">
    <!-- Person -->
    <section v-if="showCoappears" class="mb-4" data-testid="es-coappears">
      <CollapsibleSection :title="t('ec.sigCoappears')" section-key="signals-coappears" :level="3">
        <div class="flex flex-wrap gap-1.5">
          <button
            v-for="p in shownCoappears"
            :key="p.id"
            type="button"
            class="rounded-full bg-overlay px-2.5 py-1 text-xs text-person transition hover:bg-elevated"
            @click="emit('open', { kind: 'person', id: p.id })"
          >
            {{ p.name }} <span class="text-muted">· {{ p.count }}</span>
          </button>
        </div>
        <ShowAllToggle
          v-if="caps.overflows(coappears.length, false, 'coappears')"
          :expanded="caps.remaining('coappears', coappears.length) === 0"
          :count="coappears.length"
          :remaining="caps.remaining('coappears', coappears.length)"
          data-testid="es-coappears-more"
          @toggle="caps.toggle('coappears', coappears.length)"
        />
      </CollapsibleSection>
    </section>

    <section v-if="showConsensus" class="mb-4" data-testid="es-consensus">
      <!-- Each row names the other person ONCE (operator 2026-10-05). It used to open "Bob on ai
           regulation", show this person's claim unattributed, then repeat "Bob: …" — "Where they agree"
           never said who "they" were, and the name appeared twice. Now: the heading names this
           person, the topic is the row's kicker, this person's claim is the quote (it is their page),
           and the other person is named once, on their agreeing line. -->
      <CollapsibleSection
        :title="props.name ? t('ec.sigConsensusWith', { name: props.name }) : t('ec.sigConsensus')"
        section-key="signals-consensus"
        :level="3"
      >
        <ul class="flex flex-col gap-2">
          <li
            v-for="(c, i) in shownConsensus"
            :key="i"
            class="rounded-md bg-overlay px-3 py-2"
            data-testid="es-consensus-row"
          >
            <p class="lp-kicker" data-testid="es-consensus-topic">{{ c.topic }}</p>
            <p v-if="c.selfText" class="mt-1 text-xs text-canvas-foreground">“{{ c.selfText }}”</p>
            <p class="mt-1 text-xs text-muted" data-testid="es-consensus-other">
              <button
                type="button"
                class="font-semibold text-person hover:underline"
                @click="emit('open', { kind: 'person', id: c.otherId })"
              >
                {{ c.otherName }}
              </button>
              <span>{{ " " + t("ec.sigAgrees") + (c.otherText ? ": " : "") }}</span
              ><span v-if="c.otherText">“{{ c.otherText }}”</span>
            </p>
          </li>
        </ul>
        <ShowAllToggle
          v-if="caps.overflows(consensus.length, false, 'consensus')"
          :expanded="caps.remaining('consensus', consensus.length) === 0"
          :count="consensus.length"
          :remaining="caps.remaining('consensus', consensus.length)"
          data-testid="es-consensus-more"
          @toggle="caps.toggle('consensus', consensus.length)"
        />
      </CollapsibleSection>
    </section>
  </div>
</template>
