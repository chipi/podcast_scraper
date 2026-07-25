<script setup lang="ts">
/**
 * Search v3 §S8 — Compare (2 subjects) operator panel.
 *
 * Renders inside ``ResultSetOperatorBar`` when the Compare chip is
 * active. Two-slot picker seeded from the current visible hits (persons /
 * topics / episodes / feeds discoverable in metadata). "Run compare"
 * emits ``run-compare`` with the two picker slots; the store fires
 * ``POST /api/search/compare`` (wrapping ``build_briefing_pack`` twice,
 * RFC-093). Response renders as a 2-column briefing view; the
 * deterministic judge summary is muted when either side is ungrounded.
 */
import { computed, ref, watch } from 'vue'
import type {
  CompareSubjectRef,
  SearchCompareResponse,
  SearchHit,
} from '../../api/searchApi'
import { useSearchStore } from '../../stores/search'

const search = useSearchStore()

const props = defineProps<{
  visibleHits: SearchHit[]
  compareResult: SearchCompareResponse | null
  compareLoading: boolean
  compareError: string | null
}>()

const emit = defineEmits<{
  'run-compare': [
    payload: {
      subjectA: CompareSubjectRef
      subjectB: CompareSubjectRef
      /** RFC-072 GIL v1.1 insight_type filter (``claim`` /
       *  ``recommendation`` / ``observation`` / ``question`` /
       *  ``unknown``). Empty ⇒ no filter — narrows both sides
       *  symmetrically on the server. */
      insightTypes: string[]
    },
  ]
  'clear-compare': []
}>()

/** RFC-072 GIL v1.1 insight_type filter options — mirrors the migration in
 * ``migrations/gil_kg_identity_migrations.py`` and the server's
 * ``SearchCompareRequest.insight_types`` field. */
const INSIGHT_TYPE_OPTIONS: ReadonlyArray<{ value: string; label: string }> = [
  { value: 'claim', label: 'Claim' },
  { value: 'recommendation', label: 'Recommendation' },
  { value: 'observation', label: 'Observation' },
  { value: 'question', label: 'Question' },
  { value: 'unknown', label: 'Unknown' },
]

interface DiscoveredSubject {
  key: string // unique — used as picker option value
  ref: CompareSubjectRef
  count: number // how many hits mentioned this subject (for the picker order)
}

/**
 * Walk the visible hits' metadata and surface every distinct subject the
 * user could reasonably compare on. Order: highest hit count first, then
 * alphabetical. Each entry is deduped on ``kind + id``.
 *
 * Sources per kind:
 *   * person — ``metadata.supporting_quotes[].speaker_name`` (insight tier),
 *              ``metadata.speaker_name`` / ``metadata.speaker`` (segment tier),
 *              ``metadata.person_id`` (kg_entity tier).
 *   * topic  — ``metadata.source_id`` when ``doc_type === 'kg_topic'``,
 *              ``metadata.topic_label`` when present.
 *   * episode — ``metadata.episode_id`` + ``metadata.episode_title``.
 *   * feed   — ``metadata.feed_id`` + ``metadata.feed_title``.
 */
const discoveredSubjects = computed<DiscoveredSubject[]>(() => {
  const seen = new Map<string, DiscoveredSubject>()
  const bump = (kind: CompareSubjectRef['kind'], id: string, label: string | null): void => {
    const trimmedId = id.trim()
    if (!trimmedId) return
    const key = `${kind}::${trimmedId}`
    const existing = seen.get(key)
    if (existing) {
      existing.count += 1
      if (!existing.ref.label && label) existing.ref.label = label
      return
    }
    seen.set(key, { key, ref: { kind, id: trimmedId, label: label || null }, count: 1 })
  }
  for (const hit of props.visibleHits) {
    const md = (hit.metadata ?? {}) as Record<string, unknown>
    const docType = typeof md.doc_type === 'string' ? md.doc_type : ''
    // Topic
    if (docType === 'kg_topic') {
      const src = typeof md.source_id === 'string' ? md.source_id : ''
      const label = typeof md.topic_label === 'string' ? md.topic_label : ''
      bump('topic', src, label || null)
    } else {
      const topicLabel = typeof md.topic_label === 'string' ? md.topic_label : ''
      if (topicLabel) bump('topic', topicLabel, topicLabel)
    }
    // Person
    const speaker = typeof md.speaker_name === 'string'
      ? md.speaker_name
      : typeof md.speaker === 'string'
        ? md.speaker
        : ''
    if (speaker) bump('person', speaker, speaker)
    const supporting = Array.isArray(md.supporting_quotes) ? md.supporting_quotes : []
    for (const raw of supporting) {
      const q = (raw ?? {}) as Record<string, unknown>
      const name = typeof q.speaker_name === 'string' ? q.speaker_name : ''
      if (name) bump('person', name, name)
    }
    // Episode
    const epId = typeof md.episode_id === 'string' ? md.episode_id : ''
    const epTitle = typeof md.episode_title === 'string' ? md.episode_title : ''
    if (epId) bump('episode', epId, epTitle || epId)
    // Feed / show
    const feedId = typeof md.feed_id === 'string' ? md.feed_id : ''
    const feedTitle = typeof md.feed_title === 'string' ? md.feed_title : ''
    if (feedId) bump('feed', feedId, feedTitle || feedId)
  }
  return Array.from(seen.values()).sort((a, b) => {
    if (b.count !== a.count) return b.count - a.count
    const labelA = a.ref.label || a.ref.id
    const labelB = b.ref.label || b.ref.id
    return labelA.localeCompare(labelB)
  })
})

/**
 * Search v3 §S8 — subjects pinned via the "Pin to Compare" action, in pin
 * order (pin[0] → slot A, pin[1] → slot B). A pinned subject need not be in
 * the current visible hits, so it becomes a selectable picker option in its
 * own right.
 */
const pinnedSubjects = computed<DiscoveredSubject[]>(() =>
  search.comparePins.map((r) => ({
    key: `${r.kind}::${r.id.trim()}`,
    ref: { kind: r.kind, id: r.id.trim(), label: r.label ?? null },
    count: 0,
  })),
)

/**
 * The picker's full option list: pinned subjects first (in pin order), then
 * the hit-discovered subjects, deduped on ``kind + id``. A pinned subject
 * that also appears in the visible hits adopts the discovered mention count
 * (and label) for its option row.
 */
const allSubjects = computed<DiscoveredSubject[]>(() => {
  const merged = new Map<string, DiscoveredSubject>()
  for (const s of pinnedSubjects.value) {
    if (!merged.has(s.key)) merged.set(s.key, { ...s, ref: { ...s.ref } })
  }
  for (const s of discoveredSubjects.value) {
    const existing = merged.get(s.key)
    if (existing) {
      existing.count = s.count
      if (!existing.ref.label && s.ref.label) existing.ref.label = s.ref.label
    } else {
      merged.set(s.key, s)
    }
  }
  return Array.from(merged.values())
})

const slotA = ref<string>('')
const slotB = ref<string>('')
const selectedInsightTypes = ref<string[]>([])

function toggleInsightType(value: string): void {
  const idx = selectedInsightTypes.value.indexOf(value)
  if (idx >= 0) selectedInsightTypes.value.splice(idx, 1)
  else selectedInsightTypes.value.push(value)
}

// Pin change → prefill the picker. This is the §S8 "pinning a subject
// prefills the Compare picker" behaviour: a fresh pin explicitly overrides
// any earlier auto-seed so the pinned subjects land in the slots.
watch(
  pinnedSubjects,
  (pins) => {
    if (pins.length >= 1) slotA.value = pins[0].key
    if (pins.length >= 2) slotB.value = pins[1].key
  },
  { immediate: true },
)

// New candidates → fill only the slots the user (or a pin) has NOT set, so
// the picker always has a working default without stomping a manual choice.
watch(
  allSubjects,
  (all) => {
    if (!slotA.value && all.length >= 1) slotA.value = all[0].key
    if (!slotB.value && all.length >= 2) {
      const other = all.find((s) => s.key !== slotA.value)
      slotB.value = other ? other.key : all[1].key
    }
  },
  { immediate: true },
)

function resolve(key: string): CompareSubjectRef | null {
  const found = allSubjects.value.find((s) => s.key === key)
  return found ? found.ref : null
}

const canRun = computed(() => {
  if (!slotA.value || !slotB.value) return false
  if (slotA.value === slotB.value) return false
  return true
})

function onRun(): void {
  const a = resolve(slotA.value)
  const b = resolve(slotB.value)
  if (!a || !b) return
  emit('run-compare', {
    subjectA: a,
    subjectB: b,
    insightTypes: [...selectedInsightTypes.value],
  })
}

/**
 * RFC-107 §Inherited Known Limitations — RFC-072 KL2 (Person) / KL3
 * (Topic). Alias variants can split the same real entity across the
 * corpus into distinct ``person:{slug}`` / topic ids, so a same-kind
 * Person↔Person or Topic↔Topic compare may miss the aliased twin. We
 * surface a muted hint whenever both picker slots resolve to the same
 * ``kind`` in {person, topic} — the ONLY two kinds RFC-072 flags. Feed /
 * episode / show ids are corpus-stable and don't trip the limitation.
 */
const aliasHint = computed<string | null>(() => {
  const a = resolve(slotA.value)
  const b = resolve(slotB.value)
  if (!a || !b) return null
  if (a.kind !== b.kind) return null
  if (a.kind === 'person') {
    return 'May miss aliased variants — comparing two Persons; aliased ids can split the same real speaker across the corpus.'
  }
  if (a.kind === 'topic') {
    return 'May miss aliased variants — comparing two Topics; aliased ids can split the same real topic across the corpus.'
  }
  return null
})

function onClear(): void {
  emit('clear-compare')
}

function optionLabel(subject: DiscoveredSubject): string {
  const label = subject.ref.label || subject.ref.id
  // count 0 ⇒ pinned-only (not present in the visible hits) — say "pinned"
  // rather than showing a misleading "— 0" mention count.
  const suffix = subject.count > 0 ? `— ${subject.count}` : 'pinned'
  return `${label} (${subject.ref.kind}) ${suffix}`
}
</script>

<template>
  <div
    class="rounded border border-border bg-canvas p-2"
    data-testid="operator-compare-panel"
    aria-label="Compare two subjects from the current hit set"
  >
    <div
      class="flex flex-wrap items-end gap-2 text-[11px] text-surface-foreground"
      data-testid="operator-compare-picker"
    >
      <label class="flex flex-col gap-0.5">
        <span class="text-[10px] uppercase tracking-wide text-muted">Subject A</span>
        <select
          v-model="slotA"
          class="rounded border border-border bg-surface px-1.5 py-0.5 text-[11px]"
          data-testid="operator-compare-slot-a"
        >
          <option
            v-for="s in allSubjects"
            :key="s.key"
            :value="s.key"
          >
            {{ optionLabel(s) }}
          </option>
        </select>
      </label>
      <label class="flex flex-col gap-0.5">
        <span class="text-[10px] uppercase tracking-wide text-muted">Subject B</span>
        <select
          v-model="slotB"
          class="rounded border border-border bg-surface px-1.5 py-0.5 text-[11px]"
          data-testid="operator-compare-slot-b"
        >
          <option
            v-for="s in allSubjects"
            :key="s.key"
            :value="s.key"
          >
            {{ optionLabel(s) }}
          </option>
        </select>
      </label>
      <button
        type="button"
        class="rounded border border-primary bg-primary px-2 py-0.5 text-[10px] font-medium leading-none text-primary-foreground transition-colors hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
        :disabled="!canRun || compareLoading"
        data-testid="operator-compare-run"
        @click="onRun"
      >
        {{ compareLoading ? 'Comparing…' : 'Run compare' }}
      </button>
      <button
        v-if="compareResult"
        type="button"
        class="rounded border border-border px-2 py-0.5 text-[10px] font-medium leading-none text-muted transition-colors hover:bg-overlay"
        data-testid="operator-compare-clear"
        @click="onClear"
      >
        Clear
      </button>
    </div>

    <div
      class="mt-2 flex flex-wrap items-center gap-1"
      data-testid="operator-compare-insight-types"
      aria-label="Insight type filter (both subjects narrowed symmetrically)"
    >
      <span class="text-[10px] uppercase tracking-wide text-muted">Insight types</span>
      <button
        v-for="opt in INSIGHT_TYPE_OPTIONS"
        :key="opt.value"
        type="button"
        class="rounded border px-1.5 py-0.5 text-[10px] font-medium leading-none transition-colors"
        :class="
          selectedInsightTypes.includes(opt.value)
            ? 'border-primary bg-primary text-primary-foreground'
            : 'border-border text-muted hover:bg-overlay'
        "
        :aria-pressed="selectedInsightTypes.includes(opt.value)"
        :data-testid="`operator-compare-insight-type-${opt.value}`"
        @click="toggleInsightType(opt.value)"
      >
        {{ opt.label }}
      </button>
      <span
        v-if="!selectedInsightTypes.length"
        class="text-[10px] italic text-muted"
      >
        (none — no filter)
      </span>
    </div>

    <p
      v-if="aliasHint"
      class="mt-1 text-[10px] italic text-muted"
      data-testid="operator-compare-alias-hint"
      role="note"
    >
      {{ aliasHint }}
    </p>

    <p
      v-if="allSubjects.length < 2"
      class="mt-1 text-[10px] text-muted"
      data-testid="operator-compare-empty"
    >
      Fewer than 2 comparable subjects in the current hit set.
    </p>

    <p
      v-if="compareError"
      class="mt-1 text-[10px] text-danger"
      data-testid="operator-compare-error"
    >
      {{ compareError }}
    </p>

    <div
      v-if="compareResult && !compareError"
      class="mt-2 grid gap-2 sm:grid-cols-2"
      data-testid="operator-compare-columns"
    >
      <section
        class="rounded border border-border/60 bg-surface p-2"
        data-testid="operator-compare-pack-a"
        aria-label="Compare pack A"
      >
        <p class="mb-1 flex items-center gap-1.5 text-[11px] text-surface-foreground">
          <span class="rounded bg-primary/15 px-1 py-px text-[9px] font-medium uppercase leading-none tracking-wide text-primary">A · {{ compareResult.pack_a.subject.kind }}</span>
          <span class="truncate font-medium">
            {{ compareResult.pack_a.subject.label ?? compareResult.pack_a.subject.id }}
          </span>
          <span
            v-if="!compareResult.pack_a.grounded"
            class="ml-auto shrink-0 rounded bg-overlay px-1 py-px text-[9px] uppercase leading-none tracking-wide text-muted"
            data-testid="operator-compare-pack-a-ungrounded"
            title="No supporting insight tier hit — pack rendered as ungrounded"
          >Ungrounded</span>
        </p>
        <p
          v-if="compareResult.pack_a.top_insight_text"
          class="mb-1 text-[11px] leading-snug text-surface-foreground"
          data-testid="operator-compare-pack-a-top-insight"
        >
          {{ compareResult.pack_a.top_insight_text }}
        </p>
        <p class="text-[10px] text-muted">
          {{ compareResult.pack_a.result_count }} hits · confidence
          {{ compareResult.pack_a.confidence_p50.toFixed(2) }}
        </p>
      </section>
      <section
        class="rounded border border-border/60 bg-surface p-2"
        data-testid="operator-compare-pack-b"
        aria-label="Compare pack B"
      >
        <p class="mb-1 flex items-center gap-1.5 text-[11px] text-surface-foreground">
          <span class="rounded bg-primary/15 px-1 py-px text-[9px] font-medium uppercase leading-none tracking-wide text-primary">B · {{ compareResult.pack_b.subject.kind }}</span>
          <span class="truncate font-medium">
            {{ compareResult.pack_b.subject.label ?? compareResult.pack_b.subject.id }}
          </span>
          <span
            v-if="!compareResult.pack_b.grounded"
            class="ml-auto shrink-0 rounded bg-overlay px-1 py-px text-[9px] uppercase leading-none tracking-wide text-muted"
            data-testid="operator-compare-pack-b-ungrounded"
            title="No supporting insight tier hit — pack rendered as ungrounded"
          >Ungrounded</span>
        </p>
        <p
          v-if="compareResult.pack_b.top_insight_text"
          class="mb-1 text-[11px] leading-snug text-surface-foreground"
          data-testid="operator-compare-pack-b-top-insight"
        >
          {{ compareResult.pack_b.top_insight_text }}
        </p>
        <p class="text-[10px] text-muted">
          {{ compareResult.pack_b.result_count }} hits · confidence
          {{ compareResult.pack_b.confidence_p50.toFixed(2) }}
        </p>
      </section>
    </div>

    <p
      v-if="compareResult && compareResult.judge_summary"
      class="mt-2 rounded border border-border/60 bg-surface px-2 py-1 text-[11px] italic text-surface-foreground"
      data-testid="operator-compare-judge"
      aria-label="Judge summary"
    >
      {{ compareResult.judge_summary }}
    </p>
  </div>
</template>
