<script setup lang="ts">
/**
 * #672 — Topic / Entity rail panel ("TEV"). Reads the focused subject id
 * from the subject store, looks it up in the merged GI/KG slice, and shows
 * name + aliases + a monthly mentions timeline + the linked insights/quotes.
 *
 * Mounts inside ``SubjectRail`` for ``subject.kind === 'topic'``. Entity
 * subjects still flow through ``focusGraphNode`` → existing ``NodeDetail``.
 *
 * When ``embedded`` is true (folded into NodeDetail's Details tab for topic
 * nodes), the header (kind + name) and footer action buttons are hidden so
 * NodeDetail owns the chrome. ``subjectIdOverride`` lets the caller supply the
 * node id directly instead of relying on the subject store.
 */
import { computed, ref, watch } from 'vue'
import type { RawGraphNode } from '../../types/artifact'
import { useArtifactsStore } from '../../stores/artifacts'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'
import {
  fetchCrossShow,
  fetchRelatedTopics,
  fetchTopicEntities,
  fetchWhoSaid,
  type RelatedNode,
} from '../../api/relationalApi'
import { StaleGeneration } from '../../utils/staleGeneration'
import { findRawNodeInArtifactByIdOrPrefixed } from '../../utils/parsing'
import { stripLayerPrefixesForCil } from '../../utils/mergeGiKg'
import { titleCaseWords } from '../../utils/nameCase'
import { fetchCorpusFeeds } from '../../api/corpusLibraryApi'
import PersonInitialAvatar from '../shared/PersonInitialAvatar.vue'
import ShowGlyph from '../shared/ShowGlyph.vue'
import PodcastCover from '../shared/PodcastCover.vue'
import HelpTip from '../shared/HelpTip.vue'



const props = withDefaults(
  defineProps<{ embedded?: boolean; subjectIdOverride?: string }>(),
  { embedded: false, subjectIdOverride: '' },
)

const emit = defineEmits<{
  goGraph: []
  closeSubject: []
  openLibraryEpisode: [{ metadata_relative_path: string }]
  prefillSemanticSearch: [{ query: string }]
}>()

const artifacts = useArtifactsStore()
const shell = useShellStore()
const subject = useSubjectStore()

const subjectId = computed(() => props.subjectIdOverride?.trim() || subject.topicId?.trim() || '')

/**
 * PRD-033 FR4.2 — cross-show coverage + key voices for this topic, from the
 * relational-query layer (RFC-094). Fetched async on subject change, skeleton-first
 * (RFC-094 OQ-3); a StaleGeneration gate drops responses for a superseded subject.
 */
interface CrossShowRow {
  showId: string
  insight: RelatedNode
}
interface VoiceRow {
  personId: string
  insights: RelatedNode[]
}
const crossShowLoading = ref(false)
const crossShowError = ref<string | null>(null)
const crossShowRows = ref<CrossShowRow[]>([])
const voicesLoading = ref(false)
const voicesError = ref<string | null>(null)
const voiceRows = ref<VoiceRow[]>([])
const entitiesLoading = ref(false)
const entityRows = ref<RelatedNode[]>([])
const relatedTopicRows = ref<RelatedNode[]>([]) // #1055 — topics that share insights
const relationalGate = new StaleGeneration()

function shortId(id: string): string {
  return id.replace(/^(podcast|person|topic|org):/, '').replace(/[-_]/g, ' ').trim() || id
}

function resetRelational(): void {
  crossShowRows.value = []
  crossShowError.value = null
  voiceRows.value = []
  voicesError.value = null
  entityRows.value = []
  relatedTopicRows.value = []
}

async function loadRelational(topicId: string): Promise<void> {
  const root = shell.corpusPath?.trim()
  if (!topicId || !root || !shell.healthStatus) {
    resetRelational()
    return
  }
  const seq = relationalGate.bump()
  crossShowLoading.value = true
  voicesLoading.value = true
  entitiesLoading.value = true
  resetRelational()
  // Relational endpoints canonicalize on the bare corpus id (topic:…); strip the
  // graph layer prefix (g:/k:) first or they return empty — NodeDetail passes the
  // prefixed node id, so without this every topic's sections are blank (same
  // id-format mismatch fixed for the person Connections rail).
  const canonical = stripLayerPrefixesForCil(topicId)
  try {
    const [cross, who, ents, related] = await Promise.all([
      fetchCrossShow(root, canonical).catch((e) => ({ error: String(e?.message ?? e), groups: {} })),
      fetchWhoSaid(root, canonical).catch((e) => ({ error: String(e?.message ?? e), groups: {} })),
      fetchTopicEntities(root, canonical).catch(() => ({ results: [] as RelatedNode[] })),
      fetchRelatedTopics(root, canonical).catch(() => ({ results: [] as RelatedNode[] })),
    ])
    if (relationalGate.isStale(seq)) return
    relatedTopicRows.value = related.results ?? []
    crossShowError.value = cross.error ?? null
    crossShowRows.value = Object.entries(cross.groups ?? {})
      .map(([showId, insights]) => ({ showId, insight: insights[0] }))
      .filter((r): r is CrossShowRow => r.insight != null)
    voicesError.value = who.error ?? null
    voiceRows.value = Object.entries(who.groups ?? {})
      .map(([personId, insights]) => ({ personId, insights }))
      .filter((r) => r.insights.length > 0)
    entityRows.value = ents.results ?? []
  } finally {
    if (relationalGate.isCurrent(seq)) {
      crossShowLoading.value = false
      voicesLoading.value = false
      entitiesLoading.value = false
    }
  }
}

watch(subjectId, (id) => void loadRelational(id), { immediate: true })

// Show cover art for Across-shows rows (mirror the Timeline episode rows). The
// cross-show showId is a podcast slug (podcast:planet-money); feeds key on
// display_title, so match on the normalised title.
interface ShowCover {
  imageUrl: string | null
  imageLocalRelpath: string | null
}
const showCoverByTitle = ref<Map<string, ShowCover>>(new Map())

function normTitle(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim()
}

async function loadFeedCovers(): Promise<void> {
  const root = shell.corpusPath?.trim()
  if (!root || !shell.healthStatus) {
    showCoverByTitle.value = new Map()
    return
  }
  try {
    const body = await fetchCorpusFeeds(root)
    const m = new Map<string, ShowCover>()
    for (const f of body.feeds ?? []) {
      const t = f.display_title ? normTitle(f.display_title) : ''
      if (t) m.set(t, { imageUrl: f.image_url ?? null, imageLocalRelpath: f.image_local_relpath ?? null })
    }
    showCoverByTitle.value = m
  } catch {
    showCoverByTitle.value = new Map()
  }
}
watch(() => shell.corpusPath, () => void loadFeedCovers(), { immediate: true })

function feedCoverFor(showId: string): ShowCover | null {
  return showCoverByTitle.value.get(normTitle(shortId(showId))) ?? null
}

function onClickVoice(personId: string): void {
  if (personId) subject.focusPerson(personId)
}

function onClickEntity(id: string, type: string): void {
  if (!id) return
  if (type === 'person') subject.focusPerson(id)
  else subject.focusEntity(id)
}

// #7 — collapse the entity MENTIONS into one chip per entity, tallying the distinct SHOWS and
// episodes each spans, so the topic view surfaces which entities recur across shows (not a chip
// per mention). Sorted by cross-show reach.
interface EntityGroup {
  id: string
  text: string
  type: string
  shows: number
  episodes: number
}
const entityGroups = computed<EntityGroup[]>(() => {
  const byId = new Map<
    string,
    { text: string; type: string; shows: Set<string>; eps: Set<string> }
  >()
  for (const r of entityRows.value) {
    const key = (r.id || r.text || '').trim()
    if (!key) continue
    const g = byId.get(key) ?? {
      text: (r.text || shortId(r.id)).trim(),
      type: r.type,
      shows: new Set<string>(),
      eps: new Set<string>(),
    }
    if (r.show_id) g.shows.add(r.show_id)
    if (r.episode_id) g.eps.add(r.episode_id)
    byId.set(key, g)
  }
  return [...byId.entries()]
    .map(([id, g]) => ({ id, text: g.text, type: g.type, shows: g.shows.size, episodes: g.eps.size }))
    .sort((a, b) => b.shows - a.shows || b.episodes - a.episodes || a.text.localeCompare(b.text))
})

const subjectNode = computed<RawGraphNode | null>(() => {
  const art = artifacts.displayArtifact
  const id = subjectId.value
  if (!art || !id) return null
  return findRawNodeInArtifactByIdOrPrefixed(art, id)
})

const subjectKindLabel = computed(() => {
  const t = subjectNode.value?.type
  if (t === 'Topic') return 'Topic'
  // RFC-097 v3.0: Organization is a first-class typed node — render under
  // the shared "Entity" label alongside Person / legacy Entity. Without
  // this branch, Organization subjects fall through to the generic
  // "Subject" label.
  if (t === 'Entity' || t === 'Person' || t === 'Organization') return 'Entity'
  return 'Subject'
})

const subjectName = computed(() => {
  const n = subjectNode.value
  if (!n) return subjectId.value
  const p = n.properties as Record<string, unknown> | undefined
  const label = typeof p?.label === 'string' ? p.label.trim() : ''
  if (label) return label
  const name = typeof p?.name === 'string' ? p.name.trim() : ''
  if (name) return name
  return subjectId.value
})

const subjectAliases = computed(() => {
  const a = subjectNode.value?.properties?.aliases
  if (!Array.isArray(a)) return ''
  const parts = a
    .filter((x): x is string => typeof x === 'string' && x.trim().length > 0)
    .map((x) => x.trim())
  return parts.length > 0 ? parts.join(', ') : ''
})

const subjectDescription = computed(() => {
  const d = subjectNode.value?.properties?.description
  return typeof d === 'string' && d.trim() ? d.trim() : ''
})

function onPrefillSearch(): void {
  const q = subjectName.value.trim()
  if (!q) return
  emit('prefillSemanticSearch', { query: q })
}
</script>

<template>
  <div
    :class="[
      'flex min-h-0 w-full min-w-0 flex-1 flex-col overflow-hidden',
      // Embedded in NodeDetail's Details tab, the parent owns the horizontal
      // gutter — the standalone-rail mx-3 would indent these sections past the
      // cluster/theme header. Only apply it when rendering standalone.
      embedded ? '' : 'mx-3',
    ]"
    role="region"
    aria-label="Topic or entity"
    data-testid="topic-entity-view"
  >
    <div
      v-if="!embedded"
      class="mt-1 flex shrink-0 items-baseline gap-2 border-b border-border pb-2"
    >
      <span
        class="text-[10px] font-semibold uppercase tracking-wider text-muted"
        data-testid="topic-entity-view-kind"
      >{{ subjectKindLabel }}</span>
      <h2
        class="min-w-0 flex-1 truncate text-xs font-semibold text-surface-foreground"
        data-testid="topic-entity-view-name"
        :title="subjectName"
      >
        {{ subjectName }}
      </h2>
      <button
        type="button"
        class="shrink-0 self-center rounded border border-border px-1.5 py-0.5 text-xs font-medium text-elevated-foreground hover:bg-overlay"
        data-testid="subject-rail-close"
        aria-label="Close topic detail"
        @click="emit('closeSubject')"
      >
        ×
      </button>
    </div>
    <div class="min-h-0 w-full min-w-0 flex-1 overflow-y-auto py-2">
      <div class="space-y-3">
      <p
        v-if="subjectAliases"
        class="text-[11px] text-muted"
        data-testid="topic-entity-view-aliases"
      >
        Aliases: {{ subjectAliases }}
      </p>
      <p
        v-if="subjectDescription"
        class="text-[11px] leading-snug text-surface-foreground"
        data-testid="topic-entity-view-description"
      >
        {{ subjectDescription }}
      </p>
      <!-- #1055 — related topics (topics that *share insights* with this one).
           Sits right under NodeDetail's "Theme" (topics *discussed together*)
           so the two co-topic signals are adjacent; a distinct colour (Topic
           magenta vs Theme teal) and a HelpTip keep them apart. -->
      <section
        v-if="relatedTopicRows.length"
        class="w-full min-w-0"
        aria-label="Related topics"
        data-testid="tev-related-topics"
      >
        <p
          class="mb-1 flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider"
          style="color: #da77f2"
        >
          Related topics
          <HelpTip :pref-width="260" button-aria-label="About related topics">
            <p class="font-sans text-[10px] leading-snug text-muted">
              Topics that
              <strong class="font-medium text-surface-foreground">share insights</strong> with this one —
              distinct from <strong class="font-medium text-surface-foreground">Theme</strong> above, which
              groups topics <strong class="font-medium text-surface-foreground">discussed together</strong>
              (co-occurrence).
            </p>
          </HelpTip>
        </p>
        <div class="flex flex-wrap gap-1.5">
          <button
            v-for="t in relatedTopicRows"
            :key="t.id"
            type="button"
            class="rounded-full border border-transparent px-2 py-0.5 text-[10px] text-surface-foreground hover:opacity-90"
            :style="{ backgroundColor: 'rgba(218,119,242,0.22)' }"
            data-testid="tev-related-topic-chip"
            :title="`Open ${t.text}`"
            @click="subject.focusTopic(t.id)"
          >{{ t.text }}</button>
        </div>
      </section>
      <!-- PRD-033 FR4.2 — cross-show coverage (the corpus differentiator). -->
      <section
        v-if="crossShowLoading || crossShowError || crossShowRows.length"
        class="w-full min-w-0"
        aria-label="Across shows"
        data-testid="tev-cross-show"
      >
        <h3 class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">
          Across shows
        </h3>
        <p
          v-if="crossShowLoading"
          data-testid="tev-cross-show-loading"
          class="text-[10px] text-muted"
        >
          Loading…
        </p>
        <p
          v-else-if="crossShowError"
          class="text-[10px] text-warning"
        >
          {{ crossShowError }}
        </p>
        <ul
          v-else
          class="space-y-1.5"
          data-testid="tev-cross-show-list"
        >
          <li
            v-for="row in crossShowRows"
            :key="row.showId"
            data-testid="tev-cross-show-row"
            class="min-w-0 text-[11px] leading-snug"
          >
            <button
              type="button"
              class="flex w-full min-w-0 items-start gap-2 rounded text-left hover:bg-overlay/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
              data-testid="tev-cross-show-open"
              :title="`Open ${shortId(row.showId)}`"
              @click="subject.focusGraphNode(row.showId)"
            >
              <div class="shrink-0 self-start">
                <PodcastCover
                  v-if="feedCoverFor(row.showId)"
                  :corpus-path="shell.corpusPath"
                  :feed-image-url="feedCoverFor(row.showId)?.imageUrl ?? null"
                  :feed-image-local-relpath="feedCoverFor(row.showId)?.imageLocalRelpath ?? null"
                  :alt="`Cover for ${shortId(row.showId)}`"
                  size-class="h-9 w-9"
                />
                <ShowGlyph v-else :name="shortId(row.showId)" size-class="h-9 w-9 rounded-lg text-sm" />
              </div>
              <div class="min-w-0 flex-1">
                <p class="break-words font-semibold text-surface-foreground">
                  {{ titleCaseWords(shortId(row.showId)) }}
                </p>
                <p class="break-words text-[10px] text-muted line-clamp-2" :title="row.insight.text">
                  {{ row.insight.text }}
                </p>
              </div>
            </button>
          </li>
        </ul>
      </section>

      <!-- PRD-033 FR4.2 — key voices (Person→Insight for this topic). -->
      <section
        v-if="voicesLoading || voicesError || voiceRows.length"
        class="w-full min-w-0"
        aria-label="Key voices"
        data-testid="tev-voices"
      >
        <h3 class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">
          Key voices
        </h3>
        <p
          v-if="voicesLoading"
          class="text-[10px] text-muted"
        >
          Loading…
        </p>
        <p
          v-else-if="voicesError"
          class="text-[10px] text-warning"
        >
          {{ voicesError }}
        </p>
        <ul
          v-else
          class="space-y-1.5"
          data-testid="tev-voices-list"
        >
          <li
            v-for="row in voiceRows"
            :key="row.personId"
            data-testid="tev-voice-row"
            class="min-w-0 text-[11px] leading-snug"
          >
            <button
              type="button"
              data-testid="tev-voice-link"
              class="flex w-full min-w-0 items-start gap-2 rounded text-left hover:bg-overlay/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
              :title="`Open Person panel for ${shortId(row.personId)}`"
              @click="onClickVoice(row.personId)"
            >
              <PersonInitialAvatar :name="shortId(row.personId)" size-class="h-9 w-9 text-sm" />
              <div class="min-w-0 flex-1">
                <p class="break-words font-semibold text-primary">
                  {{ titleCaseWords(shortId(row.personId)) }}
                  <span class="font-normal text-muted">({{ row.insights.length }})</span>
                </p>
                <p
                  v-if="row.insights[0]?.text"
                  class="break-words text-[10px] text-muted line-clamp-2"
                >
                  {{ row.insights[0]?.text }}
                </p>
              </div>
            </button>
          </li>
        </ul>
      </section>

      <!-- PRD-033 FR4.2 — entities involved (Insight→MENTIONS→Entity), most-mentioned first. -->
      <section
        v-if="entitiesLoading || entityRows.length"
        class="w-full min-w-0"
        aria-label="Entities involved"
        data-testid="tev-entities"
      >
        <h3 class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">
          Entities involved
        </h3>
        <p
          v-if="entitiesLoading"
          class="text-[10px] text-muted"
        >
          Loading…
        </p>
        <div
          v-else
          class="flex flex-wrap gap-1"
          data-testid="tev-entities-list"
        >
          <button
            v-for="ent in entityGroups"
            :key="ent.id"
            type="button"
            data-testid="tev-entity-chip"
            class="rounded border border-border bg-elevated/40 px-1.5 py-0.5 text-[10px] text-surface-foreground hover:border-primary hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
            :title="ent.shows > 1 ? `${ent.text} — involved across ${ent.shows} shows` : `Open panel for ${ent.text}`"
            @click="onClickEntity(ent.id, ent.type)"
          >{{ ent.text
            }}<span
              v-if="ent.shows > 1"
              class="ml-1 text-[9px] font-semibold text-muted"
              data-testid="tev-entity-shows"
            >· {{ ent.shows }} shows</span></button>
        </div>
      </section>

      <div
        v-if="!embedded"
        class="flex shrink-0 flex-wrap gap-2 pt-2"
      >
        <button
          type="button"
          class="rounded border border-border px-2 py-1 text-[11px] font-medium hover:bg-overlay"
          data-testid="topic-entity-view-go-graph"
          @click="emit('goGraph')"
        >
          Open in graph
        </button>
        <button
          type="button"
          class="rounded border border-border px-2 py-1 text-[11px] font-medium hover:bg-overlay"
          data-testid="topic-entity-view-prefill-search"
          @click="onPrefillSearch"
        >
          Prefill semantic search
        </button>
      </div>
      </div>
    </div>
  </div>
</template>
