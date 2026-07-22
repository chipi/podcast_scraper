<script setup lang="ts">
/**
 * Corpus-wide grounded search (PRD-042 FR5 / RFC-099 §Home). Searches the whole library via
 * GET /api/app/search (extractive, no request-time LLM). Rather than a flat wall of mixed
 * passages, results are **grouped by episode** (ranked by their best hit) and each passage is
 * labelled by kind (Insight / Transcript / Topic). A "Play from …" jump appears only when the
 * passage carries a real timestamp — otherwise we open the episode rather than fake a 0:00.
 */
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import { resolveEntity, searchCorpus } from '../services/api'
import type { EntityRef, SearchHit } from '../services/types'
import { hitStartSeconds } from '../player/insights'
import { formatTime } from '../player/transcriptSync'
import { formatPublishDate } from '../utils/format'
import { aggregateRelatedTopics } from '../utils/relatedTopics'
import {
  collapseFoldableHits,
  isFoldedCluster,
  type CollapsedRow,
  type FoldedHitCluster,
} from '../utils/collapseFoldableHits'
import { summarizeMatchedFields } from '../utils/matchedFields'
import { useAuthStore } from '../stores/auth'
import EntityCard from '../components/EntityCard.vue'

const { t, locale } = useI18n()
const route = useRoute()
const router = useRouter()
const auth = useAuthStore()

// Scope (P3 Recall, #1124): 'all' = whole library; 'mine' = grounded recall over the user's
// heard∪captured corpus ("what have I learned about X"). The toggle only shows when signed in.
const scope = ref<'all' | 'mine'>(route.query.scope === 'mine' ? 'mine' : 'all')

const query = ref(String(route.query.q ?? ''))
const results = ref<SearchHit[]>([])
const entity = ref<EntityRef | null>(null)
const cardTarget = ref<{ kind: 'person' | 'topic'; id: string } | null>(null)
const searching = ref(false)
const error = ref(false)
const ran = ref(false)

type Kind = 'insight' | 'transcript' | 'topic' | 'passage'
interface EpisodeGroup {
  slug: string | null
  title: string
  show: string | null
  date: string | null
  art: string | null
  hits: SearchHit[]
  /** #1261-3: hits collapsed so multiple same-kind foldable rows (transcript,
   *  title, description, summary) render as one expandable summary row. */
  rows: CollapsedRow[]
}

const md = (h: SearchHit) => h.metadata as Record<string, unknown>
const hitSlug = (h: SearchHit) => (md(h).episode_slug as string | undefined) ?? null
const hitEpisode = (h: SearchHit) => (md(h).episode_title as string | undefined) ?? null
const hitShow = (h: SearchHit) => (md(h).podcast_title as string | undefined) ?? null
const hitDate = (h: SearchHit) => (md(h).publish_date as string | undefined) ?? null
const hitArt = (h: SearchHit) => (md(h).episode_artwork as string | undefined) ?? null

function hitKind(h: SearchHit): Kind {
  const dt = md(h).doc_type
  if (dt === 'insight') return 'insight'
  if (dt === 'transcript') return 'transcript'
  if (dt === 'kg_topic') return 'topic'
  return 'passage'
}

// #1261-2: aggregate per-hit related_topics into a chip row above the episode
// groups. Tapping a chip opens the Topic EntityCard modal — deeper exploration
// without expanding the search surface itself.
const relatedTopicChips = computed(() => aggregateRelatedTopics(results.value, 8))

function openTopicChip(topicId: string): void {
  cardTarget.value = { kind: 'topic', id: topicId }
}

// #1261-5: shim so the template's inline v-for can call the helper by name.
// Wrapped so summarizeMatchedFields can be swapped in unit tests independently.
function matchedFieldChips(hits: SearchHit[]) {
  return summarizeMatchedFields(hits)
}

// Group passages under their source episode, preserving rank order (results arrive best-first,
// so an episode's rank is its first appearance).
const groups = computed<EpisodeGroup[]>(() => {
  const byKey = new Map<string, EpisodeGroup>()
  const order: string[] = []
  for (const h of results.value) {
    const slug = hitSlug(h)
    const key = slug ?? `doc:${h.doc_id}`
    let g = byKey.get(key)
    if (!g) {
      g = {
        slug,
        title: hitEpisode(h) ?? t('player.notFound'),
        show: hitShow(h),
        date: hitDate(h),
        art: hitArt(h),
        hits: [],
        rows: [],
      }
      byKey.set(key, g)
      order.push(key)
    }
    g.hits.push(h)
  }
  const built = order.map((k) => byKey.get(k)!)
  for (const g of built) g.rows = collapseFoldableHits(g.hits)
  return built
})

// #1261-3: expand/collapse state per folded-cluster row, keyed by "<slug>|<kind>".
// A Set of expanded keys — reactive because the ref is a plain Set instance and
// the template mutates via ``clusterExpanded.value = new Set(...)`` on toggle.
const clusterExpanded = ref<Set<string>>(new Set())
function clusterKey(groupKey: string | null, cluster: FoldedHitCluster): string {
  return `${groupKey ?? 'nogroup'}|${cluster.foldedKind}`
}
function toggleCluster(groupKey: string | null, cluster: FoldedHitCluster): void {
  const key = clusterKey(groupKey, cluster)
  const next = new Set(clusterExpanded.value)
  if (next.has(key)) next.delete(key)
  else next.add(key)
  clusterExpanded.value = next
}
function isClusterOpen(groupKey: string | null, cluster: FoldedHitCluster): boolean {
  return clusterExpanded.value.has(clusterKey(groupKey, cluster))
}

async function run(q: string): Promise<void> {
  const term = q.trim()
  if (!term) {
    results.value = []
    entity.value = null
    ran.value = false
    return
  }
  searching.value = true
  error.value = false
  const recall = scope.value === 'mine'
  // Resolve a person/topic entity match in parallel with the passage search (3.4) — but only in the
  // whole-library scope; recall is about the user's own passages, not a global entity card.
  const entityP = recall
    ? Promise.resolve((entity.value = null))
    : resolveEntity(term).then(
        (r) => (entity.value = r.entity),
        () => (entity.value = null),
      )
  try {
    // #1261-1: always ask the server for related-topic decoration; a broken
    // enricher chain degrades to plain hits and the chip row simply disappears.
    const resp = await searchCorpus(term, 12, recall ? 'mine' : 'all', true)
    results.value = resp.results
    error.value = Boolean(resp.error)
  } catch {
    error.value = true
  } finally {
    await entityP
    searching.value = false
    ran.value = true
  }
}

function setScope(s: 'all' | 'mine'): void {
  if (scope.value === s) return
  scope.value = s
  void router.replace({ name: 'search', query: { q: query.value.trim() || undefined, scope: s } })
  void run(query.value)
}

function openEntity(): void {
  if (entity.value) cardTarget.value = { kind: entity.value.kind, id: entity.value.id }
}

function submit(): void {
  const q = query.value.trim() || undefined
  void router.replace({ name: 'search', query: { q, scope: scope.value } })
}

function openEpisode(slug: string | null, hit?: SearchHit): void {
  if (!slug) return
  const s = hit ? hitStartSeconds(hit) : null
  void router.push({
    name: 'player',
    params: { slug },
    query: s != null ? { t: String(Math.floor(s)) } : {},
  })
}

watch(() => route.query.q, (q) => run(String(q ?? '')), { immediate: true })

const showEmpty = computed(
  () =>
    ran.value &&
    !searching.value &&
    !error.value &&
    results.value.length === 0 &&
    entity.value === null,
)
</script>

<template>
  <section>
    <h1 class="mb-4 font-display text-3xl font-extrabold tracking-tight">{{ t('search.title') }}</h1>

    <form class="flex gap-2" @submit.prevent="submit">
      <label class="sr-only" for="search-q">{{ t('search.title') }}</label>
      <input
        id="search-q"
        v-model="query"
        type="search"
        :placeholder="t('search.placeholder')"
        class="min-w-0 flex-1 rounded-full border border-border bg-surface px-4 py-3 text-sm"
      />
      <button type="submit" class="rounded-full bg-accent px-5 py-3 font-bold text-accent-foreground">
        {{ t('search.title') }}
      </button>
    </form>

    <!-- Recall scope (P3 #1124): search everything, or just your corpus. Auth-gated. -->
    <div
      v-if="auth.isAuthenticated"
      role="tablist"
      :aria-label="t('search.scopeLabel')"
      class="mt-3 inline-flex gap-1 rounded-full border border-border bg-surface p-0.5 text-sm"
    >
      <button
        v-for="opt in (['all', 'mine'] as const)"
        :key="opt"
        type="button"
        role="tab"
        :aria-selected="scope === opt"
        class="rounded-full px-3 py-1 font-semibold transition"
        :class="scope === opt ? 'bg-accent text-accent-foreground' : 'text-muted hover:text-canvas-foreground'"
        @click="setScope(opt)"
      >
        {{ opt === 'all' ? t('search.scopeAll') : t('search.scopeMine') }}
      </button>
    </div>

    <!-- Entity match (3.4): a person/topic card above the passages, opening the full card on tap. -->
    <button
      v-if="entity && !searching"
      type="button"
      class="mt-4 flex w-full items-center gap-3 rounded-xl border border-border bg-surface p-4 text-left transition hover:bg-overlay"
      :aria-label="t('kp.openEntity', { term: entity.label })"
      @click="openEntity"
    >
      <span class="min-w-0 flex-1">
        <span class="lp-kicker block">{{ entity.kind === 'person' ? t('ec.person') : t('ec.topic') }}</span>
        <span class="block font-display text-lg font-bold text-canvas-foreground">{{ entity.label }}</span>
      </span>
      <span class="shrink-0 text-sm font-semibold text-accent">{{ t('search.viewEntity') }} ›</span>
    </button>

    <p v-if="searching" class="mt-4 text-muted">{{ t('search.searching') }}</p>
    <p v-else-if="error" class="mt-4 text-muted">{{ t('search.error') }}</p>
    <p v-else-if="showEmpty && scope === 'mine'" class="mt-4 text-muted">{{ t('search.recallEmpty') }}</p>
    <p v-else-if="showEmpty" class="mt-4 text-muted">{{ t('search.noResults') }}</p>

    <template v-else-if="results.length">
      <p class="mt-4 text-xs font-semibold uppercase tracking-wider text-muted">
        {{ t('search.summary', { passages: results.length, episodes: groups.length }) }}
      </p>

      <!-- #1261-2: related-topic chip row above the episode groups. Silent
           when the QueryEnricher chain returned nothing (broken corpus, no
           topic_similarity.json, or no hits carried topic decorations). -->
      <div
        v-if="relatedTopicChips.length"
        class="mt-3 flex flex-wrap items-center gap-1.5"
        data-testid="related-topic-chips"
      >
        <span class="lp-kicker mr-1">{{ t('search.alsoAbout') }}</span>
        <button
          v-for="chip in relatedTopicChips"
          :key="chip.topicId"
          type="button"
          class="rounded-full border border-border bg-surface px-2.5 py-1 text-xs font-semibold text-canvas-foreground transition hover:bg-overlay"
          :aria-label="t('search.openTopicChip', { label: chip.label })"
          @click="openTopicChip(chip.topicId)"
        >
          {{ chip.label }}
        </button>
      </div>

      <ul class="mt-3 flex flex-col gap-3">
        <li
          v-for="g in groups"
          :key="g.slug ?? g.title"
          class="overflow-hidden rounded-xl border border-border bg-surface"
        >
          <!-- Episode header: opens the player -->
          <button
            type="button"
            class="flex w-full items-center gap-3 px-4 pt-4 text-left"
            @click="openEpisode(g.slug)"
          >
            <img
              v-if="g.art"
              :src="g.art"
              alt=""
              loading="lazy"
              class="h-12 w-12 shrink-0 rounded-md bg-elevated object-cover"
            />
            <span class="min-w-0 flex-1">
              <span class="block font-display text-base font-bold leading-snug text-canvas-foreground">
                {{ g.title }}
              </span>
              <span v-if="g.show || g.date" class="lp-kicker mt-0.5 block">
                {{ g.show }}<template v-if="g.show && g.date"> · </template>{{ g.date ? formatPublishDate(g.date, locale) : '' }}
              </span>
              <!-- #1261-5: matched-field breakdown ("Matched: Title · Summary
                   ×2 · Transcript") — small kicker line so the listener knows
                   why this episode surfaced without tapping through. Hidden
                   when nothing resolved to an episode-level field. -->
              <span
                v-if="matchedFieldChips(g.hits).length"
                class="lp-kicker mt-0.5 block"
                data-testid="matched-fields"
              >
                {{ t('search.matchedPrefix') }}
                <template v-for="(m, mi) in matchedFieldChips(g.hits)" :key="m.label">
                  <template v-if="mi > 0"> · </template>
                  <span class="font-semibold text-canvas-foreground">
                    {{ m.label }}<template v-if="m.count > 1"> ×{{ m.count }}</template>
                  </span>
                </template>
              </span>
            </span>
            <span class="shrink-0 text-xs font-semibold text-muted">{{ t('search.matchCount', g.hits.length) }}</span>
          </button>

          <!-- Matching passages (#1261-3: foldable rows collapse to one
               expandable summary per (episode, source-kind)). -->
          <ul class="mt-3 flex flex-col">
            <template v-for="(row, i) in g.rows" :key="row.doc_id ? row.doc_id + i : `c${i}`">
              <!-- FoldedHitCluster: N hits of the same foldable kind (transcript /
                   title / description / summary) collapsed into one expandable row. -->
              <li v-if="isFoldedCluster(row)" class="border-t border-border">
                <button
                  type="button"
                  class="flex w-full items-center gap-2 px-4 py-3 text-left"
                  :aria-expanded="isClusterOpen(g.slug, row)"
                  :aria-label="t('search.expandCluster', {
                    kind: t(`search.foldedKind.${row.foldedKind}`),
                    count: row.members.length,
                  })"
                  data-testid="folded-cluster-row"
                  @click="toggleCluster(g.slug, row)"
                >
                  <span
                    class="rounded bg-overlay px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-canvas-foreground"
                  >
                    {{ t(`search.foldedKind.${row.foldedKind}`) }}
                  </span>
                  <span class="text-xs font-semibold text-muted">
                    {{ t('search.foldedCount', row.members.length) }}
                  </span>
                  <span class="ml-auto text-xs font-bold text-accent">
                    {{ isClusterOpen(g.slug, row) ? '▲' : '▼' }}
                  </span>
                </button>
                <ul v-if="isClusterOpen(g.slug, row)" class="flex flex-col">
                  <li
                    v-for="(m, mi) in row.members"
                    :key="m.doc_id + mi"
                    class="border-t border-border px-6 py-2"
                  >
                    <div class="flex items-center gap-2">
                      <button
                        v-if="hitStartSeconds(m) != null && g.slug"
                        type="button"
                        class="ml-auto font-mono text-xs font-bold text-accent"
                        :aria-label="t('search.jumpTo', {
                          time: formatTime(hitStartSeconds(m) ?? 0),
                          episode: g.title,
                        })"
                        @click="openEpisode(g.slug, m)"
                      >
                        ▶ {{ t('search.playHere', { time: formatTime(hitStartSeconds(m) ?? 0) }) }}
                      </button>
                    </div>
                    <p class="mt-1 line-clamp-2 text-sm leading-relaxed text-surface-foreground">
                      {{ m.text }}
                    </p>
                  </li>
                </ul>
              </li>
              <!-- Plain hit (insight / kg_topic / kg_entity / lifted transcript). -->
              <li v-else class="border-t border-border px-4 py-3">
                <div class="flex items-center gap-2">
                  <span
                    class="rounded bg-overlay px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider"
                    :class="{
                      'text-grounded': hitKind(row) === 'insight',
                      'text-canvas-foreground': hitKind(row) === 'transcript' || hitKind(row) === 'passage',
                      'text-topic': hitKind(row) === 'topic',
                    }"
                  >
                    {{ t(`search.kind.${hitKind(row)}`) }}
                  </span>
                  <button
                    v-if="hitStartSeconds(row) != null && g.slug"
                    type="button"
                    class="ml-auto font-mono text-xs font-bold text-accent"
                    :aria-label="t('search.jumpTo', { time: formatTime(hitStartSeconds(row) ?? 0), episode: g.title })"
                    @click="openEpisode(g.slug, row)"
                  >
                    ▶ {{ t('search.playHere', { time: formatTime(hitStartSeconds(row) ?? 0) }) }}
                  </button>
                </div>
                <p
                  class="mt-1.5 line-clamp-2 text-sm leading-relaxed"
                  :class="hitKind(row) === 'topic' ? 'italic text-muted' : 'text-surface-foreground'"
                >
                  {{ row.text }}
                </p>
              </li>
            </template>
          </ul>
        </li>
      </ul>
    </template>

    <EntityCard
      v-if="cardTarget"
      :kind="cardTarget.kind"
      :id="cardTarget.id"
      @close="cardTarget = null"
    />
  </section>
</template>
