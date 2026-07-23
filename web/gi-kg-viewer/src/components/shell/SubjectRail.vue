<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import EpisodeDetailPanel from '../episode/EpisodeDetailPanel.vue'
import ShowRailPanel from '../episode/ShowRailPanel.vue'
import GraphConnectionsSection from '../graph/GraphConnectionsSection.vue'
import GraphNodeRailPanel from '../graph/GraphNodeRailPanel.vue'
import { useGraphFilterStore } from '../../stores/graphFilters'
import { useGraphNavigationStore } from '../../stores/graphNavigation'
import { useSubjectStore } from '../../stores/subject'

const props = defineProps<{
  mainTab: 'digest' | 'library' | 'search' | 'graph' | 'dashboard' | 'ops' | 'admin'
}>()

const subject = useSubjectStore()
const graphFilters = useGraphFilterStore()
const graphNav = useGraphNavigationStore()

type FocusSearchPayload = {
  feed: string
  query: string
  since?: string
  feedDisplayTitle?: string
}

const emit = defineEmits<{
  closeSubject: []
  goGraph: []
  focusSearchHandoff: [FocusSearchPayload]
  /** Search v3 §S6 — episode-scoped rail launcher from EpisodeDetailPanel. */
  openSearchInEpisode: [{ episodeId: string; query?: string }]
  prefillSemanticSearch: [{ query: string }]
  openSearchTopicFilter: [{ topic: string }]
  openSearchSpeakerFilter: [{ speaker: string }]
  openSearchInsightFilters: [{ groundedOnly: boolean; minConfidence: number | null }]
  openLibraryEpisode: [{ metadata_relative_path: string }]
  switchMainTab: ['digest' | 'library' | 'graph' | 'dashboard']
}>()

const episodeConnectionsViewArtifact = computed(() =>
  graphFilters.viewWithEgo(graphNav.graphEgoFocusCyId),
)

const episodeSubjectNeighbourhoodEnabled = computed(
  () => props.mainTab === 'graph' && Boolean(subject.graphConnectionsCyId?.trim()),
)

type EpisodeSubjectDetailTab = 'details' | 'enrichment' | 'neighbourhood'
const episodeSubjectDetailTab = ref<EpisodeSubjectDetailTab>('details')

// EpisodeDetailPanel bubbles this up once EpisodeEnrichmentSection has probed
// the envelopes; the Enrichment tab is hidden until an episode actually has
// signals. Falls back to Details if the tab empties while it's active.
const episodeEnrichmentHasContent = ref(false)
watch(episodeEnrichmentHasContent, (has) => {
  if (!has && episodeSubjectDetailTab.value === 'enrichment') {
    episodeSubjectDetailTab.value = 'details'
  }
})

watch(
  () => subject.episodeMetadataPath,
  () => {
    episodeSubjectDetailTab.value = 'details'
    episodeEnrichmentHasContent.value = false
  },
)

watch(
  () => subject.graphConnectionsCyId,
  () => {
    episodeSubjectDetailTab.value = 'details'
  },
)

const emptyHint =
  'Select an episode, topic, or graph node to see details here.'
</script>

<template>
  <div class="flex min-h-0 flex-1 flex-col">
    <div
      v-if="subject.kind === null"
      class="mx-3 mt-4 text-center text-[11px] leading-snug text-muted"
      data-testid="subject-rail-empty"
    >
      {{ emptyHint }}
    </div>
    <template v-else>
      <template v-if="subject.kind === 'graph-node' && subject.graphNodeCyId?.trim()">
        <GraphNodeRailPanel
          @go-graph="emit('goGraph')"
          @prefill-semantic-search="emit('prefillSemanticSearch', $event)"
          @open-search-topic-filter="emit('openSearchTopicFilter', $event)"
          @open-search-speaker-filter="emit('openSearchSpeakerFilter', $event)"
          @open-search-insight-filters="emit('openSearchInsightFilters', $event)"
          @open-library-episode="emit('openLibraryEpisode', $event)"
          @close-subject-rail="emit('closeSubject')"
        />
      </template>
      <template v-else-if="subject.kind === 'episode' && subject.episodeMetadataPath?.trim()">
        <div
          class="mx-3 flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden"
          role="region"
          aria-label="Episode"
          data-testid="episode-detail-rail"
        >
          <div class="mt-1 flex shrink-0 items-center justify-between gap-2 border-b border-border pb-2">
            <div class="flex min-w-0 items-center gap-1.5">
              <button
                v-if="subject.canGoBack"
                type="button"
                class="shrink-0 rounded border border-border px-1.5 py-0.5 text-xs font-medium text-elevated-foreground hover:bg-overlay"
                data-testid="subject-rail-back"
                aria-label="Back to previous node"
                @click="subject.back()"
              >
                ←
              </button>
              <h2 class="min-w-0 truncate text-xs font-semibold text-surface-foreground">
                Episode
              </h2>
            </div>
            <button
              type="button"
              class="shrink-0 self-center rounded border border-border px-1.5 py-0.5 text-xs font-medium text-elevated-foreground hover:bg-overlay"
              data-testid="subject-rail-close"
              aria-label="Close episode detail"
              @click="emit('closeSubject')"
            >
              ×
            </button>
          </div>
          <EpisodeDetailPanel
            class="min-h-0 min-w-0 flex-1"
            :rail-neighbourhood-enabled="episodeSubjectNeighbourhoodEnabled"
            :rail-detail-tab="episodeSubjectDetailTab"
            @focus-search="emit('focusSearchHandoff', $event)"
            @open-search-in-episode="emit('openSearchInEpisode', $event)"
            @switch-main-tab="emit('switchMainTab', $event)"
            @enrichment-has-content="episodeEnrichmentHasContent = $event"
          >
            <template #episode-rail-tabs>
              <nav
                v-if="episodeSubjectNeighbourhoodEnabled || episodeEnrichmentHasContent"
                class="flex shrink-0 gap-1 border-b border-border bg-elevated/50 px-2 py-1.5"
                role="tablist"
                aria-label="Episode detail sections"
              >
                <button
                  id="episode-detail-rail-tab-details"
                  type="button"
                  role="tab"
                  class="flex-1 rounded px-2 py-1 text-center text-xs font-medium transition-colors"
                  :class="
                    episodeSubjectDetailTab === 'details'
                      ? 'bg-primary text-primary-foreground'
                      : 'text-elevated-foreground hover:bg-overlay'
                  "
                  :aria-selected="episodeSubjectDetailTab === 'details'"
                  aria-controls="episode-detail-rail-panel-details"
                  data-testid="episode-detail-rail-tab-details"
                  :tabindex="episodeSubjectDetailTab === 'details' ? 0 : -1"
                  @click="episodeSubjectDetailTab = 'details'"
                >
                  Details
                </button>
                <button
                  v-if="episodeEnrichmentHasContent"
                  id="episode-detail-rail-tab-enrichment"
                  type="button"
                  role="tab"
                  class="flex-1 rounded px-2 py-1 text-center text-xs font-medium transition-colors"
                  :class="
                    episodeSubjectDetailTab === 'enrichment'
                      ? 'bg-primary text-primary-foreground'
                      : 'text-elevated-foreground hover:bg-overlay'
                  "
                  :aria-selected="episodeSubjectDetailTab === 'enrichment'"
                  aria-controls="episode-detail-rail-panel-enrichment"
                  data-testid="episode-detail-rail-tab-enrichment"
                  :tabindex="episodeSubjectDetailTab === 'enrichment' ? 0 : -1"
                  @click="episodeSubjectDetailTab = 'enrichment'"
                >
                  Enrichment
                </button>
                <button
                  v-if="episodeSubjectNeighbourhoodEnabled"
                  id="episode-detail-rail-tab-neighbourhood"
                  type="button"
                  role="tab"
                  class="flex-1 rounded px-2 py-1 text-center text-xs font-medium transition-colors"
                  :class="
                    episodeSubjectDetailTab === 'neighbourhood'
                      ? 'bg-primary text-primary-foreground'
                      : 'text-elevated-foreground hover:bg-overlay'
                  "
                  :aria-selected="episodeSubjectDetailTab === 'neighbourhood'"
                  aria-controls="episode-detail-rail-panel-neighbourhood"
                  data-testid="episode-detail-rail-tab-neighbourhood"
                  :tabindex="episodeSubjectDetailTab === 'neighbourhood' ? 0 : -1"
                  @click="episodeSubjectDetailTab = 'neighbourhood'"
                >
                  Neighbourhood
                </button>
              </nav>
            </template>
            <template #episode-rail-neighbourhood>
              <GraphConnectionsSection
                :view-artifact="episodeConnectionsViewArtifact"
                :node-id="subject.graphConnectionsCyId"
                :dense-neighbor-list="false"
                @go-graph="emit('goGraph')"
                @open-library-episode="emit('openLibraryEpisode', $event)"
                @prefill-semantic-search="emit('prefillSemanticSearch', $event)"
              />
            </template>
          </EpisodeDetailPanel>
        </div>
      </template>
      <template v-else-if="subject.kind === 'show' && subject.feedId?.trim()">
        <div
          class="mx-3 flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden"
          role="region"
          aria-label="Show"
          data-testid="show-detail-rail"
        >
          <div class="mt-1 flex shrink-0 items-center justify-between gap-2 border-b border-border pb-2">
            <div class="flex min-w-0 items-center gap-1.5">
              <button
                v-if="subject.canGoBack"
                type="button"
                class="shrink-0 rounded border border-border px-1.5 py-0.5 text-xs font-medium text-elevated-foreground hover:bg-overlay"
                data-testid="subject-rail-back"
                aria-label="Back to previous subject"
                @click="subject.back()"
              >
                ←
              </button>
              <h2 class="min-w-0 truncate text-xs font-semibold text-surface-foreground">
                Show
              </h2>
            </div>
            <button
              type="button"
              class="shrink-0 self-center rounded border border-border px-1.5 py-0.5 text-xs font-medium text-elevated-foreground hover:bg-overlay"
              data-testid="subject-rail-close"
              aria-label="Close show detail"
              @click="emit('closeSubject')"
            >
              ×
            </button>
          </div>
          <ShowRailPanel
            class="min-h-0 min-w-0 flex-1"
            @switch-main-tab="emit('switchMainTab', $event)"
          />
        </div>
      </template>
    </template>
  </div>
</template>
