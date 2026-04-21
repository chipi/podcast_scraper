<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { SearchHit } from '../../api/searchApi'
import EpisodeDetailPanel from '../episode/EpisodeDetailPanel.vue'
import GraphConnectionsSection from '../graph/GraphConnectionsSection.vue'
import GraphNodeRailPanel from '../graph/GraphNodeRailPanel.vue'
import { useGraphFilterStore } from '../../stores/graphFilters'
import { useGraphNavigationStore } from '../../stores/graphNavigation'
import { useSubjectStore } from '../../stores/subject'

const props = defineProps<{
  mainTab: 'digest' | 'library' | 'graph' | 'dashboard'
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
  prefillSemanticSearch: [{ query: string }]
  openExploreTopicFilter: [{ topic: string }]
  openExploreSpeakerFilter: [{ speaker: string }]
  openExploreInsightFilters: [{ groundedOnly: boolean; minConfidence: number | null }]
  openLibraryEpisode: [{ metadata_relative_path: string }]
  openEpisodeSummary: [SearchHit]
  switchMainTab: ['digest' | 'library' | 'graph' | 'dashboard']
}>()

const episodeConnectionsViewArtifact = computed(() =>
  graphFilters.viewWithEgo(graphNav.graphEgoFocusCyId),
)

const episodeSubjectNeighbourhoodEnabled = computed(
  () => props.mainTab === 'graph' && Boolean(subject.graphConnectionsCyId?.trim()),
)

type EpisodeSubjectDetailTab = 'details' | 'neighbourhood'
const episodeSubjectDetailTab = ref<EpisodeSubjectDetailTab>('details')

watch(
  () => subject.episodeMetadataPath,
  () => {
    episodeSubjectDetailTab.value = 'details'
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
      <div class="mx-3 mt-1 flex shrink-0 items-center justify-end border-b border-border pb-1">
        <button
          type="button"
          class="rounded border border-border px-2 py-0.5 text-[10px] font-medium text-elevated-foreground hover:bg-overlay"
          data-testid="subject-rail-close"
          aria-label="Close subject panel"
          @click="emit('closeSubject')"
        >
          ×
        </button>
      </div>
      <template v-if="subject.kind === 'graph-node' && subject.graphNodeCyId?.trim()">
        <GraphNodeRailPanel
          @go-graph="emit('goGraph')"
          @prefill-semantic-search="emit('prefillSemanticSearch', $event)"
          @open-explore-topic-filter="emit('openExploreTopicFilter', $event)"
          @open-explore-speaker-filter="emit('openExploreSpeakerFilter', $event)"
          @open-explore-insight-filters="emit('openExploreInsightFilters', $event)"
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
            <h2 class="text-xs font-semibold text-surface-foreground">
              Episode
            </h2>
          </div>
          <EpisodeDetailPanel
            class="min-h-0 min-w-0 flex-1"
            :rail-neighbourhood-enabled="episodeSubjectNeighbourhoodEnabled"
            :rail-detail-tab="episodeSubjectDetailTab"
            @focus-search="emit('focusSearchHandoff', $event)"
            @switch-main-tab="emit('switchMainTab', $event)"
          >
            <template #episode-rail-tabs>
              <nav
                v-if="episodeSubjectNeighbourhoodEnabled"
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
      <template v-else-if="subject.kind === 'topic'">
        <div class="mx-3 mt-2 text-[11px] text-muted">
          Topic detail is not available in this build.
        </div>
      </template>
      <template v-else-if="subject.kind === 'person'">
        <div class="mx-3 mt-2 text-[11px] text-muted">
          Person detail is not available in this build.
        </div>
      </template>
    </template>
  </div>
</template>
