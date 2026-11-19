<script setup lang="ts">
/**
 * The episodes a topic, person or organisation appears in, newest first, from the KG
 * (`POST /api/corpus/node-episodes`; the server joins title, show and date from the catalog).
 * Without grounded insights the topic and person rails had nothing to show beyond buttons; this
 * is what a KG-only corpus knows about them. Renders nothing when the node is in no episode.
 */
import { computed, ref, watch } from 'vue'
import { fetchNodeEpisodes, type CorpusNodeEpisodeItem } from '../../api/corpusLibraryApi'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'
import { StaleGeneration } from '../../utils/staleGeneration'

/** Rows shown; the heading gives the total when there are more. */
const SHOWN = 10

const props = defineProps<{ nodeId: string }>()

const shell = useShellStore()
const subject = useSubjectStore()
const episodes = ref<CorpusNodeEpisodeItem[]>([])
const total = ref(0)
const gate = new StaleGeneration()

const rows = computed(() => episodes.value.filter((e) => (e.metadata_relative_path ?? '').trim()))

async function load(): Promise<void> {
  const seq = gate.bump()
  const root = shell.corpusPath.trim()
  const id = props.nodeId.trim()
  episodes.value = []
  total.value = 0
  if (!root || !id || !shell.healthStatus) return
  try {
    const res = await fetchNodeEpisodes(root, id, SHOWN)
    if (gate.isStale(seq)) return
    episodes.value = res.episodes ?? []
    total.value = res.total_matched ?? episodes.value.length
  } catch {
    if (!gate.isStale(seq)) episodes.value = []
  }
}

watch(() => [props.nodeId, shell.corpusPath, shell.healthStatus], () => void load(), {
  immediate: true,
})

function fmtDate(iso: string | null | undefined): string {
  const s = (iso ?? '').trim()
  return s ? s.slice(0, 10) : ''
}

function open(e: CorpusNodeEpisodeItem): void {
  subject.focusEpisode(e.metadata_relative_path ?? '', {
    uiTitle: e.episode_title?.trim() || null,
    episodeId: e.episode_id ?? null,
  })
}
</script>

<template>
  <section v-if="rows.length" class="mb-3" data-testid="node-episodes">
    <h4 class="mb-1 text-[10px] font-semibold uppercase tracking-wide text-muted">
      Episodes
      <span v-if="total > rows.length" class="font-normal normal-case" data-testid="node-episodes-count"
        >({{ rows.length }} newest of {{ total }})</span
      >
    </h4>
    <ul class="space-y-1">
      <li v-for="(e, i) in rows" :key="e.metadata_relative_path ?? i">
        <div
          role="button"
          tabindex="0"
          :data-testid="`node-episodes-row-${i}`"
          class="cursor-pointer rounded px-1 py-0.5 outline-none hover:bg-overlay focus-visible:ring-2 focus-visible:ring-primary"
          :aria-label="`${e.episode_title || 'Episode'}${e.feed_title ? `, ${e.feed_title}` : ''}`"
          @click="open(e)"
          @keydown.enter.prevent="open(e)"
          @keydown.space.prevent="open(e)"
        >
          <div class="flex items-baseline justify-between gap-2">
            <span class="min-w-0 flex-1 truncate text-xs font-medium text-surface-foreground">{{
              e.episode_title || e.episode_id || 'Episode'
            }}</span>
            <span v-if="fmtDate(e.publish_date)" class="shrink-0 text-[11px] text-muted">{{
              fmtDate(e.publish_date)
            }}</span>
          </div>
          <p v-if="e.feed_title" class="truncate text-[11px] text-muted">{{ e.feed_title }}</p>
        </div>
      </li>
    </ul>
  </section>
</template>
