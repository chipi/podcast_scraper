<script setup lang="ts">
/**
 * #672 — Topic / Entity rail panel ("TEV"). Reads the focused subject id
 * from the subject store, looks it up in the merged graph slice, and shows
 * name + aliases + description, then the registered topic view sections.
 *
 * Mounts inside ``SubjectRail`` for ``subject.kind === 'topic'``. Entity
 * subjects still flow through ``focusGraphNode`` → existing ``NodeDetail``.
 *
 * When ``embedded`` is true (folded into NodeDetail's Details tab for topic
 * nodes), the header (kind + name) and footer action buttons are hidden so
 * NodeDetail owns the chrome. ``subjectIdOverride`` lets the caller supply the
 * node id directly instead of relying on the subject store.
 */
import { computed } from 'vue'
import { topicViewSections } from '../../extensions'
import type { RawGraphNode } from '../../types/artifact'
import { useArtifactsStore } from '../../stores/artifacts'
import { useSubjectStore } from '../../stores/subject'
import { findRawNodeInArtifactByIdOrPrefixed } from '../../utils/parsing'

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
const subject = useSubjectStore()

const subjectId = computed(() => props.subjectIdOverride?.trim() || subject.topicId?.trim() || '')

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
      <template v-if="subjectId">
        <component
          :is="section.component"
          v-for="section in topicViewSections.list().filter((x) => x.region === 'after-description')"
          :key="section.id"
          :topic-id="subjectId"
        />
      </template>

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
