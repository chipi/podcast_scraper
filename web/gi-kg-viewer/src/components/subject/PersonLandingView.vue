<script setup lang="ts">
/**
 * #672 — Person Landing rail panel. Reads ``subject.personId`` (or ``subjectIdOverride``) and
 * shows the person's identity — name, role, aliases, description — plus the registered person
 * view sections (extension point ``personProfileSections``).
 *
 * It renders in slices (``view``): ``profile`` is the node detail's Details tab; ``positions`` is
 * a person's positions view, whose content is a registered ``positions`` section; ``full`` stacks
 * the profile and a ``full`` section (the standalone layout, used by tests). ``embedded`` hides the
 * standalone header and footer chrome so the node detail panel owns it.
 */
import { computed } from 'vue'
import { personProfileSections } from '../../extensions'
import type { RawGraphNode } from '../../types/artifact'
import { useArtifactsStore } from '../../stores/artifacts'
import { useSubjectStore } from '../../stores/subject'
import { findRawNodeInArtifactByIdOrPrefixed, personRoleFromNode } from '../../utils/parsing'

const emit = defineEmits<{
  goGraph: []
  closeSubject: []
  prefillSemanticSearch: [{ query: string }]
}>()

const props = withDefaults(
  defineProps<{
    embedded?: boolean
    subjectIdOverride?: string
    view?: 'full' | 'profile' | 'positions'
  }>(),
  { embedded: false, subjectIdOverride: '', view: 'full' },
)

const artifacts = useArtifactsStore()
const subject = useSubjectStore()

/** Person subject id — from the embed prop (NodeDetail fold) or the store. */
const personId = computed(() => props.subjectIdOverride?.trim() || subject.personId?.trim() || '')

const personNode = computed<RawGraphNode | null>(() => {
  const art = artifacts.displayArtifact
  const id = personId.value
  if (!art || !id) return null
  return findRawNodeInArtifactByIdOrPrefixed(art, id)
})

const personName = computed(() => {
  const n = personNode.value
  if (!n) return personId.value
  const p = n.properties as Record<string, unknown> | undefined
  const name = typeof p?.name === 'string' ? p.name.trim() : ''
  if (name) return name
  const label = typeof p?.label === 'string' ? p.label.trim() : ''
  return label || personId.value
})

const personAliases = computed(() => {
  const a = personNode.value?.properties?.aliases
  if (!Array.isArray(a)) return ''
  const parts = a
    .filter((x): x is string => typeof x === 'string' && x.trim().length > 0)
    .map((x) => x.trim())
  return parts.length > 0 ? parts.join(', ') : ''
})

const personDescription = computed(() => {
  const d = personNode.value?.properties?.description
  return typeof d === 'string' && d.trim() ? d.trim() : ''
})

// #1048 — the role badge (host / guest / mention) per PRD-029 FR1.
const personRole = computed(() => personRoleFromNode(personNode.value))
const personRoleLabel = computed(() => {
  const r = personRole.value
  if (!r) return ''
  return r.charAt(0).toUpperCase() + r.slice(1)
})

function onPrefillSearch(): void {
  const q = personName.value.trim()
  if (!q) return
  emit('prefillSemanticSearch', { query: q })
}
</script>

<template>
  <div
    class="flex min-h-0 min-w-0 flex-1 flex-col"
    :class="props.embedded ? '' : 'mx-3 overflow-hidden'"
    role="region"
    :aria-label="props.view === 'positions' ? 'Person positions' : 'Person'"
    :data-testid="props.view === 'positions' ? 'person-landing-positions-view' : 'person-landing-view'"
  >
    <div class="mt-1 shrink-0 border-b border-border pb-2">
      <div v-if="!props.embedded" class="flex items-baseline gap-2">
        <span
          class="text-[10px] font-semibold uppercase tracking-wider text-muted"
        >Person</span>
        <h2
          class="min-w-0 flex-1 truncate text-xs font-semibold text-surface-foreground"
          data-testid="person-landing-view-name"
          :title="personName"
        >
          {{ personName }}
        </h2>
        <span
          v-if="personRoleLabel"
          class="rounded bg-overlay px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wider text-surface-foreground"
          data-testid="person-landing-role"
          :data-role="personRole"
          :title="`Role: ${personRoleLabel}`"
        >
          {{ personRoleLabel }}
        </span>
        <button
          type="button"
          class="shrink-0 self-center rounded border border-border px-1.5 py-0.5 text-xs font-medium text-elevated-foreground hover:bg-overlay"
          data-testid="subject-rail-close"
          aria-label="Close person detail"
          @click="emit('closeSubject')"
        >
          ×
        </button>
      </div>
      <component
        :is="section.component"
        v-for="section in personProfileSections.list().filter((x) => x.region === 'header')"
        :key="section.id"
        :person-id="personId"
        :view="props.view"
      />
    </div>
    <div
      :id="props.view === 'positions' ? 'person-landing-panel-positions' : 'person-landing-panel-profile'"
      :data-testid="props.view === 'positions'
        ? 'person-landing-panel-positions'
        : 'person-landing-panel-profile'"
      class="min-h-0 flex-1 space-y-3 overflow-y-auto px-1 py-2"
    >
      <!-- Profile sections: hidden in the positions view -->
      <template v-if="props.view !== 'positions'">
        <!-- Role badge (host / guest / mentioned) — the embedded header (owned
             by NodeDetail) doesn't carry it, so surface it here so the reader
             knows how this person relates to the corpus. Host stands out. -->
        <span
          v-if="personRoleLabel"
          class="inline-flex items-center self-start rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider"
          :class="personRole === 'host'
            ? 'bg-primary text-primary-foreground'
            : 'bg-overlay text-surface-foreground'"
          data-testid="person-landing-role-embedded"
          :data-role="personRole"
        >{{ personRoleLabel }}</span>
        <component
          :is="section.component"
          v-for="section in personProfileSections.list().filter((x) => x.region === 'profile-head')"
          :key="section.id"
          :person-id="personId"
          :view="props.view"
        />
        <p
          v-if="personAliases"
          class="text-[11px] text-muted"
          data-testid="person-landing-aliases"
        >
          Aliases: {{ personAliases }}
        </p>
        <p
          v-if="personDescription"
          class="text-[11px] leading-snug text-surface-foreground"
          data-testid="person-landing-description"
        >
          {{ personDescription }}
        </p>
        <component
          :is="section.component"
          v-for="section in personProfileSections.list().filter((x) => x.region === 'profile')"
          :key="section.id"
          :person-id="personId"
          :view="props.view"
        />
      </template>

      <template v-if="props.view === 'positions'">
        <component
          :is="section.component"
          v-for="section in personProfileSections.list().filter((x) => x.region === 'positions')"
          :key="section.id"
          :person-id="personId"
          :view="props.view"
        />
      </template>

      <template v-if="props.view === 'full'">
        <component
          :is="section.component"
          v-for="section in personProfileSections.list().filter((x) => x.region === 'full')"
          :key="section.id"
          :person-id="personId"
          :view="props.view"
        />
      </template>

      <div v-if="!props.embedded" class="flex shrink-0 flex-wrap gap-2 pt-2">
        <button
          type="button"
          class="rounded border border-border px-2 py-1 text-[11px] font-medium hover:bg-overlay"
          data-testid="person-landing-go-graph"
          @click="emit('goGraph')"
        >
          Open in graph
        </button>
        <button
          type="button"
          class="rounded border border-border px-2 py-1 text-[11px] font-medium hover:bg-overlay"
          data-testid="person-landing-prefill-search"
          @click="onPrefillSearch"
        >
          Prefill semantic search
        </button>
      </div>
    </div>
  </div>
</template>
