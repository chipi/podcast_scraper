<script setup lang="ts">
import { computed } from 'vue'
import type { ParsedArtifact } from '../../types/artifact'
import { graphNodeTypeChrome } from '../../utils/colors'
import { truncate } from '../../utils/formatting'
import {
  findRawNodeInArtifact,
  fullPrimaryNodeLabel,
  nodeLabel,
} from '../../utils/parsing'
import { graphTypeAvatarLetter } from '../../utils/graphTypeAvatar'
import HelpTip from '../shared/HelpTip.vue'
import {
  SEARCH_RESULT_DIAGNOSTICS_HELP_CHIP_CLASS,
  SEARCH_RESULT_EPISODE_ID_BUTTON_CLASS,
} from '../../utils/searchResultActionStyles'
import { visualGroupForNode } from '../../utils/visualGroup'
import GraphConnectionsSection from './GraphConnectionsSection.vue'

const emit = defineEmits<{ close: []; 'go-graph': [] }>()

const props = defineProps<{
  viewArtifact: ParsedArtifact | null
  nodeId: string | null
  /** Embedded in App right rail: full width, no fixed 280px strip. */
  embedInRail?: boolean
}>()

const HIDDEN_PROPS = new Set([
  'slug',
  'label',
  'name',
  'text',
  'title',
  'description',
  'entity_kind',
])

const node = computed(() => {
  const art = props.viewArtifact
  const id = props.nodeId
  if (!art || id == null) return null
  return findRawNodeInArtifact(art, id)
})

const displayName = computed(() => {
  const n = node.value
  if (!n) return ''
  return nodeLabel(n)
})

/** Uncapped primary label for native tooltip when the heading uses ``line-clamp``. */
const displayNameFull = computed(() => {
  const n = node.value
  if (!n) return ''
  return fullPrimaryNodeLabel(n)
})

const nodeType = computed(() => {
  const n = node.value
  if (!n) return '?'
  return String(n.type ?? '?')
})

const visualType = computed(() => visualGroupForNode(node.value))

const nodeTypeAvatarStyle = computed((): Record<string, string> => {
  const c = graphNodeTypeChrome(visualType.value)
  return {
    backgroundColor: c.background,
    border: `3px solid ${c.border}`,
    color: c.labelColor,
  }
})

const avatarLetter = computed(() => graphTypeAvatarLetter(visualType.value))

const graphNodeIdTooltip = computed((): string => {
  const id = props.nodeId
  if (!id) return ''
  return (
    `Graph node id (Cytoscape / merged graph): ${id}. ` +
    'Same id as in artifact edges; use for support and tooling.'
  )
})

const entityKind = computed(() => {
  const p = node.value?.properties
  if (!p) return null
  const ek = p.entity_kind
  return typeof ek === 'string' && ek.trim() ? ek.trim() : null
})

const bodyText = computed(() => {
  const n = node.value
  if (!n) return null
  const p = n.properties
  if (!p) return null
  const explicit = p.text ?? p.description
  if (typeof explicit === 'string' && explicit.trim()) return explicit.trim()
  const label = p.label ?? p.name ?? p.title
  if (typeof label === 'string' && label.length > displayName.value.length) {
    return label.trim()
  }
  return null
})

/** Omit body paragraph when it only repeats the full title (rail already shows 2-line clamp + tooltip). */
const bodyTextForTemplate = computed(() => {
  const b = bodyText.value
  if (!b) return null
  const full = displayNameFull.value.trim()
  if (full && b.trim() === full) return null
  return b
})

/**
 * ``entity_kind`` is often ``episode`` on grounded insights; do not show that as a second title under **Insight**.
 */
const showEntityKindSubtitle = computed(() => {
  const ek = entityKind.value
  if (!ek?.trim()) return false
  const nt = nodeType.value
  if (nt !== 'Episode' && ek.trim().toLowerCase() === 'episode') {
    return false
  }
  return true
})

const showTypeSubtitleRow = computed(
  () =>
    (!props.embedInRail && Boolean(nodeType.value)) || showEntityKindSubtitle.value,
)

const extraProps = computed(() => {
  const p = node.value?.properties
  if (!p || typeof p !== 'object') return []
  const out: { k: string; v: string }[] = []
  for (const k of Object.keys(p).sort()) {
    if (HIDDEN_PROPS.has(k)) continue
    // Corpus episode id is redundant with the **E** header chip on non-Episode nodes (Insight, Quote, …).
    if (nodeType.value !== 'Episode' && k === 'episode_id') continue
    let v: unknown = p[k]
    if (v === null || v === undefined || v === '') continue
    if (typeof v === 'boolean') {
      v = v ? 'yes' : 'no'
    } else if (typeof v === 'object') {
      try {
        v = JSON.stringify(v)
      } catch {
        v = String(v)
      }
    } else {
      v = String(v)
    }
    out.push({ k, v: truncate(String(v), 400) })
  }
  return out
})

const nodeDiagnosticsEntries = computed((): { label: string; value: string }[] => {
  const id = props.nodeId
  const n = node.value
  if (!id || !n) return []
  const rows: { label: string; value: string }[] = [
    { label: 'Graph node id', value: id },
    { label: 'Type', value: nodeType.value },
    { label: 'Visual group', value: visualType.value },
  ]
  if (entityKind.value) {
    rows.push({ label: 'Entity kind', value: entityKind.value })
  }
  return rows
})
</script>

<template>
  <aside
    v-if="nodeId && node"
    class="relative z-20 text-surface-foreground"
    :class="
      props.embedInRail
        ? 'flex min-h-0 w-full min-w-0 flex-1 flex-col border-0 bg-surface shadow-none'
        : 'border-l border-border bg-elevated text-elevated-foreground shadow-lg'
    "
    :style="props.embedInRail ? undefined : { width: '280px' }"
  >
    <div
      class="flex items-start justify-between gap-2"
      :class="props.embedInRail ? 'border-b border-border px-2 py-2' : 'border-b border-border px-3 py-2'"
    >
      <div class="flex min-w-0 flex-1 gap-3">
        <div
          class="flex h-[4.5rem] w-[4.5rem] shrink-0 items-center justify-center rounded-2xl text-2xl font-black leading-none shadow-md ring-1 ring-black/15 dark:ring-white/15"
          :style="nodeTypeAvatarStyle"
          aria-hidden="true"
        >
          {{ avatarLetter }}
        </div>
        <div class="min-w-0 flex-1">
          <div class="flex min-w-0 items-start justify-between gap-1">
            <h3
              class="min-w-0 flex-1 break-words text-base font-semibold leading-snug text-surface-foreground line-clamp-2"
              :title="displayNameFull || displayName"
            >
              {{ displayName }}
            </h3>
            <div class="ml-1 flex shrink-0 items-start gap-0.5 pt-0.5">
              <button
                v-if="nodeId"
                type="button"
                :class="SEARCH_RESULT_EPISODE_ID_BUTTON_CLASS"
                :aria-label="graphNodeIdTooltip"
                :title="graphNodeIdTooltip"
                @click.stop.prevent
              >
                E
              </button>
              <HelpTip
                v-if="nodeId"
                :pref-width="300"
                :button-class="SEARCH_RESULT_DIAGNOSTICS_HELP_CHIP_CLASS"
                button-aria-label="Graph node diagnostics"
              >
                <p class="mb-2 font-sans text-[11px] font-semibold text-surface-foreground">
                  Troubleshooting
                </p>
                <p class="mb-2 font-sans text-[10px] text-muted">
                  Ids and typing for support — same node the graph and artifact use.
                </p>
                <dl class="space-y-1.5 font-mono text-[10px] leading-snug">
                  <template v-for="(row, di) in nodeDiagnosticsEntries" :key="di">
                    <dt class="font-sans font-medium text-muted">
                      {{ row.label }}
                    </dt>
                    <dd class="break-words text-elevated-foreground">
                      {{ row.value }}
                    </dd>
                  </template>
                </dl>
              </HelpTip>
              <button
                v-if="!props.embedInRail"
                type="button"
                class="shrink-0 text-xs text-muted hover:text-canvas-foreground"
                aria-label="Close detail"
                @click="emit('close')"
              >
                ✕
              </button>
            </div>
          </div>
          <p
            v-if="showTypeSubtitleRow"
            class="mt-1 text-xs text-muted"
          >
            <span
              v-if="!props.embedInRail"
              class="inline-block rounded bg-primary/10 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-primary"
            >{{ nodeType }}</span>
            <template v-if="showEntityKindSubtitle">
              <span v-if="!props.embedInRail" class="mx-1 text-muted">·</span>
              <span>{{ entityKind }}</span>
            </template>
          </p>
        </div>
      </div>
    </div>

    <div
      class="px-2 py-2"
      :class="props.embedInRail ? 'min-h-0 flex-1 overflow-y-auto' : 'overflow-y-auto px-3'"
      :style="props.embedInRail ? undefined : { maxHeight: 'calc(100vh - 12rem)' }"
    >
      <p
        v-if="bodyTextForTemplate"
        class="mb-3 text-xs leading-relaxed text-muted"
      >
        {{ truncate(bodyTextForTemplate, 600) }}
      </p>

      <dl
        v-if="extraProps.length"
        class="mb-2 space-y-1.5"
      >
        <template v-for="(r, i) in extraProps" :key="i">
          <dt class="text-[10px] font-medium uppercase tracking-wide text-muted">
            {{ r.k }}
          </dt>
          <dd class="break-words text-xs text-surface-foreground">
            {{ r.v }}
          </dd>
        </template>
      </dl>

      <GraphConnectionsSection
        class="mt-2"
        :view-artifact="props.viewArtifact"
        :node-id="props.nodeId"
        @go-graph="emit('go-graph')"
      />
    </div>
  </aside>
</template>
