<script setup lang="ts">
import { computed } from 'vue'
import type { ParsedArtifact, RawGraphEdge } from '../../types/artifact'
import { truncate } from '../../utils/formatting'
import { findRawNodeInArtifact, nodeLabel } from '../../utils/parsing'

defineEmits<{ close: [] }>()

const props = defineProps<{
  viewArtifact: ParsedArtifact | null
  nodeId: string | null
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

const nodeType = computed(() => {
  const n = node.value
  if (!n) return '?'
  return String(n.type ?? '?')
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

const extraProps = computed(() => {
  const p = node.value?.properties
  if (!p || typeof p !== 'object') return []
  const out: { k: string; v: string }[] = []
  for (const k of Object.keys(p).sort()) {
    if (HIDDEN_PROPS.has(k)) continue
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

interface Neighbor {
  id: string
  label: string
  type: string
  edgeType: string
  direction: 'in' | 'out'
}

const neighbors = computed((): Neighbor[] => {
  const art = props.viewArtifact
  const id = props.nodeId
  if (!art?.data || id == null) return []
  const edges: RawGraphEdge[] = Array.isArray(art.data.edges) ? art.data.edges : []
  const sid = String(id)
  const out: Neighbor[] = []
  const seen = new Set<string>()
  for (const e of edges) {
    if (!e) continue
    const from = e.from != null ? String(e.from) : ''
    const to = e.to != null ? String(e.to) : ''
    let neighborId: string | null = null
    let direction: 'in' | 'out' = 'out'
    if (from === sid && to && to !== sid) {
      neighborId = to
      direction = 'out'
    } else if (to === sid && from && from !== sid) {
      neighborId = from
      direction = 'in'
    }
    if (!neighborId) continue
    const key = `${neighborId}:${direction}:${e.type ?? ''}`
    if (seen.has(key)) continue
    seen.add(key)
    const nNode = findRawNodeInArtifact(art, neighborId)
    out.push({
      id: neighborId,
      label: nNode ? nodeLabel(nNode) : neighborId,
      type: nNode ? String(nNode.type ?? '?') : '?',
      edgeType: String(e.type ?? ''),
      direction,
    })
  }
  return out
})
</script>

<template>
  <aside
    v-if="nodeId && node"
    class="relative z-20 border-l border-border bg-elevated text-elevated-foreground shadow-lg"
    style="width: 280px"
  >
    <div class="flex items-start justify-between border-b border-border px-3 py-2">
      <div class="min-w-0 flex-1">
        <div class="mb-0.5 flex items-center gap-1.5">
          <span
            class="inline-block rounded bg-primary/15 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-primary"
          >{{ nodeType }}</span>
          <span
            v-if="entityKind"
            class="text-[10px] text-muted"
          >{{ entityKind }}</span>
        </div>
        <p class="text-sm font-medium leading-snug text-surface-foreground">
          {{ displayName }}
        </p>
      </div>
      <button
        type="button"
        class="ml-2 shrink-0 text-xs text-muted hover:text-canvas-foreground"
        aria-label="Close detail"
        @click="$emit('close')"
      >
        ✕
      </button>
    </div>

    <div class="overflow-y-auto px-3 py-2" style="max-height: calc(100vh - 12rem)">
      <p
        v-if="bodyText"
        class="mb-2 text-xs leading-relaxed text-surface-foreground"
      >
        {{ truncate(bodyText, 600) }}
      </p>

      <dl
        v-if="extraProps.length"
        class="mb-2 space-y-1"
      >
        <template v-for="(r, i) in extraProps" :key="i">
          <dt class="text-[10px] font-medium uppercase tracking-wide text-muted">
            {{ r.k }}
          </dt>
          <dd class="break-words text-xs">
            {{ r.v }}
          </dd>
        </template>
      </dl>

      <div
        v-if="neighbors.length"
        class="border-t border-border pt-2"
      >
        <p class="mb-1 text-[10px] font-semibold uppercase tracking-wide text-muted">
          Connections ({{ neighbors.length }})
        </p>
        <ul class="space-y-1">
          <li
            v-for="nb in neighbors"
            :key="`${nb.id}-${nb.direction}-${nb.edgeType}`"
            class="flex items-start gap-1 text-[11px] leading-snug"
          >
            <span class="shrink-0 text-muted">{{ nb.direction === 'out' ? '→' : '←' }}</span>
            <span>
              <span class="text-surface-foreground">{{ truncate(nb.label, 50) }}</span>
              <span
                v-if="nb.edgeType"
                class="ml-1 text-[10px] text-muted"
              >({{ nb.edgeType }})</span>
            </span>
          </li>
        </ul>
      </div>

      <p class="mt-2 select-all break-all font-mono text-[10px] text-muted/60">
        {{ nodeId }}
      </p>
    </div>
  </aside>
</template>
