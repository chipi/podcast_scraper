<script setup lang="ts">
import { computed } from 'vue'
import type { ParsedArtifact } from '../../types/artifact'
import { truncate } from '../../utils/formatting'
import { findRawNodeInArtifact, nodeLabel } from '../../utils/parsing'

defineEmits<{ close: [] }>()

const props = defineProps<{
  viewArtifact: ParsedArtifact | null
  nodeId: string | null
}>()

const rows = computed(() => {
  const art = props.viewArtifact
  const id = props.nodeId
  if (!art || id == null) return []
  const n = findRawNodeInArtifact(art, id)
  if (!n) return []
  const out: { k: string; v: string }[] = [
    { k: 'id', v: String(n.id ?? '') },
    { k: 'type', v: String(n.type ?? '?') },
    { k: 'display', v: nodeLabel(n) },
  ]
  const p = n.properties
  if (p && typeof p === 'object') {
    const keys = Object.keys(p).sort()
    for (const k of keys) {
      let v: unknown = p[k]
      if (v === null || v === undefined) v = ''
      else if (typeof v === 'object') {
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
  }
  return out
})
</script>

<template>
  <aside
    v-if="nodeId && rows.length"
    class="relative z-20 border-l border-border bg-elevated p-3 text-sm text-elevated-foreground shadow-lg"
  >
    <div class="mb-2 flex items-center justify-between">
      <span class="font-semibold text-surface-foreground">Node</span>
      <button
        type="button"
        class="text-xs text-muted hover:text-canvas-foreground"
        aria-label="Close detail"
        @click="$emit('close')"
      >
        ✕
      </button>
    </div>
    <dl class="graph-node-detail-dl space-y-1">
      <template v-for="(r, i) in rows" :key="i">
        <dt class="text-xs font-medium text-muted">
          {{ r.k }}
        </dt>
        <dd class="break-words whitespace-pre-wrap font-mono text-xs">
          {{ r.v }}
        </dd>
      </template>
    </dl>
  </aside>
</template>
