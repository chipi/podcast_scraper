<script setup lang="ts">
/**
 * Visible failure surface for graph handoffs (decision #15 / C7).
 *
 * Replaces today's silent swallow at GraphCanvas.vue:901-903 (now wired through
 * FSM `handoffFailed` in C6). When the FSM's last result is `failed`, this
 * strip renders above the graph canvas with a concise reason and a dismiss
 * button. Stuck-handoff timeouts (decision #16) fail through the same path.
 *
 * Intentionally minimal: ephemeral status; the next successful handoff (or
 * the dismiss click) clears it. The strip is plain HTML — no toast library,
 * no dependency cost. Tests reference `data-testid="handoff-error-strip"`.
 */

import { computed } from 'vue'
import { useGraphHandoffStore } from '../../stores/graphHandoff'

const handoff = useGraphHandoffStore()

const visible = computed(() => handoff.lastResult?.status === 'failed')
const reason = computed(() => handoff.lastResult?.reason ?? 'unknown error')

function dismiss(): void {
  handoff.lastResult = null
}
</script>

<template>
  <div
    v-if="visible"
    role="alert"
    aria-live="polite"
    data-testid="handoff-error-strip"
    class="flex items-center gap-2 px-3 py-2 text-sm border border-warning/40 bg-warning/10 text-warning-foreground"
  >
    <span class="flex-1 font-medium">
      Could not open episode in graph: {{ reason }}
    </span>
    <button
      type="button"
      data-testid="handoff-error-strip-dismiss"
      class="px-2 py-0.5 rounded text-xs font-medium border border-warning/50 hover:bg-warning/20 motion-safe:transition-colors"
      @click="dismiss"
    >
      Dismiss
    </button>
  </div>
</template>
