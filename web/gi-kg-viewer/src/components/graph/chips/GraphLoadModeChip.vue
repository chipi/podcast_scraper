<script setup lang="ts">
/**
 * graph-v3 tier 8-5 — load-mode toggle chip.
 *
 * Currently a **noop** — both `Top-down` and `Everything` render the
 * same graph. The chip ships now so tier 8-1..8-4/8-6 diffs can wire
 * behaviour to the mode without touching bottom-bar / popover layout.
 *
 * `Shift+E` toggles the mode from anywhere in the graph (bound in
 * GraphCanvas). Persists via USERPREFS-1 (`graphLoadMode`).
 */
import { computed } from 'vue'
import { useGraphLoadModeStore } from '../../../stores/graphLoadMode'

const loadMode = useGraphLoadModeStore()

const label = computed(() =>
  loadMode.isTopDown ? 'Load-mode: Top-down ▾' : 'Load-mode: Everything ▾',
)

const title = computed(() =>
  loadMode.isTopDown
    ? 'Top-down: mount only super-themes; expand on tap. Click to switch to Everything (Shift+E)'
    : 'Everything: mount the full merged graph. Click to switch to Top-down (Shift+E)',
)
</script>

<template>
  <button
    type="button"
    class="inline-flex shrink-0 items-center gap-1 rounded border border-border bg-surface px-2 py-0.5 text-[11px] leading-none text-surface-foreground hover:bg-overlay"
    data-testid="graph-load-mode-chip"
    :aria-pressed="loadMode.isTopDown"
    :title="title"
    @click="loadMode.toggleMode()"
  >
    {{ label }}
  </button>
</template>
