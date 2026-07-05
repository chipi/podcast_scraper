<script setup lang="ts">
/**
 * Log-replay control bar + REPLAY banner, shown over the graph while a session is being replayed
 * (driven by ``graphReplay``). Makes it unmistakable this is a captured log, not live, and offers
 * all three controls: step ◀▶, timed play/pause, and a scrub slider.
 */
import { storeToRefs } from 'pinia'
import { useGraphReplayStore } from '../../stores/graphReplay'

const replay = useGraphReplayStore()
const { sessionId, step, total, playing, currentEvent } = storeToRefs(replay)

function label(e: Record<string, unknown> | null): string {
  if (!e) return 'start'
  const a = String(e.action ?? '')
  if (a === 'graph_rail_nav') return `→ ${e.to_id ?? '?'}`
  if (a === 'graph_recenter') return `re-centre ${e.target_id ?? '?'}`
  if (a === 'graph_node_tap') return `tap ${e.id ?? '?'}`
  if (a === 'graph_redraw') return `redraw · ${e.nodes ?? '?'} nodes`
  if (a === 'graph_broke') return `broke: ${e.reason ?? '?'}`
  return a
}
</script>

<template>
  <div
    class="pointer-events-auto absolute inset-x-0 top-0 z-50 flex flex-wrap items-center gap-2 border-b-2 border-warning bg-warning/15 px-3 py-1.5 text-xs backdrop-blur"
    data-testid="graph-replay-bar"
    role="region"
    aria-label="Log replay"
  >
    <span class="font-bold uppercase tracking-wide text-warning">⏮ Replay</span>
    <span class="max-w-[8rem] truncate text-muted">{{ sessionId }}</span>
    <span class="font-semibold text-surface-foreground" data-testid="replay-step">
      step {{ step }} / {{ total }}
    </span>
    <span class="min-w-0 flex-1 truncate text-muted" data-testid="replay-current">
      {{ label(currentEvent) }}
    </span>

    <button
      type="button"
      class="rounded border border-border px-1.5 py-0.5 hover:bg-overlay disabled:opacity-40"
      :disabled="step === 0"
      data-testid="replay-prev"
      aria-label="Previous step"
      @click="replay.prev()"
    >
      ◀
    </button>
    <button
      type="button"
      class="rounded border border-border px-1.5 py-0.5 hover:bg-overlay"
      data-testid="replay-playpause"
      :aria-label="playing ? 'Pause' : 'Play'"
      @click="playing ? replay.pause() : replay.play()"
    >
      {{ playing ? '⏸' : '▶' }}
    </button>
    <button
      type="button"
      class="rounded border border-border px-1.5 py-0.5 hover:bg-overlay disabled:opacity-40"
      :disabled="step >= total"
      data-testid="replay-next"
      aria-label="Next step"
      @click="replay.next()"
    >
      ▶
    </button>
    <input
      type="range"
      min="0"
      :max="total"
      :value="step"
      class="w-28"
      data-testid="replay-scrub"
      aria-label="Scrub replay"
      @input="replay.setStep(Number(($event.target as HTMLInputElement).value))"
    />
    <button
      type="button"
      class="rounded border border-border px-1.5 py-0.5 font-medium hover:bg-overlay"
      data-testid="replay-exit"
      @click="replay.exit()"
    >
      Exit
    </button>
  </div>
</template>
