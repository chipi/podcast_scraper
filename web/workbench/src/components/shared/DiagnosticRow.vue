<script setup lang="ts">
/**
 * Pre-#656 foundation: shared diagnostic-row primitive.
 *
 * Three different #656 items (bridge partition indicator, pipeline
 * cleanup metrics panel, ad-excision diagnostic) all render "small
 * label + value (+ optional badge / tooltip)" rows on per-episode and
 * per-run surfaces. Without a shared primitive, we'd duplicate styling
 * three times across three PRs and drift on spacing / typography.
 *
 * Also reused by ``EpisodeDetailPanel``'s existing "Troubleshooting"
 * dialog — its inline ``<dl>/<dt>/<dd>`` rows migrate here too so
 * future changes only have one place to edit.
 *
 * Rendering:
 *   - default layout is ``<dt>`` (label, sans serif, muted) above
 *     ``<dd>`` (value, monospace, wrap-safe) — dense diagnostic vibe.
 *   - ``kind`` controls an optional right-aligned chip: ``default``
 *     renders nothing; ``info`` / ``warning`` / ``success`` render a
 *     colored pill meant for one-to-two-word status tags.
 *
 * Accessibility:
 *   - Always a semantic ``<dl>``-child pair so screen readers
 *     announce it as "label / value" regardless of variant.
 *   - ``tooltip`` maps to the native ``title`` attribute on the whole
 *     row — accessible without extra ARIA noise.
 *
 * Safety:
 *   - All text renders via ``{{ … }}`` interpolation; no ``v-html``
 *     anywhere. Values from the API / pipeline are treated as
 *     untrusted strings.
 */

export type DiagnosticKind = 'default' | 'info' | 'warning' | 'success'

defineProps<{
  label: string
  value: string | number
  /** Optional chip kind for status emphasis. ``default`` hides the chip. */
  kind?: DiagnosticKind
  /** Short label shown inside the chip (e.g. ``"ml"``, ``"filtered"``). */
  badge?: string
  /** Native ``title`` tooltip for hover-help without opening a dialog. */
  tooltip?: string
  /** Stable hook for viewer e2e. */
  dataTestid?: string
}>()

const KIND_CHIP_CLASS: Record<DiagnosticKind, string> = {
  default: '',
  info: 'bg-primary/15 text-primary border border-primary/30',
  warning: 'bg-warning/15 text-warning border border-warning/30',
  success: 'bg-success/15 text-success border border-success/30',
}
</script>

<template>
  <div
    class="flex items-baseline justify-between gap-2 py-0.5"
    :title="tooltip || undefined"
    :data-testid="dataTestid"
  >
    <dt class="font-sans text-[10px] font-medium text-muted">
      {{ label }}
    </dt>
    <div class="flex items-baseline gap-1.5">
      <dd class="break-words font-mono text-[10px] leading-snug text-elevated-foreground">
        {{ value }}
      </dd>
      <span
        v-if="badge && kind && kind !== 'default'"
        :class="[
          'inline-flex items-center rounded-full px-1.5 py-0.5 font-sans text-[9px] font-semibold',
          KIND_CHIP_CLASS[kind],
        ]"
      >
        {{ badge }}
      </span>
    </div>
  </div>
</template>
