<script setup lang="ts">
/**
 * Enriched-answer chip — Search v3 §S5 (RFC-107, UXS-016). Toggles the
 * ``enrich_results=true`` flag on ``/api/search`` so the server runs the
 * shipped QueryEnricher chain (RFC-088 chunk 5) and decorates hits with
 * ``metadata.query_enrichments.related_topics``. The workspace's
 * ``EnrichedAnswerHero`` above the results renders an aggregated summary
 * when the response carries enrichment output.
 *
 * The chip's default follows ``shell.enrichedSearchAvailable`` (auto-on
 * when the server advertises enrichment capability, auto-off otherwise).
 * The user can still explicitly toggle; the store keeps a tri-state
 * (``null`` = auto, ``true`` / ``false`` = explicit) so we don't overwrite
 * an explicit off just because the capability signal changes.
 *
 * When the server does NOT advertise enrichment
 * (``!shell.enrichedSearchAvailable``), the chip renders visibly disabled
 * with a tooltip pointing at the config knob.
 */
import { computed } from 'vue'
import { useSearchStore } from '../../../stores/search'
import { useShellStore } from '../../../stores/shell'

const search = useSearchStore()
const shell = useShellStore()

const capabilityOn = computed(() => Boolean(shell.enrichedSearchAvailable))

/**
 * Effective on/off: user's explicit choice wins; when unset (null), the
 * chip mirrors the server's capability signal so first-time users see the
 * enricher's output without hunting for a toggle.
 */
const isActive = computed(() => {
  const raw = search.filters.enrichResults
  if (raw === null) return capabilityOn.value
  return raw === true
})

const chipLabel = computed(() => (isActive.value ? 'Enriched ✓' : 'Enriched'))

const chipTitle = computed(() => {
  if (!capabilityOn.value) {
    return 'Enrichment not configured on this server (no ``query_enrichers`` output).'
  }
  return isActive.value
    ? 'Enriched answers on — server runs the QueryEnricher chain and hoists the summary above results.'
    : 'Enriched answers off — click to run the QueryEnricher chain on your next search.'
})

function toggle(): void {
  if (!capabilityOn.value) return
  // Flip from the effective state so the very first click always reads
  // as "off" (the user is overriding the auto-on default) or "on" (they
  // are opting in when auto is off).
  search.filters.enrichResults = !isActive.value
}
</script>

<template>
  <button
    type="button"
    class="inline-flex h-6 items-center rounded border px-2 text-[11px] leading-none transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary disabled:cursor-not-allowed disabled:opacity-40"
    :class="
      isActive
        ? 'border-primary bg-primary text-primary-foreground'
        : 'border-border/70 text-muted hover:bg-overlay'
    "
    :aria-pressed="isActive"
    :disabled="!capabilityOn"
    data-testid="search-chip-enriched"
    aria-label="Enriched answers (QueryEnricher chain)"
    :title="chipTitle"
    @click="toggle"
  >
    {{ chipLabel }}
  </button>
</template>
