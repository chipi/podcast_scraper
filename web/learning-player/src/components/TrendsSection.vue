<script setup lang="ts">
/**
 * Trends — the titled `DiscoveryExplorer` section on Discover: 3 rows on a phone and 5 on desktop,
 * more rows expanding in place from the header's "all ›". Home rendered it too until 2026-10-07,
 * when Trends became Discover-only (operator); the page decides what a tap opens.
 */
import { ref } from "vue"
import { useI18n } from "vue-i18n"
import DiscoveryExplorer from "./DiscoveryExplorer.vue"
import { useIsDesktop } from "../composables/useMediaQuery"

type Kind = "topic" | "theme" | "storyline" | "person"

const props = defineProps<{
  /** Which kind to open on (Discover's `?trends=`). */
  kind?: Kind
}>()
const emit = defineEmits<{ (e: "open", payload: { kind: Kind; id: string; rank: number }): void }>()

const { t } = useI18n()
const isDesktop = useIsDesktop()

/** The section element, so Discover can scroll it into view for a `?trends=` deep link. */
const trendsEl = ref<HTMLElement | null>(null)
defineExpose({ trendsEl })
</script>

<template>
  <section
    id="trends"
    ref="trendsEl"
    class="mt-7 scroll-mt-4"
    data-testid="browse-discovery"
  >
    <!-- A prop cannot follow a media query in CSS, hence `useIsDesktop`. -->
    <DiscoveryExplorer
      :collapsed="isDesktop ? 5 : 3"
      see-all
      :kind="props.kind"
      :title="t('browse.trendsTitle')"
      @open="emit('open', $event)"
    />
  </section>
</template>
