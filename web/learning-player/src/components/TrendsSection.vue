<script setup lang="ts">
/**
 * Trends — the titled `DiscoveryExplorer` section Home and Discover both render (operator
 * 2026-10-05: the two are one screen family and must look identical).
 *
 * Sharing `DiscoveryExplorer` alone was not enough: each page wrapped it itself, and the wrappings
 * drifted — Discover spaced it `mt-4` against Home's `mt-7`, showed 10 rows against 3, and expanded
 * differently. So the wrapping lives here too: one spacing, 3 rows on a phone and 5 on desktop, more
 * rows expanding in place from the header's "all ›". The page decides only what a tap opens (Home:
 * an overlay card; Discover: the page) and keeps its own test id through `prefix`.
 */
import { ref } from "vue"
import { useI18n } from "vue-i18n"
import DiscoveryExplorer from "./DiscoveryExplorer.vue"
import { useIsDesktop } from "../composables/useMediaQuery"

type Kind = "topic" | "theme" | "storyline" | "person"

const props = defineProps<{
  /** `home` or `browse` — selects the test id each page's specs already use. */
  prefix: "home" | "browse"
  /** Which kind to open on (Discover's `?trends=`). */
  kind?: Kind
}>()
const emit = defineEmits<{ (e: "open", payload: { kind: Kind; id: string; rank: number }): void }>()

const { t } = useI18n()
const isDesktop = useIsDesktop()
const home = props.prefix === "home"

/** The section element, so Discover can scroll it into view for a `?trends=` deep link. */
const trendsEl = ref<HTMLElement | null>(null)
defineExpose({ trendsEl })
</script>

<template>
  <section
    id="trends"
    ref="trendsEl"
    class="mt-7 scroll-mt-4"
    :data-testid="home ? 'home-discovery' : 'browse-discovery'"
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
