<script setup lang="ts">
/**
 * Dev↔prod target switch (#1310, guide §5 / Orrery ADR-083). A small pill in the header, shown ONLY
 * in an internal native build (`tierSwitchEnabled()`) — never on the web or in a prod-locked release.
 * Flipping it repoints the API base + error telemetry (services/tier.ts) and reloads so both
 * re-resolve. dev = the local machine (make serve-app); prod = the live player.
 */
import { computed, ref } from 'vue'
import { getTier, setTier, tierSwitchEnabled, type Tier } from '../services/tier'

const enabled = tierSwitchEnabled()
const tier = ref<Tier>(getTier())
const label = computed(() => (tier.value === 'dev' ? 'DEV' : 'PROD'))

function toggle(): void {
  const next: Tier = tier.value === 'dev' ? 'prod' : 'dev'
  setTier(next)
  tier.value = next
  // Reload so api.ts BASE + Sentry re-resolve against the new tier.
  window.location.reload()
}
</script>

<template>
  <button
    v-if="enabled"
    type="button"
    data-testid="tier-switch"
    class="shrink-0 rounded-full border px-1.5 py-px text-[9px] font-bold tracking-wide transition"
    :class="
      tier === 'dev'
        ? 'border-danger text-danger hover:bg-danger/10'
        : 'border-border text-muted hover:bg-overlay'
    "
    :title="`Target: ${label} — tap to switch (internal build only)`"
    :aria-label="`Backend target ${label}, tap to switch`"
    @click="toggle"
  >
    {{ label }}
  </button>
</template>
