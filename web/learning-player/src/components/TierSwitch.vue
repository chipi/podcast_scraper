<script setup lang="ts">
/**
 * Dev↔prod target switch (#1310, guide §5 / Orrery ADR-083). A small pill in the header, shown ONLY
 * in an internal native build (`tierSwitchEnabled()`) — never on the web or in a prod-locked release.
 * Flipping it repoints the API base + error telemetry (services/tier.ts) and reloads so both
 * re-resolve. dev = the local machine (make serve-app); prod = the live player.
 */
import { computed, ref } from 'vue'
import { getTier, isTargetingProd, resolveApiBase, setTier, tierSwitchEnabled, type Tier } from '../services/tier'

const enabled = tierSwitchEnabled()
const tier = ref<Tier>(getTier())
/*
 * The label names where requests ACTUALLY go, not what the switch is set to.
 *
 * `getTier()` alone said PROD on every simulator and e2e build — those bake `VITE_API_BASE_URL`,
 * which `resolveApiBase()` honours over the prod base while the stored tier stays 'prod'. So the
 * pill claimed the live backend while the app talked to `127.0.0.1`, which is precisely backwards
 * from what the badge is for: it exists so you can tell, at a glance, which backend you are looking
 * at (operator 2026-09-24).
 */
const label = computed(() => (isTargetingProd() ? 'PROD' : 'DEV'))

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
      label === 'DEV'
        ? 'border-danger text-danger hover:bg-danger/10'
        : 'border-border text-muted hover:bg-overlay'
    "
    :title="`Target: ${label} (${resolveApiBase()}) — tap to switch (internal build only)`"
    :aria-label="`Backend target ${label}, tap to switch`"
    @click="toggle"
  >
    {{ label }}
  </button>
</template>
