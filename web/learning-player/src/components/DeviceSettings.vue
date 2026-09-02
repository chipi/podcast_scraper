<script setup lang="ts">
/**
 * Device section of the profile (#1905) — settings that belong to THIS PHONE rather than to the
 * account.
 *
 * Deliberately not in `userPreferences` (which syncs through /api/app/preferences and follows the
 * account across devices): a metered phone and an unmetered tablet must be able to disagree. And
 * deliberately NOT namespaced per user — whoever is holding the device gets to say how it uses
 * their data plan, so these are shared by every account that signs in here. That is the opposite
 * rule from the downloads registry itself, which IS per-account because the list of downloaded
 * episodes is listening history.
 *
 * Native only: on the web build there is no offline audio, so these would promise nothing.
 */
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import {
  DEFAULT_POLICY,
  applyDownloadCap,
  getNetworkPolicy,
  setNetworkPolicy,
  type NetworkPolicy,
} from '../services/downloadScheduler'
import { CAP_CHOICES, DEFAULT_CAP_BYTES, getDownloadCap } from '../services/downloads'
import { isNative } from '../services/native'
import { useDownloadsStore } from '../stores/downloads'

const { t } = useI18n()
const native = isNative()
const downloads = useDownloadsStore()
const policy = ref<NetworkPolicy>(DEFAULT_POLICY)
const cap = ref<number>(DEFAULT_CAP_BYTES)
const gb = (bytes: number): number => Math.round(bytes / (1024 * 1024 * 1024))
const usedMb = computed(() => (downloads.bytesOnDisk / (1024 * 1024)).toFixed(0))

onMounted(async () => {
  if (!native) return
  await downloads.ensureLoaded()
  policy.value = await getNetworkPolicy()
  cap.value = await getDownloadCap()
})

async function chooseCap(bytes: number): Promise<void> {
  cap.value = bytes
  // Raising the cap should release whatever was refused for want of room.
  await applyDownloadCap(bytes)
}

async function choosePolicy(next: NetworkPolicy): Promise<void> {
  policy.value = next
  // Relaxing the policy releases whatever was waiting, rather than making the user hunt for it.
  await setNetworkPolicy(next)
}
</script>

<template>
  <section v-if="native" data-testid="device-settings" class="mt-6">
    <h2 class="lp-section mb-1">{{ t('profile.device') }}</h2>
    <p class="mb-3 text-sm text-muted">{{ t('profile.deviceHelp') }}</p>

    <fieldset>
      <legend class="mb-2 text-sm text-muted">{{ t('downloads.network') }}</legend>
      <div class="flex gap-2">
        <button
          v-for="opt in (['wifi-only', 'any'] as NetworkPolicy[])"
          :key="opt"
          type="button"
          :data-testid="`device-policy-${opt}`"
          class="flex-1 rounded-full border px-3 py-2 text-sm"
          :class="
            policy === opt
              ? 'border-accent text-accent'
              : 'border-border text-muted hover:text-canvas-foreground'
          "
          :aria-pressed="policy === opt"
          @click="choosePolicy(opt)"
        >
          {{ opt === 'wifi-only' ? t('downloads.wifiOnly') : t('downloads.wifiAndCellular') }}
        </button>
      </div>
      <p class="lp-kicker mt-2">{{ t('downloads.networkHint') }}</p>
    </fieldset>

    <fieldset class="mt-4">
      <legend class="mb-2 text-sm text-muted">{{ t('downloads.cap') }}</legend>
      <div class="flex gap-2">
        <button
          v-for="choice in CAP_CHOICES"
          :key="choice"
          type="button"
          :data-testid="`device-cap-${gb(choice)}`"
          class="flex-1 rounded-full border px-3 py-2 text-sm"
          :class="
            cap === choice
              ? 'border-accent text-accent'
              : 'border-border text-muted hover:text-canvas-foreground'
          "
          :aria-pressed="cap === choice"
          @click="chooseCap(choice)"
        >
          {{ t('downloads.gb', { n: gb(choice) }) }}
        </button>
      </div>
      <p class="lp-kicker mt-2">{{ t('downloads.capHint') }}</p>
    </fieldset>

    <dl class="mt-4 flex flex-col gap-2 text-sm">
      <div class="flex items-center justify-between gap-3">
        <dt class="text-muted">{{ t('downloads.storageUsed') }}</dt>
        <dd class="font-semibold tabular-nums" data-testid="device-storage">{{ usedMb }} MB</dd>
      </div>
    </dl>
  </section>
</template>
