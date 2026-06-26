<script setup lang="ts">
/**
 * Profile / account — where the signed-in user sees who they are and edits their personalization,
 * starting with their interest topics (chosen at sign-in via the onboarding card). Auth-gated.
 */
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { getTopClusters, getUserInterests } from '../services/api'
import type { InterestCluster } from '../services/types'
import { useAuthStore } from '../stores/auth'
import InterestsPicker from '../components/InterestsPicker.vue'

const { t } = useI18n()
const auth = useAuthStore()

const interests = ref<string[]>([])
const clusters = ref<InterestCluster[]>([])
const pickerOpen = ref(false)

// Map saved cluster ids → human labels (fall back to a de-slugged id if it's outside the top set).
const interestLabels = computed(() => {
  const byId = new Map(clusters.value.map((c) => [c.id, c.label]))
  return interests.value.map((id) => ({
    id,
    label: byId.get(id) ?? id.replace(/^tc:/, '').replace(/-/g, ' '),
  }))
})

async function load(): Promise<void> {
  const [ints, tops] = await Promise.all([
    getUserInterests().catch(() => [] as string[]),
    getTopClusters(50).catch(() => [] as InterestCluster[]),
  ])
  interests.value = ints
  clusters.value = tops
}

function onSaved(ids: string[]): void {
  interests.value = ids
}

onMounted(load)
</script>

<template>
  <section class="max-w-2xl">
    <h1 class="mb-1 font-display text-3xl font-extrabold tracking-tight">{{ t('profile.title') }}</h1>
    <p class="mb-6 text-muted">{{ auth.user?.name }}<span v-if="auth.user?.email"> · {{ auth.user?.email }}</span></p>

    <section class="rounded-2xl border border-border p-5">
      <div class="mb-3 flex items-center justify-between gap-2">
        <h2 class="lp-kicker">{{ t('profile.interests') }}</h2>
        <button type="button" class="text-sm font-bold text-accent" @click="pickerOpen = true">
          {{ t('profile.editInterests') }}
        </button>
      </div>
      <p class="mb-3 text-sm text-muted">{{ t('profile.interestsHelp') }}</p>
      <div v-if="interestLabels.length" class="flex flex-wrap gap-1.5">
        <span
          v-for="i in interestLabels"
          :key="i.id"
          class="rounded-full bg-overlay px-2.5 py-1 text-xs text-topic"
        >{{ i.label }}</span>
      </div>
      <p v-else class="text-sm text-muted">{{ t('profile.noInterests') }}</p>
    </section>

    <InterestsPicker v-if="pickerOpen" @close="pickerOpen = false" @saved="onSaved" />
  </section>
</template>
