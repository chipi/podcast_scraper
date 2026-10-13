<script setup lang="ts">
/**
 * Profile › Interests: what the listener asked Discover to suggest less of (ADR-163 §6) — the place
 * "Less like this" is undone later. Hidden when there is nothing on it; a failed load says so
 * rather than showing an empty list that reads as "nothing hidden".
 */
import { onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { getSuggestLess, undoSuggestLess, type SuggestLessItem } from '../services/api'
import { useGoneReferencesStore } from '../stores/goneReferences'
import GoneBadge from './GoneBadge.vue'

const { t } = useI18n()
const items = ref<SuggestLessItem[]>([])
const failed = ref(false)
const gone = useGoneReferencesStore()

onMounted(async () => {
  void gone.ensureLoaded()
  try {
    items.value = await getSuggestLess()
  } catch {
    failed.value = true
  }
})

async function undo(token: string): Promise<void> {
  try {
    items.value = await undoSuggestLess(token)
  } catch {
    failed.value = true
  }
}
</script>

<template>
  <section v-if="items.length || failed" class="mt-4 rounded-2xl border border-border p-5" data-testid="suggested-less">
    <h2 class="lp-section mb-1">{{ t('profile.suggestedLess') }}</h2>
    <p v-if="failed" class="text-sm text-muted" data-testid="suggested-less-unavailable">{{ t('profile.unavailable') }}</p>
    <template v-else>
      <p class="mb-3 text-sm text-muted">{{ t('profile.suggestedLessHelp') }}</p>
      <ul class="flex flex-wrap gap-2">
        <li v-for="item in items" :key="item.token">
          <button
            type="button"
            class="lp-tap rounded-full bg-overlay px-3 py-1 text-xs hover:bg-elevated"
            :class="{ 'opacity-60': gone.isGone(item.token) }"
            :aria-label="t('profile.suggestedLessUndo', { subject: item.label })"
            data-testid="suggested-less-item"
            @click="undo(item.token)"
          >{{ item.label }}<template v-if="gone.isGone(item.token)"> · <GoneBadge /></template> <span aria-hidden="true" class="ml-1 text-muted">×</span></button>
        </li>
      </ul>
    </template>
  </section>
</template>
