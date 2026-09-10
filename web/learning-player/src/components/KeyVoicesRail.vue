<script setup lang="ts">
/**
 * Key-voices rail (UXS-012, wave-G) — the people most present in the signed-in user's own corpus,
 * a horizontal rail of avatar chips linking to each person's card. The per-USER flavor of "key
 * voices" (prominence, not clustering); the per-TOPIC flavor lives on the topic card.
 *
 * Passive: it fetches on mount and simply renders nothing when there is nothing to show (signed
 * out, or no graph-carrying listening yet) — a rail is a claim, so no data means no rail rather
 * than an empty shell.
 */
import { onMounted, ref } from "vue"
import { useI18n } from "vue-i18n"
import { RouterLink } from "vue-router"

import { getKeyVoices } from "../services/api"
import type { KeyVoice } from "../services/types"
import ProfileAvatar from "./ProfileAvatar.vue"

const { t } = useI18n()
const voices = ref<KeyVoice[]>([])

onMounted(async () => {
  try {
    voices.value = (await getKeyVoices()).voices
  } catch {
    voices.value = [] // passive surface — a failure hides the rail, never blocks the page
  }
})
</script>

<template>
  <section v-if="voices.length" class="mt-6" data-testid="key-voices-rail">
    <h2 class="lp-section mb-2">{{ t("keyVoices.title") }}</h2>
    <ul class="flex gap-3 overflow-x-auto pb-1">
      <li v-for="v in voices" :key="v.id" class="shrink-0">
        <RouterLink
          :to="{ name: 'person', params: { id: v.id } }"
          class="flex w-20 flex-col items-center gap-1 no-underline"
          :aria-label="v.label"
          data-testid="key-voice"
        >
          <ProfileAvatar :name="v.label" :src="v.image_url" :size="48" />
          <span class="line-clamp-2 text-center text-xs font-medium text-canvas-foreground">
            {{ v.label }}
          </span>
        </RouterLink>
      </li>
    </ul>
  </section>
</template>
