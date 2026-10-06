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
import { personName } from "../utils/personName"
import CardRail from "./CardRail.vue"
import ProfileAvatar from "./ProfileAvatar.vue"
import SectionHeading from "./SectionHeading.vue"

const { t } = useI18n()
const voices = ref<KeyVoice[]>([])

onMounted(async () => {
  try {
    // Cased on arrival rather than in the template: `v.label` is read three times below
    // (aria-label, avatar initials, visible text) and they must not disagree.
    voices.value = (await getKeyVoices()).voices.map((v) => ({ ...v, label: personName(v.label) }))
  } catch {
    voices.value = [] // passive surface — a failure hides the rail, never blocks the page
  }
})
</script>

<template>
  <!-- The standard rail (operator 2026-10-05: every rail looks the same on every page) — CardRail,
       SectionHeading, the section spacing every Home rail uses. The tile stays an avatar: a person
       is not a show, and a square-artwork slot around a 48px face would be mostly empty. -->
  <section v-if="voices.length" class="mt-7" data-testid="key-voices-rail">
    <SectionHeading :title="t('keyVoices.title')" />
    <CardRail>
      <li v-for="v in voices" :key="v.id" class="shrink-0">
        <RouterLink
          :to="{ name: 'person', params: { id: v.id } }"
          class="flex w-20 flex-col items-center gap-1 no-underline"
          :aria-label="v.label"
          data-testid="key-voice"
        >
          <ProfileAvatar :name="v.label" :src="v.image_url" :size="48" />
          <span class="lp-tile-title text-center text-xs font-medium text-canvas-foreground">
            {{ v.label }}
          </span>
        </RouterLink>
      </li>
    </CardRail>
  </section>
</template>
