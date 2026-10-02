<script setup lang="ts">
/**
 * "Top voices" — the people who drive a topic or a storyline, as a grid of avatar + name.
 *
 * ONE component for both surfaces (operator 2026-09-30). The topic card had this grid while the
 * storyline page listed the same people as plain "Related people" chips — the same data from the
 * same endpoint (`related_people`, server-ranked by co-occurrence, photos hydrated), drawn two ways.
 * Defining it once is what keeps the two from drifting again.
 *
 * The first `limit` are shown: related_people is ranked, so the top few ARE the key voices.
 *
 * Links or buttons, by caller: a PAGE passes `routeFor`, so each voice is a real link (an address a
 * long-press can open or copy); a card/sheet passes nothing and gets buttons, because there a tap
 * layers the person over or replaces in place. Either way the tap is emitted, so the caller decides
 * — a page's handler may let the link navigate, a sheet's prevents it.
 */
import { computed } from "vue"
import { useI18n } from "vue-i18n"
import { RouterLink, type RouteLocationRaw } from "vue-router"
import type { Entity } from "../services/types"
import { personName } from "../utils/personName"
import ProfileAvatar from "./ProfileAvatar.vue"

const props = withDefaults(
  defineProps<{
    people: Entity[]
    limit?: number
    /** Heading level — the topic card nests it under its own h2, the storyline page does not. */
    headingLevel?: 2 | 3
    routeFor?: (id: string) => RouteLocationRaw
  }>(),
  { limit: 8, headingLevel: 3, routeFor: undefined },
)
const emit = defineEmits<{ (e: "open", id: string, event: MouseEvent): void }>()
const { t } = useI18n()

// Names are cased HERE, once, so the aria-label, the avatar's initials and the visible text
// cannot disagree — three bindings read `p.name` below and the pipeline sends many of them
// lowercase (operator 2026-10-01).
const shown = computed(() =>
  props.people.slice(0, props.limit).map((p) => ({ ...p, name: personName(p.name) })),
)
</script>

<template>
  <section v-if="shown.length" data-testid="ec-top-voices">
    <component :is="headingLevel === 2 ? 'h2' : 'h3'" class="lp-section mb-2">
      {{ t("ec.topVoices") }}
    </component>
    <!-- A 4-column grid that fills the row width (operator 2026-09-15): the old flex-wrap left a
         dead gap on the right of each row; the grid spreads the avatars evenly and lets a partial
         last row sit left with empty space below rather than an uneven ragged edge. -->
    <div class="grid grid-cols-4 gap-3">
      <component
        :is="routeFor ? RouterLink : 'button'"
        v-for="p in shown"
        :key="p.id"
        v-bind="routeFor ? { to: routeFor(p.id) } : { type: 'button' }"
        class="flex flex-col items-center gap-1 no-underline"
        :aria-label="p.name"
        data-testid="ec-top-voice"
        @click="(e: MouseEvent) => emit('open', p.id, e)"
      >
        <ProfileAvatar :name="p.name" :src="p.image_url" :size="44" />
        <span class="line-clamp-2 text-center text-xs font-medium text-canvas-foreground">
          {{ p.name }}
        </span>
      </component>
    </div>
  </section>
</template>
