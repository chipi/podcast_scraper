<script setup lang="ts">
/**
 * The small kind word at the start of a pill — TOPIC, THEME, STORYLINE, PERSON — for rows that mix
 * kinds (operator 2026-10-04 on the Brief, 2026-10-12 for Discover). Colour alone told them apart,
 * and colour is not a label. A row under a kind heading, or of one kind only, carries none.
 */
import { useI18n } from 'vue-i18n'

defineProps<{ kind: 'topic' | 'person' | 'theme' | 'storyline' }>()
const { t } = useI18n()
</script>

<template>
  <!-- A real space after the word, not only a margin: without it the pill's text, and so what a
       screen reader says, ran together ("Topicalpha"). -->
  <span class="mr-1 font-mono text-[10px] uppercase tracking-wide opacity-80" data-testid="kind-word">{{
    `${t(`notes.kind_${kind}`)} `
  }}</span>
</template>
