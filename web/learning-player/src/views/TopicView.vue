<script setup lang="ts">
/**
 * Standalone Topic page (#1261-6) — full-page equivalent of the modal
 * ``EntityCard`` for a topic id (`topic:...`). Enables direct deep-links
 * from external referrers, subject-jumps from the browse index, and shared
 * URLs — replaces the mobile-hostile Cmd-K palette explicitly ruled out
 * of scope in the parent issue.
 *
 * KG-grounded via ``/api/app/topics/{id}``. Rendering is delegated to
 * ``EntityCardBody`` (the same body used by the modal + KnowledgePanel
 * inline), so the topic card stays visually consistent everywhere.
 */
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import EntityCardBody from '../components/EntityCardBody.vue'

const props = defineProps<{ id: string }>()
const router = useRouter()
const { t } = useI18n()

function onClose(): void {
  if (window.history.length > 1) router.back()
  else void router.push({ name: 'home' })
}
</script>

<template>
  <!-- NO horizontal padding here (operator 2026-09-30). The card inside (EntityCardBody) pads its
       own header and body with `px-4`, so a `px-4` on this wrapper too put 32px of gutter on each
       side — visibly narrower than the storyline page, whose single `px-4` is the app's page gutter.
       The card's padding IS the gutter; this wrapper only centres and caps the width. -->
  <section
    class="mx-auto max-w-3xl pb-8 pt-4"
    data-testid="topic-view"
    :aria-label="t('browse.topicPage')"
  >
    <EntityCardBody kind="topic" :id="props.id" variant="inline" root-control="close" @close="onClose" />
  </section>
</template>
