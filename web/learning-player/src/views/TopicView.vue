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
  <!-- The one page width (`lp-page`) and the app shell's gutter, nothing more (operator 2026-10-05).
       The card is `flush` here: on this route it IS the page, so its content starts at the same left
       edge as every other page. It once padded itself `px-4` inside a centred 768px column, and a
       `px-4` on this wrapper too had made it 32px a side (2026-09-30). -->
  <section
    class="lp-page pb-8 pt-4"
    data-testid="topic-view"
    :aria-label="t('browse.topicPage')"
  >
    <EntityCardBody kind="topic" :id="props.id" variant="inline" root-control="close" flush @close="onClose" />
  </section>
</template>
