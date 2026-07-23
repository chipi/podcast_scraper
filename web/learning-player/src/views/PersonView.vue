<script setup lang="ts">
/**
 * Standalone Person page (#1261-6) — full-page equivalent of the modal
 * ``EntityCard`` for a person id (`person:...`). Enables direct deep-links
 * from external referrers, subject-jumps from the browse index, and shared
 * URLs. See ``TopicView`` for the parallel rationale.
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
  <section
    class="mx-auto max-w-3xl px-4 pb-8 pt-4"
    data-testid="person-view"
    :aria-label="t('browse.personPage')"
  >
    <EntityCardBody kind="person" :id="props.id" variant="inline" @close="onClose" />
  </section>
</template>
