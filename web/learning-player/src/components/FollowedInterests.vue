<script setup lang="ts">
/**
 * Followed interests — the topics, people and storylines you follow (the ＋ on trending topics /
 * storylines / entity cards), grouped by type like the Saved tab's sections. Following these was
 * previously invisible: the tokens went into your interests profile but nothing surfaced them. This
 * makes them visible, navigable and unfollow-able. Complements the followed-shows grid above it.
 *
 * Labels: clusters resolve via the top-cluster set + storylines list; topics/people de-slug from
 * their id (`topic:personal-growth` → "personal growth"), matching ProfileView.
 */
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import { storeToRefs } from 'pinia'
import { useInterestsStore } from '../stores/interests'
import { getStorylines, getTopClusters } from '../services/api'
import type { Storyline } from '../services/types'
import StorylineCard from './StorylineCard.vue'

const { t } = useI18n()
const router = useRouter()
const interests = useInterestsStore()
const { ids } = storeToRefs(interests)

const clusterLabels = ref<Map<string, string>>(new Map())
const storylines = ref<Storyline[]>([])

onMounted(async () => {
  await interests.ensureLoaded().catch(() => {})
  const [clusters, stories] = await Promise.all([
    getTopClusters(60).catch(() => []),
    getStorylines(60).catch(() => []),
  ])
  clusterLabels.value = new Map(clusters.map((c) => [c.id, c.label]))
  storylines.value = stories
})

const storylineById = computed(() => new Map(storylines.value.map((s) => [s.id, s])))
const deslug = (id: string) => id.replace(/^(tc|thc|topic|person):/, '').replace(/[-_]+/g, ' ')
function labelOf(id: string): string {
  return clusterLabels.value.get(id) ?? storylineById.value.get(id)?.label ?? deslug(id)
}

const topics = computed(() => ids.value.filter((i) => i.startsWith('topic:')))
const persons = computed(() => ids.value.filter((i) => i.startsWith('person:')))
// Storylines (thc:) + interest clusters (tc:) — both theme groupings; shown together as "storylines".
const storylineTokens = computed(() =>
  ids.value.filter((i) => i.startsWith('thc:') || i.startsWith('tc:')),
)
const isEmpty = computed(
  () => !topics.value.length && !persons.value.length && !storylineTokens.value.length,
)

function unfollow(id: string): void {
  void interests.toggle(id)
}

const storylineTarget = ref<Storyline | null>(null)
function openStoryline(id: string): void {
  const s = storylineById.value.get(id)
  if (s) storylineTarget.value = s // resolvable → open its sheet; else the chip is display-only
}
function openStorylineTopic(id: string): void {
  storylineTarget.value = null
  void router.push({ name: 'topic', params: { id } })
}
function openStorylinePerson(id: string): void {
  storylineTarget.value = null
  void router.push({ name: 'person', params: { id } })
}
</script>

<template>
  <div data-testid="followed-interests">
    <p v-if="isEmpty" class="text-sm text-muted">{{ t('library.followingEmpty') }}</p>

    <section v-if="topics.length" class="mb-5">
      <h3 class="lp-kicker mb-2">{{ t('library.followingTopics') }}</h3>
      <ul class="flex flex-wrap gap-1.5">
        <li v-for="id in topics" :key="id" class="inline-flex items-center rounded-full bg-overlay">
          <button
            type="button"
            class="max-w-[12rem] truncate py-1 pl-3 pr-1.5 text-sm font-semibold text-topic transition hover:opacity-80"
            @click="router.push({ name: 'topic', params: { id } })"
          >{{ labelOf(id) }}</button>
          <button
            type="button"
            class="rounded-r-full py-1 pl-1 pr-2.5 text-xs text-muted transition hover:text-danger"
            :aria-label="t('library.unfollow', { label: labelOf(id) })"
            data-testid="unfollow"
            @click="unfollow(id)"
          >✕</button>
        </li>
      </ul>
    </section>

    <section v-if="persons.length" class="mb-5">
      <h3 class="lp-kicker mb-2">{{ t('library.followingPeople') }}</h3>
      <ul class="flex flex-wrap gap-1.5">
        <li v-for="id in persons" :key="id" class="inline-flex items-center rounded-full bg-overlay">
          <button
            type="button"
            class="max-w-[12rem] truncate py-1 pl-3 pr-1.5 text-sm font-semibold text-person transition hover:opacity-80"
            @click="router.push({ name: 'person', params: { id } })"
          >{{ labelOf(id) }}</button>
          <button
            type="button"
            class="rounded-r-full py-1 pl-1 pr-2.5 text-xs text-muted transition hover:text-danger"
            :aria-label="t('library.unfollow', { label: labelOf(id) })"
            data-testid="unfollow"
            @click="unfollow(id)"
          >✕</button>
        </li>
      </ul>
    </section>

    <section v-if="storylineTokens.length">
      <h3 class="lp-kicker mb-2">{{ t('library.followingStorylines') }}</h3>
      <ul class="flex flex-wrap gap-1.5">
        <li v-for="id in storylineTokens" :key="id" class="inline-flex items-center rounded-full bg-overlay">
          <button
            type="button"
            class="max-w-[14rem] truncate py-1 pl-3 pr-1.5 text-sm font-semibold text-canvas-foreground transition hover:opacity-80"
            @click="openStoryline(id)"
          >{{ labelOf(id) }}</button>
          <button
            type="button"
            class="rounded-r-full py-1 pl-1 pr-2.5 text-xs text-muted transition hover:text-danger"
            :aria-label="t('library.unfollow', { label: labelOf(id) })"
            data-testid="unfollow"
            @click="unfollow(id)"
          >✕</button>
        </li>
      </ul>
    </section>

    <StorylineCard
      v-if="storylineTarget"
      :id="storylineTarget.id"
      :label="storylineTarget.label"
      :anchor-topic-id="storylineTarget.anchor_topic_id"
      @open-topic="openStorylineTopic"
      @open-person="openStorylinePerson"
      @close="storylineTarget = null"
    />
  </div>
</template>
