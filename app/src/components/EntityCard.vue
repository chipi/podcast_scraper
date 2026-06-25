<script setup lang="ts">
/**
 * Entity card (PRD-043 FR2/FR3; RFC-102) — a dismissible overlay opened by tapping a
 * person or topic chip. KG-grounded projection from the dedicated `/api/app/persons|topics/{id}`
 * endpoints: appears-in / episodes-about, related people, and (topics) the cluster's sibling
 * themes. The library search lives *inside* the card as one explicit action — tapping a chip
 * opens its card, not a search (the Epic-2 chip→search behaviour now lives behind this button).
 *
 * Re-entrant: tapping a related person/topic inside the card pushes onto a small back stack so
 * the user can walk the graph and step back, all without leaving the player.
 */
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink, useRouter } from 'vue-router'
import { getPersonCard, getTopicCard } from '../services/api'
import type { Entity, EpisodeSummary, PersonCard, Topic, TopicCard } from '../services/types'

type Target = { kind: 'person' | 'topic'; id: string }

const props = defineProps<{ kind: 'person' | 'topic'; id: string }>()
const emit = defineEmits<{ (e: 'close'): void }>()

const { t } = useI18n()
const router = useRouter()

// Back stack — the bottom is the entity the panel was opened on; the top is what's shown.
const stack = ref<Target[]>([{ kind: props.kind, id: props.id }])
const current = computed<Target>(() => stack.value[stack.value.length - 1])

const person = ref<PersonCard | null>(null)
const topic = ref<TopicCard | null>(null)
const loading = ref(false)
const failed = ref(false)

async function load(target: Target): Promise<void> {
  loading.value = true
  failed.value = false
  person.value = null
  topic.value = null
  try {
    if (target.kind === 'person') person.value = await getPersonCard(target.id)
    else topic.value = await getTopicCard(target.id)
  } catch {
    failed.value = true
  } finally {
    loading.value = false
  }
}

// Re-open on a brand-new target (parent opened a different chip) — reset the stack.
watch(
  () => [props.kind, props.id] as const,
  ([kind, id]) => {
    stack.value = [{ kind, id }]
  },
)
watch(current, (target) => void load(target), { immediate: true })

function open(kind: 'person' | 'topic', id: string): void {
  stack.value = [...stack.value, { kind, id }]
}
function back(): void {
  if (stack.value.length > 1) stack.value = stack.value.slice(0, -1)
}

const label = computed(() => person.value?.label ?? topic.value?.label ?? '')
const episodes = computed<EpisodeSummary[]>(
  () => person.value?.episodes ?? topic.value?.episodes ?? [],
)
const relatedPeople = computed<Entity[]>(
  () => person.value?.related_people ?? topic.value?.related_people ?? [],
)
const relatedTopics = computed<Topic[]>(() => person.value?.related_topics ?? [])
const siblings = computed<Topic[]>(() => topic.value?.sibling_topics ?? [])
const episodeCount = computed(() => person.value?.episode_count ?? topic.value?.episode_count ?? 0)
const themeLabel = computed(() => topic.value?.cluster_label ?? null)

const epArt = (e: EpisodeSummary) => e.artwork_url || e.episode_image_url || e.feed_image_url

function searchLibrary(): void {
  const term = label.value.trim()
  emit('close')
  if (term) void router.push({ name: 'search', query: { q: term } })
}

// --- focus management (modal a11y): trap Tab inside the card, restore focus on close ---
const dialogEl = ref<HTMLElement | null>(null)
let restoreFocus: HTMLElement | null = null

function focusables(): HTMLElement[] {
  if (!dialogEl.value) return []
  const sel = 'a[href], button:not([disabled]), input, [tabindex]:not([tabindex="-1"])'
  return Array.from(dialogEl.value.querySelectorAll<HTMLElement>(sel))
}

function onKeydown(e: KeyboardEvent): void {
  if (e.key === 'Escape') {
    emit('close')
    return
  }
  if (e.key !== 'Tab') return
  const items = focusables()
  if (items.length === 0) return
  const first = items[0]
  const last = items[items.length - 1]
  if (e.shiftKey && document.activeElement === first) {
    e.preventDefault()
    last.focus()
  } else if (!e.shiftKey && document.activeElement === last) {
    e.preventDefault()
    first.focus()
  }
}

onMounted(() => {
  restoreFocus = document.activeElement as HTMLElement | null
  window.addEventListener('keydown', onKeydown)
  void nextTick(() => (focusables()[0] ?? dialogEl.value)?.focus())
})
onUnmounted(() => {
  window.removeEventListener('keydown', onKeydown)
  restoreFocus?.focus?.()
})
</script>

<template>
  <div
    class="fixed inset-0 z-40 flex items-end justify-center bg-black/40 sm:items-center"
    role="dialog"
    aria-modal="true"
    :aria-label="label"
    @click.self="emit('close')"
  >
    <div
      ref="dialogEl"
      tabindex="-1"
      class="flex max-h-[85vh] w-full max-w-lg flex-col rounded-t-2xl bg-surface outline-none sm:rounded-2xl"
    >
      <header class="flex items-center gap-2 border-b border-border px-4 py-3">
        <button
          v-if="stack.length > 1"
          type="button"
          class="text-muted"
          :aria-label="t('ec.back')"
          @click="back"
        >‹</button>
        <span class="min-w-0 flex-1">
          <span class="lp-kicker block">{{ current.kind === 'person' ? t('ec.person') : t('ec.topic') }}</span>
          <span class="block truncate font-display text-lg font-bold">{{ label || '…' }}</span>
        </span>
        <button type="button" class="text-muted" :aria-label="t('ec.close')" @click="emit('close')">✕</button>
      </header>

      <div class="min-h-0 flex-1 overflow-y-auto px-4 py-4">
        <p v-if="loading" class="text-sm text-muted">{{ t('ec.loading') }}</p>
        <p v-else-if="failed || (!person && !topic)" class="text-sm text-muted">{{ t('ec.notFound') }}</p>

        <template v-else>
          <!-- Topic theme line + sibling themes -->
          <p v-if="themeLabel" class="mb-3 text-xs text-topic">{{ t('kp.theme', { cluster: themeLabel }) }}</p>

          <button
            type="button"
            class="mb-4 w-full rounded-full bg-accent px-4 py-2 text-sm font-bold text-accent-foreground"
            @click="searchLibrary"
          >
            {{ t('ec.searchLibrary', { term: label }) }}
          </button>

          <!-- Sibling topics (same cluster theme) -->
          <section v-if="siblings.length" class="mb-4">
            <h3 class="lp-kicker mb-2">{{ t('ec.siblingTopics') }}</h3>
            <div class="flex flex-wrap gap-1.5">
              <button
                v-for="s in siblings"
                :key="s.id"
                type="button"
                class="rounded-full bg-overlay px-2.5 py-1 text-xs text-topic transition hover:bg-elevated"
                @click="open('topic', s.id)"
              >{{ s.label }}</button>
            </div>
          </section>

          <!-- Episodes (appears-in / about) -->
          <section v-if="episodes.length" class="mb-4">
            <h3 class="lp-kicker mb-2">
              {{
                current.kind === 'person'
                  ? t('ec.personEpisodes', episodeCount, { named: { count: episodeCount } })
                  : t('ec.topicEpisodes', episodeCount, { named: { count: episodeCount } })
              }}
            </h3>
            <ul class="flex flex-col">
              <li v-for="e in episodes" :key="e.slug">
                <RouterLink
                  :to="{ name: 'player', params: { slug: e.slug } }"
                  class="flex items-center gap-3 border-b border-border py-2 no-underline text-canvas-foreground hover:bg-overlay"
                  @click="emit('close')"
                >
                  <img
                    v-if="epArt(e)"
                    :src="epArt(e)!"
                    alt=""
                    loading="lazy"
                    class="h-10 w-10 shrink-0 rounded-md bg-elevated object-cover"
                  />
                  <div v-else class="h-10 w-10 shrink-0 rounded-md bg-elevated" />
                  <span class="min-w-0 flex-1">
                    <span class="block truncate text-sm font-semibold">{{ e.title }}</span>
                    <span v-if="e.podcast_title" class="lp-kicker block truncate">{{ e.podcast_title }}</span>
                  </span>
                </RouterLink>
              </li>
            </ul>
          </section>

          <!-- Related people -->
          <section v-if="relatedPeople.length" class="mb-4">
            <h3 class="lp-kicker mb-2">{{ t('ec.relatedPeople') }}</h3>
            <div class="flex flex-wrap gap-1.5">
              <button
                v-for="p in relatedPeople"
                :key="p.id"
                type="button"
                class="rounded-full bg-overlay px-2.5 py-1 text-xs text-person transition hover:bg-elevated"
                @click="open('person', p.id)"
              >{{ p.name }}</button>
            </div>
          </section>

          <!-- Related topics (person card only) -->
          <section v-if="relatedTopics.length">
            <h3 class="lp-kicker mb-2">{{ t('ec.relatedTopics') }}</h3>
            <div class="flex flex-wrap gap-1.5">
              <button
                v-for="tp in relatedTopics"
                :key="tp.id"
                type="button"
                class="rounded-full bg-overlay px-2.5 py-1 text-xs text-topic transition hover:bg-elevated"
                @click="open('topic', tp.id)"
              >{{ tp.label }}</button>
            </div>
          </section>
        </template>
      </div>
    </div>
  </div>
</template>
