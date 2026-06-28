<script setup lang="ts">
/**
 * Highlights review (P2 Capture, PRD-040) — the user's captured moments / spans / saved insights,
 * grouped by episode, each with jump-to-moment, inline notes, and delete. A Markdown export link
 * sits in the header (the single export format, REMEMBER-half-scope §4). Embedded in the Library
 * "Highlights" tab. Auth-gated (the store no-ops + stays empty when signed out).
 */
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import { getEpisode, highlightsExportUrl } from '../services/api'
import type { Highlight } from '../services/types'
import { useCaptureStore } from '../stores/capture'
import { formatTime } from '../player/transcriptSync'

const { t } = useI18n()
const capture = useCaptureStore()

// Episode titles for the group headings (slug → title), hydrated lazily; slug is the fallback.
const titles = ref<Record<string, string>>({})

interface Group {
  slug: string
  title: string
  highlights: Highlight[]
}

const groups = computed<Group[]>(() => {
  const bySlug = new Map<string, Highlight[]>()
  for (const h of capture.highlights) {
    const list = bySlug.get(h.episode_slug) ?? []
    list.push(h)
    bySlug.set(h.episode_slug, list)
  }
  return [...bySlug.entries()].map(([slug, highlights]) => ({
    slug,
    title: titles.value[slug] ?? slug,
    highlights,
  }))
})

function jumpQuery(h: Highlight): Record<string, string> {
  return h.start_ms != null ? { t: String(Math.floor(h.start_ms / 1000)) } : {}
}

function label(h: Highlight): string {
  if (h.kind === 'moment') return t('highlights.moment')
  return h.quote_text ?? t('highlights.span')
}

// --- notes (inline add / edit) ---
const editing = ref<string | null>(null) // note id being edited, or `new:<highlightId>`
const draft = ref('')

function startAdd(highlightId: string): void {
  editing.value = `new:${highlightId}`
  draft.value = ''
}
function startEdit(noteId: string, text: string): void {
  editing.value = noteId
  draft.value = text
}
function cancel(): void {
  editing.value = null
  draft.value = ''
}
async function save(): Promise<void> {
  const text = draft.value.trim()
  const key = editing.value
  if (!key || !text) {
    cancel()
    return
  }
  if (key.startsWith('new:')) {
    await capture.addNote('highlight', key.slice(4), text)
  } else {
    await capture.editNote(key, text)
  }
  cancel()
}

onMounted(async () => {
  await capture.ensureLoaded()
  const slugs = [...new Set(capture.highlights.map((h) => h.episode_slug))]
  await Promise.all(
    slugs.map(async (slug) => {
      const d = await getEpisode(slug).catch(() => null)
      if (d) titles.value[slug] = d.title
    }),
  )
})
</script>

<template>
  <div>
    <div class="mb-4 flex items-center justify-between gap-3">
      <p class="text-sm text-muted">{{ t('highlights.count', capture.count, { named: { count: capture.count } }) }}</p>
      <a
        v-if="capture.count"
        :href="highlightsExportUrl()"
        download="my-highlights.md"
        class="rounded-full border border-border px-3 py-1 text-sm font-bold text-accent no-underline transition hover:bg-overlay"
      >{{ t('highlights.export') }}</a>
    </div>

    <p v-if="!capture.count" class="text-muted">{{ t('highlights.empty') }}</p>

    <section v-for="g in groups" :key="g.slug" class="mb-6">
      <RouterLink
        :to="{ name: 'player', params: { slug: g.slug } }"
        class="lp-section mb-2 block no-underline hover:text-accent"
      >{{ g.title }}</RouterLink>
      <ul class="flex flex-col gap-3">
        <li
          v-for="h in g.highlights"
          :key="h.id"
          class="rounded-xl border border-border p-3"
        >
          <div class="flex items-start justify-between gap-2">
            <div class="min-w-0">
              <span
                v-if="h.kind !== 'moment'"
                class="lp-kicker"
              >{{ h.kind === 'insight' ? t('highlights.insight') : t('highlights.span') }}</span>
              <p class="text-sm font-semibold leading-snug">{{ label(h) }}</p>
              <p v-if="h.speaker" class="lp-speaker mt-0.5 text-xs">{{ h.speaker }}</p>
              <span
                v-if="h.anchor_status === 'drifted'"
                class="mt-1 inline-block rounded-full bg-overlay px-2 py-0.5 text-xs text-danger"
                :title="t('highlights.driftedHint')"
              >⚠ {{ t('highlights.drifted') }}</span>
            </div>
            <div class="flex shrink-0 items-center gap-2">
              <RouterLink
                v-if="h.start_ms != null"
                :to="{ name: 'player', params: { slug: h.episode_slug }, query: jumpQuery(h) }"
                class="font-mono text-xs text-accent no-underline"
              >▶ {{ formatTime(h.start_ms / 1000) }}</RouterLink>
              <button
                type="button"
                class="rounded-full p-1 text-muted transition hover:text-danger"
                :aria-label="t('highlights.remove')"
                :title="t('highlights.remove')"
                @click="capture.remove(h.id)"
              >✕</button>
            </div>
          </div>

          <!-- Notes attached to this highlight -->
          <ul v-if="capture.notesFor('highlight', h.id).length" class="mt-2 flex flex-col gap-1">
            <li
              v-for="n in capture.notesFor('highlight', h.id)"
              :key="n.id"
              class="border-l-2 border-border pl-2 text-sm text-muted"
            >
              <div v-if="editing === n.id">
                <textarea
                  v-model="draft"
                  rows="2"
                  class="w-full rounded border border-border bg-canvas px-2 py-1 text-sm"
                  :aria-label="t('highlights.noteLabel')"
                />
                <div class="mt-1 flex gap-2">
                  <button type="button" class="text-xs font-bold text-accent" @click="save">{{ t('highlights.saveNote') }}</button>
                  <button type="button" class="text-xs text-muted" @click="cancel">{{ t('highlights.cancel') }}</button>
                </div>
              </div>
              <div v-else class="flex items-start justify-between gap-2">
                <span class="min-w-0 flex-1 whitespace-pre-line">{{ n.text }}</span>
                <span class="flex shrink-0 gap-1">
                  <button type="button" class="text-xs text-accent" @click="startEdit(n.id, n.text)">{{ t('highlights.editNote') }}</button>
                  <button type="button" class="text-xs text-muted hover:text-danger" :aria-label="t('highlights.removeNote')" @click="capture.removeNote(n.id)">✕</button>
                </span>
              </div>
            </li>
          </ul>

          <!-- Add a new note -->
          <div v-if="editing === `new:${h.id}`" class="mt-2">
            <textarea
              v-model="draft"
              rows="2"
              class="w-full rounded border border-border bg-canvas px-2 py-1 text-sm"
              :aria-label="t('highlights.noteLabel')"
              :placeholder="t('highlights.notePlaceholder')"
            />
            <div class="mt-1 flex gap-2">
              <button type="button" class="text-xs font-bold text-accent" @click="save">{{ t('highlights.saveNote') }}</button>
              <button type="button" class="text-xs text-muted" @click="cancel">{{ t('highlights.cancel') }}</button>
            </div>
          </div>
          <button
            v-else
            type="button"
            class="mt-2 text-xs font-bold text-accent"
            @click="startAdd(h.id)"
          >+ {{ t('highlights.addNote') }}</button>
        </li>
      </ul>
    </section>
  </div>
</template>
