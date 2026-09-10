<script setup lang="ts">
/**
 * NoteComposer (NT.1/NT.2/NT.3) — the ONE reusable "add a note" affordance for any note target
 * (episode / highlight / insight, later more). Lists the target's existing notes with their
 * timestamp, lets you add one (auth-gated like every per-user write), and offers voice dictation
 * where the platform supports the built-in Web Speech API.
 *
 * Dictation deliberately uses `SpeechRecognition` (no new dependency): available in most browsers
 * and Android WebViews, absent on the iOS WKWebView — where the mic simply doesn't render. Full
 * cross-platform native dictation would need a Capacitor speech plugin (a dependency decision).
 */
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useCaptureStore } from '../stores/capture'
import { useSignInGate } from '../composables/useSignInGate'
import { useVoiceInput } from '../composables/useVoiceInput'
import { formatPublishDate } from '../utils/format'
import type { NoteTarget } from '../services/types'

const props = defineProps<{ target: NoteTarget; targetId: string }>()
const { t, locale } = useI18n()
const capture = useCaptureStore()
const { isGated, gated } = useSignInGate()
const { enabled: voiceEnabled } = useVoiceInput()

const notes = computed(() => capture.notesFor(props.target, props.targetId))
const draft = ref('')

// Self-hydrate so the note list works on surfaces that don't already load captures (entity cards,
// the show page). No-op after the first load; caught so a signed-out/offline fetch stays quiet.
onMounted(() => void capture.ensureLoaded().catch(() => {}))

const save = gated(async () => {
  const text = draft.value.trim()
  if (!text) return
  draft.value = ''
  await capture.addNote(props.target, props.targetId, text)
})

/** Deleting a note is a per-user write — gate it like `save` (#1590). Per-item id, so wrap and
 * invoke a zero-arg gated closure rather than passing an arg `gated()` does not accept. */
function removeNote(id: string): void {
  void gated(() => capture.removeNote(id))()
}

function noteDate(unixSeconds: number): string {
  return formatPublishDate(new Date(unixSeconds * 1000).toISOString(), locale.value) ?? ''
}

// --- Voice dictation (NT.3), built-in Web Speech API only, graceful where absent ---
interface SpeechRecognitionLike {
  interimResults: boolean
  lang: string
  onresult: ((e: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null
  onend: (() => void) | null
  start(): void
  stop(): void
}
type SRCtor = new () => SpeechRecognitionLike
const SR: SRCtor | undefined =
  typeof window === 'undefined'
    ? undefined
    : (window as unknown as { SpeechRecognition?: SRCtor; webkitSpeechRecognition?: SRCtor })
        .SpeechRecognition ??
      (window as unknown as { webkitSpeechRecognition?: SRCtor }).webkitSpeechRecognition
// Mic shows only when the operator has opted in (Settings) AND the platform can actually dictate.
const canDictate = computed(() => voiceEnabled.value && !!SR)
const dictating = ref(false)
let recog: SpeechRecognitionLike | null = null
let base = ''

const toggleDictation = gated(() => {
  if (!SR) return
  if (dictating.value) {
    recog?.stop()
    return
  }
  base = draft.value ? draft.value + ' ' : ''
  recog = new SR()
  recog.interimResults = true
  recog.lang = locale.value || 'en'
  recog.onresult = (e) => {
    let transcript = ''
    for (let i = 0; i < e.results.length; i++) transcript += e.results[i][0].transcript
    draft.value = base + transcript
  }
  recog.onend = () => {
    dictating.value = false
  }
  recog.start()
  dictating.value = true
})

onBeforeUnmount(() => recog?.stop())
</script>

<template>
  <section class="mt-4" data-testid="note-composer">
    <h3 class="lp-section mb-2">{{ t('notes.title') }}</h3>

    <ul v-if="notes.length" class="mb-3 flex flex-col gap-2">
      <li
        v-for="n in notes"
        :key="n.id"
        class="rounded-lg border border-border p-3"
        data-testid="note-item"
      >
        <p class="whitespace-pre-wrap text-sm leading-relaxed text-canvas-foreground">{{ n.text }}</p>
        <div class="mt-1.5 flex items-center justify-between gap-2">
          <span class="lp-kicker">{{ noteDate(n.created_at) }}</span>
          <button
            type="button"
            class="text-xs font-semibold text-muted transition hover:text-danger"
            :aria-label="t('notes.remove')"
            data-testid="note-delete"
            @click="removeNote(n.id)"
          >
            {{ t('notes.remove') }}
          </button>
        </div>
      </li>
    </ul>

    <div class="flex items-end gap-2">
      <textarea
        v-model="draft"
        rows="2"
        :placeholder="isGated ? t('auth.signInToSave') : t('notes.placeholder')"
        class="min-w-0 flex-1 resize-y rounded-xl border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent"
        data-testid="note-input"
      />
      <button
        v-if="canDictate"
        type="button"
        class="lp-tap flex h-9 w-9 shrink-0 items-center justify-center rounded-full border transition"
        :class="dictating ? 'border-accent text-accent' : 'border-border text-muted hover:text-canvas-foreground'"
        :aria-label="dictating ? t('notes.dictateStop') : t('notes.dictate')"
        :title="dictating ? t('notes.dictateStop') : t('notes.dictate')"
        :aria-pressed="dictating"
        data-testid="note-dictate"
        @click="toggleDictation"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" class="h-4 w-4" aria-hidden="true"><path d="M12 2a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/><path d="M19 10v1a7 7 0 0 1-14 0v-1M12 18v4"/></svg>
      </button>
      <button
        type="button"
        class="shrink-0 rounded-full bg-accent px-4 py-2 text-sm font-bold text-accent-foreground disabled:opacity-50"
        :disabled="!draft.trim()"
        data-testid="note-save"
        @click="save"
      >
        {{ t('notes.add') }}
      </button>
    </div>
  </section>
</template>
