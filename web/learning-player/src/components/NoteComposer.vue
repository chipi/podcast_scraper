<script setup lang="ts">
/**
 * NoteComposer (NT.1/NT.2/NT.3) — the ONE reusable "add a note" affordance for any note target
 * (episode / highlight / insight, later more). Lists the target's existing notes with their
 * timestamp, lets you add one (auth-gated like every per-user write), and offers voice dictation
 * where the platform supports the built-in Web Speech API.
 *
 * Dictation runs through {@link useDictation}: the Capacitor speech plugin on iOS/Android (the
 * browser Web Speech API is present-but-inert in the iOS WKWebView — it latched the mic on and
 * transcribed nothing), and the built-in `SpeechRecognition` in a real browser. The mic shows only
 * when the Settings opt-in is on AND the platform can dictate.
 */
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useCaptureStore } from '../stores/capture'
import { useSignInGate } from '../composables/useSignInGate'
import { useVoiceInput } from '../composables/useVoiceInput'
import { useDictation } from '../composables/useDictation'
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
  // Stop dictation first: a live recogniser holds a `base` snapshot and its next partial would
  // rewrite `draft` from that base, re-inserting the text we just cleared and saved.
  dictation.stop()
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

// --- Voice dictation (NT.3): native via the Capacitor speech plugin, browser via Web Speech.
// One interface over both lives in useDictation; the caller owns the draft. ---
let base = ''
const dictateError = ref(false)
const dictation = useDictation({
  lang: () => locale.value,
  onStart: () => {
    base = draft.value ? draft.value + ' ' : ''
    dictateError.value = false
  },
  onText: (text) => {
    draft.value = base + text
  },
  onError: () => {
    dictateError.value = true
  },
})
// Mic shows only when the operator has opted in (Settings) AND the platform can actually dictate.
const canDictate = computed(() => voiceEnabled.value && dictation.canDictate)
const dictating = dictation.dictating
// Starting dictation is a per-user action, gated like save; STOPPING never is — if the session
// expires mid-dictation the user must still be able to turn the mic off.
const startDictation = gated(() => dictation.toggle())
function onMicClick(): void {
  if (dictating.value) dictation.stop()
  else startDictation()
}
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

    <!-- Textarea on its OWN full-width row; the mic + Add ride a row BENEATH it. The mic used to sit
         inline beside the field and stole its width, shrinking the note (operator). -->
    <div class="flex flex-col gap-2">
      <textarea
        v-model="draft"
        rows="2"
        :aria-label="t('notes.title')"
        :placeholder="isGated ? t('auth.signInToSave') : t('notes.placeholder')"
        class="w-full resize-y rounded-xl border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent"
        data-testid="note-input"
      />
      <div class="flex items-center justify-end gap-2">
        <button
          v-if="canDictate"
          type="button"
          class="lp-tap flex h-9 w-9 shrink-0 items-center justify-center rounded-full border transition"
          :class="dictating ? 'border-accent text-accent' : 'border-border text-muted hover:text-canvas-foreground'"
          :aria-label="dictating ? t('notes.dictateStop') : t('notes.dictate')"
          :title="dictating ? t('notes.dictateStop') : t('notes.dictate')"
          :aria-pressed="dictating"
          data-testid="note-dictate"
          @click="onMicClick"
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
      <p
        v-if="dictateError"
        class="text-right text-xs font-semibold text-danger"
        role="alert"
        data-testid="note-dictate-error"
      >
        {{ t('notes.dictateError') }}
      </p>
    </div>
  </section>
</template>
