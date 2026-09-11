import { ref } from 'vue'

/**
 * Voice-input opt-in (operator 2026-09-09) — a DEVICE-scoped flag that gates note dictation. Default
 * OFF: the mic never listens until the user turns it on in Settings. Device-local (localStorage),
 * not synced across devices, because microphone use is a per-handset choice.
 *
 * Gating the capability is separate from whether the platform CAN dictate: NoteComposer shows the
 * mic only when this is on AND the browser/WebView exposes SpeechRecognition. (Full native dictation
 * on iOS would additionally need a Capacitor speech plugin — a dependency decision.)
 */
const KEY = 'lp.voiceInputEnabled'

function readInitial(): boolean {
  try {
    return localStorage.getItem(KEY) === '1'
  } catch {
    return false
  }
}

// Module-level so every caller shares one reactive source (Settings toggles it, NoteComposer reads).
const enabled = ref(readInitial())

export function useVoiceInput(): { enabled: typeof enabled; setEnabled: (v: boolean) => void } {
  function setEnabled(v: boolean): void {
    enabled.value = v
    try {
      localStorage.setItem(KEY, v ? '1' : '0')
    } catch {
      /* storage blocked — the in-memory flag still governs this session */
    }
  }
  return { enabled, setEnabled }
}
