<script setup lang="ts">
/**
 * An export, OPEN, with its formats in the top-right corner (operator 2026-10-05).
 *
 * Episode notes and Library highlights each had a Markdown chip beside a PDF chip. PDF opened the
 * print-styled page, and its "download" saved HTML — not the PDF the reader asked for — while two
 * chips side by side read as two different documents. Now ONE link opens the document here, and
 * this viewer carries the formats:
 *
 * - **Markdown** — a download link on the web; on native the share sheet with the file, because
 *   WKWebView ignores `<a download>`.
 * - **Print or share** — whatever the platform does with a document: the share sheet on native
 *   (iOS offers Print there, whose preview saves a PDF); on the web the browser's file share where
 *   it has one, else its print dialog, where Save as PDF is a destination. No PDF library: the
 *   operator chose to leave PDF to each platform.
 *
 * ## Why an iframe with `srcdoc`
 *
 * The document is fetched by the CALLER with the app's own credentials (the shell's bearer token on
 * native) and shown from memory — no second request. The external browser was tried and failed:
 * SFSafariViewController does not share the app's cookie jar, so the export arrived unauthenticated
 * and rendered the sign-in gate. The export is a complete standalone document with its own print
 * stylesheet, so it needs its own document: injecting it would break both pages' styling.
 *
 * `sandbox` grants no scripts and no navigation: the document is ours, but it is assembled from
 * episode content. It grants `allow-same-origin` + `allow-modals` so the PARENT can call the
 * frame's `print()`; without `allow-scripts`, same-origin lets nothing inside the frame run.
 *
 * ## Why it teleports to the caller's target
 *
 * The episode-notes panel is `showModal()`'d on mobile, which puts it in the TOP LAYER — above the
 * whole normal layer whatever its z-index. Teleported to `body`, the viewer rendered invisibly
 * behind the panel ("nothing happens when I click PDF", operator 2026-09-27). The caller passes
 * `sheetTeleportTarget()`, resolved when it opens.
 */
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { isNative, saveAndShareText } from '../services/native'

const props = defineProps<{
  /** The fetched print-styled document. */
  html: string
  /** File name for the shared page, `…html`. */
  htmlFilename: string
  /** File name for the Markdown, `…md`. */
  mdFilename: string
  /** The Markdown export's URL — the web's download link. */
  mdUrl: string
  /** Fetches the Markdown — native hands it to the share sheet. */
  fetchMarkdown: () => Promise<string>
  /** The document's name, for the dialog and the shared file's title. */
  title: string
  /** Where to mount: the open dialog when there is one, else `body`. */
  to: HTMLElement | string
}>()
const emit = defineEmits<{ close: []; markdown: []; share: [] }>()
const { t } = useI18n()

const frame = ref<HTMLIFrameElement | null>(null)
const savingMd = ref(false)

async function shareMarkdownNative(): Promise<void> {
  if (savingMd.value) return
  emit('markdown')
  savingMd.value = true
  try {
    await saveAndShareText(props.mdFilename, await props.fetchMarkdown())
  } finally {
    savingMd.value = false
  }
}

async function printOrShare(): Promise<void> {
  emit('share')
  if (isNative()) {
    await saveAndShareText(props.htmlFilename, props.html, 'text/html')
    return
  }
  const file = new File([props.html], props.htmlFilename, { type: 'text/html' })
  if (typeof navigator.canShare === 'function' && navigator.canShare({ files: [file] })) {
    try {
      await navigator.share({ files: [file], title: props.title })
      return
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return // the reader closed the sheet — done
    }
  }
  frame.value?.contentWindow?.print()
}
</script>

<template>
  <Teleport :to="to">
    <div
      class="fixed inset-0 z-[60] flex flex-col bg-canvas pb-[env(safe-area-inset-bottom)]"
      role="dialog"
      aria-modal="true"
      :aria-label="title"
      data-testid="export-viewer"
    >
      <div
        class="flex shrink-0 items-center justify-between gap-2 border-b border-border px-4 pb-2 pt-[max(0.5rem,env(safe-area-inset-top))]"
      >
        <button
          type="button"
          class="rounded-full border border-border px-3 py-1.5 text-sm font-bold text-canvas-foreground transition hover:bg-overlay"
          data-testid="export-viewer-close"
          @click="emit('close')"
        >
          {{ t('export.close') }}
        </button>
        <div class="flex items-center gap-2">
          <button
            v-if="isNative()"
            type="button"
            :disabled="savingMd"
            :aria-label="t('export.markdownLabel')"
            class="rounded-full border border-border px-3 py-1.5 text-sm font-bold text-accent transition hover:bg-overlay disabled:opacity-50"
            data-testid="export-viewer-md"
            @click="shareMarkdownNative"
          >
            {{ t('export.markdown') }}
          </button>
          <a
            v-else
            :href="mdUrl"
            :download="mdFilename"
            :aria-label="t('export.markdownLabel')"
            class="rounded-full border border-border px-3 py-1.5 text-sm font-bold text-accent no-underline transition hover:bg-overlay"
            data-testid="export-viewer-md"
            @click="emit('markdown')"
          >{{ t('export.markdown') }}</a>
          <button
            type="button"
            class="rounded-full border border-border px-3 py-1.5 text-sm font-bold text-accent transition hover:bg-overlay"
            data-testid="export-viewer-share"
            @click="printOrShare"
          >
            {{ t('export.share') }}
          </button>
        </div>
      </div>
      <iframe
        ref="frame"
        :srcdoc="html"
        sandbox="allow-same-origin allow-modals"
        class="min-h-0 w-full flex-1 border-0 bg-white"
        :title="title"
        data-testid="export-viewer-frame"
      />
    </div>
  </Teleport>
</template>
