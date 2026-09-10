<script setup lang="ts">
/**
 * AvatarCropModal — square crop-on-upload for the account picture.
 *
 * A portrait/landscape photo rarely fills a circle well, so (like every app's avatar picker) we let
 * the user pan + zoom the image inside a fixed square frame before upload. On confirm we render the
 * framed region to an OUT×OUT canvas and emit a PNG Blob; ProfileView uploads that instead of the
 * raw file. Pure canvas + pointer events — no new dependency.
 */
import { computed, onBeforeUnmount, onMounted, ref } from "vue"
import { useI18n } from "vue-i18n"

const props = defineProps<{ file: File }>()
const emit = defineEmits<{ (e: "confirm", blob: Blob): void; (e: "cancel"): void }>()

const { t } = useI18n()

const VIEWPORT = 288 // on-screen square (px); the frame the user composes within
const OUTPUT = 512 // exported square edge (px) — crisp on retina, well under the 2 MB cap

const img = new Image()
const objectUrl = ref<string>("")
const ready = ref(false)
const loadFailed = ref(false)

// base "cover" scale (fill the square), a user zoom multiplier, and the top-left offset of the
// drawn image relative to the viewport (both ≤ 0, kept so the image always covers the frame).
const coverScale = ref(1)
const zoom = ref(1)
const offset = ref({ x: 0, y: 0 })

const drawn = computed(() => {
  const s = coverScale.value * zoom.value
  return { w: img.naturalWidth * s, h: img.naturalHeight * s }
})

function clampOffset(): void {
  offset.value.x = Math.min(0, Math.max(VIEWPORT - drawn.value.w, offset.value.x))
  offset.value.y = Math.min(0, Math.max(VIEWPORT - drawn.value.h, offset.value.y))
}

function centre(): void {
  offset.value = { x: (VIEWPORT - drawn.value.w) / 2, y: (VIEWPORT - drawn.value.h) / 2 }
}

onMounted(() => {
  objectUrl.value = URL.createObjectURL(props.file)
  img.onload = () => {
    coverScale.value = Math.max(VIEWPORT / img.naturalWidth, VIEWPORT / img.naturalHeight)
    zoom.value = 1
    centre()
    ready.value = true
  }
  img.onerror = () => (loadFailed.value = true)
  img.src = objectUrl.value
})

onBeforeUnmount(() => {
  if (objectUrl.value) URL.revokeObjectURL(objectUrl.value)
})

function onZoom(e: Event): void {
  const centreBefore = { x: VIEWPORT / 2 - offset.value.x, y: VIEWPORT / 2 - offset.value.y }
  const ratioX = centreBefore.x / drawn.value.w
  const ratioY = centreBefore.y / drawn.value.h
  zoom.value = Number((e.target as HTMLInputElement).value)
  // keep the frame centre pinned while zooming so it feels anchored, not jumpy
  offset.value = {
    x: VIEWPORT / 2 - ratioX * drawn.value.w,
    y: VIEWPORT / 2 - ratioY * drawn.value.h,
  }
  clampOffset()
}

// pointer drag to reposition
const dragging = ref(false)
let last = { x: 0, y: 0 }
function onPointerDown(e: PointerEvent): void {
  dragging.value = true
  last = { x: e.clientX, y: e.clientY }
  ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
}
function onPointerMove(e: PointerEvent): void {
  if (!dragging.value) return
  offset.value.x += e.clientX - last.x
  offset.value.y += e.clientY - last.y
  last = { x: e.clientX, y: e.clientY }
  clampOffset()
}
function onPointerUp(): void {
  dragging.value = false
}

function confirm(): void {
  const canvas = document.createElement("canvas")
  canvas.width = OUTPUT
  canvas.height = OUTPUT
  const ctx = canvas.getContext("2d")
  if (!ctx) return emit("cancel")
  const r = OUTPUT / VIEWPORT // viewport → output scale
  ctx.drawImage(img, offset.value.x * r, offset.value.y * r, drawn.value.w * r, drawn.value.h * r)
  canvas.toBlob((blob) => (blob ? emit("confirm", blob) : emit("cancel")), "image/png")
}
</script>

<template>
  <div
    class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
    role="dialog"
    aria-modal="true"
    :aria-label="t('profile.avatarCropTitle')"
    data-testid="avatar-crop-modal"
    @click.self="emit('cancel')"
  >
    <div class="w-full max-w-sm rounded-2xl bg-canvas p-5 text-canvas-foreground shadow-xl">
      <h2 class="mb-3 text-lg font-bold">{{ t("profile.avatarCropTitle") }}</h2>

      <p v-if="loadFailed" class="text-sm text-danger" data-testid="avatar-crop-error">
        {{ t("profile.avatarUploadFailed") }}
      </p>

      <template v-else>
        <!-- Square frame with a circular mask overlay so the user sees the final circle. -->
        <div
          class="relative mx-auto touch-none overflow-hidden rounded-lg bg-elevated"
          :style="{ width: `${VIEWPORT}px`, height: `${VIEWPORT}px` }"
          @pointerdown="onPointerDown"
          @pointermove="onPointerMove"
          @pointerup="onPointerUp"
          @pointercancel="onPointerUp"
        >
          <img
            v-if="ready"
            :src="objectUrl"
            alt=""
            class="pointer-events-none absolute select-none"
            :style="{
              left: `${offset.x}px`,
              top: `${offset.y}px`,
              width: `${drawn.w}px`,
              height: `${drawn.h}px`,
              maxWidth: 'none',
            }"
            draggable="false"
          />
          <div
            class="pointer-events-none absolute inset-0 rounded-lg"
            style="box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.45) inset; border-radius: 50%"
            aria-hidden="true"
          />
        </div>

        <label class="mt-4 block text-sm">
          <span class="sr-only">{{ t("profile.avatarCropZoom") }}</span>
          <input
            type="range"
            min="1"
            max="3"
            step="0.01"
            :value="zoom"
            class="w-full"
            :aria-label="t('profile.avatarCropZoom')"
            data-testid="avatar-crop-zoom"
            @input="onZoom"
          />
        </label>
      </template>

      <div class="mt-4 flex justify-end gap-2">
        <button
          type="button"
          class="rounded-lg px-4 py-2 text-sm font-semibold hover:bg-elevated"
          data-testid="avatar-crop-cancel"
          @click="emit('cancel')"
        >
          {{ t("common.cancel") }}
        </button>
        <button
          type="button"
          class="rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-accent-foreground disabled:opacity-50"
          :disabled="!ready"
          data-testid="avatar-crop-confirm"
          @click="confirm"
        >
          {{ t("profile.avatarCropConfirm") }}
        </button>
      </div>
    </div>
  </div>
</template>
