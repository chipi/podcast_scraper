<script setup lang="ts">
/**
 * Interests picker (PRD-043 FR4 / 3.5) — the onboarding sheet for choosing what shapes Home.
 *
 * The same four sections as the Profile Interests tab (`InterestSections`), so a listener meets one
 * way of choosing interests rather than two (beta feedback 2026-10-04). It used to offer only the
 * top 12 themes and storylines, with everything else the user followed in an "Also following" row
 * that could only be removed from.
 *
 * Unlike Profile, it keeps a LOCAL selection and writes once on Save: this is a step in a funnel,
 * and Cancel has to mean nothing changed. Pure overlay: backdrop / ESC / ✕ dismiss, focus trap,
 * restore focus on close.
 */
import { onMounted, ref } from "vue"
import { toCountBucket, track } from "../services/analytics"
import CloseIcon from "./CloseIcon.vue"
import InterestSections from "./InterestSections.vue"
import { useI18n } from "vue-i18n"
import { getUserInterests, putUserInterests } from "../services/api"
import { useModalSheet } from "../composables/useModalSheet"
import { useInterestsStore } from "../stores/interests"

/**
 * Where this picker was opened from (#2267).
 *
 * Required rather than defaulted: the three values answer different questions — `first_run` and
 * `home_prompt` measure whether the onboarding ask works, `profile` measures deliberate curation
 * later on. A default would silently file one as another, and the funnel step between
 * `auth_completed` and `interests_saved` is exactly where the spec expects to see drop-off.
 */
const props = defineProps<{ trigger: "first_run" | "profile" | "home_prompt" }>()
const emit = defineEmits<{ (e: "close"): void; (e: "saved", ids: string[]): void }>()
const { t } = useI18n()
// The picker OWNS the write, so it owns telling the store. Leaving that to each parent is what
// broke Home: `ProfileView` updated a local ref and `HomeView` set its dismissed flag, so the
// authoritative set was never written back here — see `replaceAll`.
const interests = useInterestsStore()

/** Ordered, so Save sends follows in the order they were made — the store's own order. */
const selected = ref<string[]>([])
/**
 * Whether a save completed, so closing afterwards is not ALSO reported as a dismissal (#2267).
 *
 * `save()` emits "saved" and then "close", so a close handler alone would record every successful
 * save as a dismissal too — and `interests_dismissed` is the funnel's drop-off signal, so it would
 * read as though everyone who chose interests had also abandoned the step.
 */
let didSave = false
const loading = ref(true)
const saving = ref(false)
/**
 * The current interests could not be read. Save is a whole-set REPLACE, so saving an unloaded
 * selection would wipe every follow the user has — the sheet says so and offers nothing to save.
 */
const loadFailed = ref(false)

function toggle(id: string): void {
  selected.value = selected.value.includes(id)
    ? selected.value.filter((t) => t !== id)
    : [...selected.value, id]
}

async function save(): Promise<void> {
  saving.value = true
  try {
    // Every kind is listed now, so the selection IS the whole set — nothing outside it to preserve.
    const stored = await putUserInterests(selected.value)
    // BEFORE the emit: every surface reading the store must be correct by the time a parent's
    // `saved` handler runs (HomeView's re-pulls discovery).
    // Bucketed, never the exact count and never the chosen ids: the spec's no-free-text rule, and
    // "how many" is all any metric reads.
    track("interests_saved", { count: toCountBucket(stored.length) })
    didSave = true
    interests.replaceAll(stored)
    emit("saved", stored)
    emit("close")
  } catch {
    saving.value = false // keep the modal open so the user can retry
  }
}

// Modal a11y — the shared sheet plumbing (focus trap + ESC / backdrop dismiss). No history entry.
const dialogEl = ref<HTMLElement | null>(null)
useModalSheet(dialogEl, closeSheet)

/** Close, reporting a dismissal unless a save already happened. */
function closeSheet(): void {
  if (!didSave) track("interests_dismissed")
  emit("close")
}

onMounted(async () => {
  // Before the fetch, like landing_view: a listener whose interests never load still opened the
  // picker, and the funnel must not under-count the people whose network failed them.
  track("interests_picker_shown", { trigger: props.trigger })
  try {
    selected.value = await getUserInterests()
  } catch {
    loadFailed.value = true
  }
  loading.value = false
})
</script>

<template>
  <Teleport to="body">
    <div
      class="lp-sheet-scrim"
      role="dialog"
      aria-modal="true"
      :aria-label="t('interests.title')"
      @click.self="closeSheet()"
    >
      <div
        ref="dialogEl"
        tabindex="-1"
        class="lp-sheet w-full max-w-lg rounded-t-2xl bg-surface outline-none sm:rounded-2xl"
      >
        <header class="flex items-center justify-between gap-2 border-b border-border px-4 py-3">
          <span class="min-w-0">
            <span class="block font-display text-lg font-bold">{{ t("interests.title") }}</span>
            <span class="lp-kicker block">{{ t("interests.subtitle") }}</span>
          </span>
          <button
            type="button"
            class="lp-nav shrink-0"
            data-testid="interests-close"
            :aria-label="t('interests.close')"
            @click="closeSheet()"
          >
            <CloseIcon />
          </button>
        </header>

        <div class="min-h-0 flex-1 overflow-y-auto px-4 py-4">
          <p v-if="loading" class="text-sm text-muted">{{ t("interests.loading") }}</p>
          <p v-else-if="loadFailed" class="text-sm text-muted" data-testid="interests-load-failed">
            {{ t("profile.unavailable") }}
          </p>
          <InterestSections v-else :selected="selected" @toggle="toggle" />
        </div>

        <footer class="flex items-center justify-end gap-2 border-t border-border px-4 py-3">
          <button
            type="button"
            class="rounded-full px-4 py-2 text-sm font-bold text-muted"
            data-testid="interests-cancel"
            @click="closeSheet()"
          >
            {{ t("interests.cancel") }}
          </button>
          <button
            type="button"
            :disabled="saving || loading || loadFailed"
            class="rounded-full bg-accent px-5 py-2 text-sm font-bold text-accent-foreground disabled:opacity-50"
            data-testid="interests-save"
            @click="save"
          >
            {{ saving ? t("interests.saving") : t("interests.save") }}
          </button>
        </footer>
      </div>
    </div>
  </Teleport>
</template>
