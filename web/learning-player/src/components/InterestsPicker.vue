<script setup lang="ts">
/**
 * Interests picker (PRD-043 FR4 / 3.5) — a dismissible modal to choose interest *clusters*
 * (the top corpus themes by prevalence). Saved per-user; the discovery feed re-ranks Home by
 * these when the personalization flag is on. Pure overlay: backdrop / ESC / ✕ dismiss, focus
 * trap, restore focus on close.
 */
import { computed, onMounted, ref } from "vue"
import CloseIcon from "./CloseIcon.vue"
import { useI18n } from "vue-i18n"
import { getStorylines, getTopClusters, getUserInterests, putUserInterests } from "../services/api"
import type { InterestCluster, Storyline } from "../services/types"
import { useModalSheet } from "../composables/useModalSheet"
import { dedupeByLabel, interestKind, interestLabel } from "../utils/interests"

const emit = defineEmits<{ (e: "close"): void; (e: "saved", ids: string[]): void }>()
const { t } = useI18n()

const clusters = ref<InterestCluster[]>([])
const storylines = ref<Storyline[]>([])
// The interests the user had when the picker opened — carried through save so follows the picker
// doesn't offer (topic:/person: from entity cards, or clusters not shown) survive the PUT replace.
const initialInterests = ref<string[]>([])
const selected = ref<Set<string>>(new Set())
const loading = ref(true)
const saving = ref(false)

const hasClusters = computed(() => clusters.value.length > 0)
const hasStorylines = computed(() => storylines.value.length > 0)
const isEmpty = computed(() => !hasClusters.value && !hasStorylines.value)

function toggle(id: string): void {
  const next = new Set(selected.value)
  if (next.has(id)) next.delete(id)
  else next.add(id)
  selected.value = next
}

/**
 * Follows the picker does NOT offer, shown so they can be seen and removed (operator 2026-09-18).
 *
 * The picker lists the top interest clusters and the storylines — but a user also follows people
 * and topics straight from entity cards, and clusters outside the top set. Those were saved,
 * rendered on the profile, and INVISIBLE here: the screen that edits interests showed five while
 * the profile showed twenty-five, and the difference was unexplained and unremovable.
 *
 * They were never at risk (`save` preserves un-offered ids, below) — they were just unreachable.
 */
const alsoFollowing = computed(() => {
  const offered = new Set<string>([
    ...clusters.value.map((c) => c.id),
    ...storylines.value.map((st) => st.id),
  ])
  const known = new Map<string, string>([
    ...clusters.value.map((c) => [c.id, c.label] as const),
    ...storylines.value.map((st) => [st.id, st.label] as const),
  ])
  return dedupeByLabel(
    initialInterests.value.filter((id) => !offered.has(id) && selected.value.has(id)),
    known,
  ).map((id) => ({ id, kind: interestKind(id), label: interestLabel(id, known) }))
})

async function save(): Promise<void> {
  saving.value = true
  try {
    // Everything the picker offers this session; the selected subset replaces the offered part.
    const offered = new Set<string>([
      ...clusters.value.map((c) => c.id),
      ...storylines.value.map((s) => s.id),
    ])
    const preserved = initialInterests.value.filter((id) => !offered.has(id))
    const chosen = [...offered].filter((id) => selected.value.has(id))
    const stored = await putUserInterests([...preserved, ...chosen])
    emit("saved", stored)
    emit("close")
  } catch {
    saving.value = false // keep the modal open so the user can retry
  }
}

// Modal a11y — the shared sheet plumbing (focus trap + ESC / backdrop dismiss). No history entry.
const dialogEl = ref<HTMLElement | null>(null)
useModalSheet(dialogEl, () => emit("close"))

onMounted(async () => {
  const [tops, tales, current] = await Promise.all([
    getTopClusters(12).catch(() => [] as InterestCluster[]),
    getStorylines(12).catch(() => [] as Storyline[]),
    getUserInterests().catch(() => [] as string[]),
  ])
  clusters.value = tops
  storylines.value = tales
  initialInterests.value = current
  selected.value = new Set(current)
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
      @click.self="emit('close')"
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
            :aria-label="t('interests.close')"
            @click="emit('close')"
          >
            <CloseIcon />
          </button>
        </header>

        <div class="min-h-0 flex-1 overflow-y-auto px-4 py-4">
          <p v-if="loading" class="text-sm text-muted">{{ t("interests.loading") }}</p>
          <p v-else-if="isEmpty" class="text-sm text-muted">{{ t("interests.empty") }}</p>
          <template v-else>
            <!-- Topics (semantic clusters) -->
            <section v-if="hasClusters" data-testid="interests-topics">
              <h3 class="lp-section mb-2">{{ t("interests.topicsHeading") }}</h3>
              <div class="flex flex-wrap gap-2">
                <button
                  v-for="c in clusters"
                  :key="c.id"
                  type="button"
                  :aria-pressed="selected.has(c.id)"
                  class="rounded-full border px-3 py-1.5 text-sm transition"
                  :class="
                    selected.has(c.id)
                      ? 'border-accent bg-accent text-accent-foreground'
                      : 'border-border bg-overlay text-topic hover:bg-elevated'
                  "
                  @click="toggle(c.id)"
                >
                  {{ c.label }}
                </button>
              </div>
            </section>

            <!-- Storylines (theme clusters — topics discussed together) -->
            <section v-if="hasStorylines" class="mt-5" data-testid="interests-storylines">
              <h3 class="lp-section mb-1">{{ t("interests.storylinesHeading") }}</h3>
              <p class="mb-2 text-xs text-muted">{{ t("interests.storylinesHint") }}</p>
              <div class="flex flex-wrap gap-2">
                <button
                  v-for="s in storylines"
                  :key="s.id"
                  type="button"
                  :aria-pressed="selected.has(s.id)"
                  class="rounded-full border px-3 py-1.5 text-sm transition"
                  :class="
                    selected.has(s.id)
                      ? 'border-accent bg-accent text-accent-foreground'
                      : 'border-theme lp-theme-chip text-surface-foreground'
                  "
                  @click="toggle(s.id)"
                >
                  {{ s.label }}
                </button>
              </div>
            </section>

            <!-- Everything else the user follows. Tapping removes it: this is the only place these
                 can be un-followed, since the sections above never list them. -->
            <section v-if="alsoFollowing.length" class="mt-5" data-testid="interests-also-following">
              <h3 class="lp-section mb-1">{{ t("interests.alsoHeading") }}</h3>
              <p class="mb-2 text-xs text-muted">{{ t("interests.alsoHint") }}</p>
              <div class="flex flex-wrap gap-2">
                <button
                  v-for="f in alsoFollowing"
                  :key="f.id"
                  type="button"
                  :aria-pressed="true"
                  :aria-label="t('interests.alsoRemove', { name: f.label })"
                  class="flex items-center gap-1.5 rounded-full border border-accent bg-accent px-3 py-1.5 text-sm text-accent-foreground transition hover:opacity-90"
                  data-testid="interests-also-chip"
                  @click="toggle(f.id)"
                >
                  {{ f.label }}
                  <CloseIcon :size="12" />
                </button>
              </div>
            </section>
          </template>
        </div>

        <footer class="flex items-center justify-end gap-2 border-t border-border px-4 py-3">
          <button
            type="button"
            class="rounded-full px-4 py-2 text-sm font-bold text-muted"
            @click="emit('close')"
          >
            {{ t("interests.cancel") }}
          </button>
          <button
            type="button"
            :disabled="saving || loading"
            class="rounded-full bg-accent px-5 py-2 text-sm font-bold text-accent-foreground disabled:opacity-50"
            @click="save"
          >
            {{ saving ? t("interests.saving") : t("interests.save") }}
          </button>
        </footer>
      </div>
    </div>
  </Teleport>
</template>
