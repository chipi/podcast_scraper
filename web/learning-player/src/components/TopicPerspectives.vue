<script setup lang="ts">
/**
 * Topic perspectives (#1146) — multi-perspective synthesis on the topic card.
 * Each guest who spoke on the topic, with their grounded insights. Self-fetches
 * from GET /api/app/topics/{id}/perspectives; renders nothing when the topic has
 * none. Speaker names tap through to their person card (the same `open` contract
 * the card's other people rows use).
 */
import { computed, ref, watch } from "vue"
import { useI18n } from "vue-i18n"
import { RouterLink } from "vue-router"

import SectionStatus from "./SectionStatus.vue"
import ProfileAvatar from "./ProfileAvatar.vue"
import { useSectionState } from "../composables/useSectionState"
import { ApiError, getTopicPerspectives } from "../services/api"
import { formatTime } from "../player/transcriptSync"
import type { TopicPerspective } from "../services/types"

const props = defineProps<{ id: string; scope?: "all" | "mine" }>()
const emit = defineEmits<{ (e: "open", payload: { kind: "person" | "topic"; id: string }): void }>()

const { t } = useI18n()

/**
 * `useSectionState` rather than a hand-rolled try/catch, because this is #1591's defect exactly:
 * a fetch failure caught into an empty array, on a section that hides when empty — so an outage,
 * a timeout and "nobody discussed this topic" all rendered identically. #1591 fixed that for the
 * Home sections; the topic card's sections were never migrated, and this is one of them.
 *
 * The trigger is ordinary: `serve` is a single process, so one concurrent search can time this
 * request out, and the card then quietly asserts that no guest had a perspective.
 */
const section = useSectionState<TopicPerspective[]>([])
const perspectives = computed(() => section.data.value)
/** Guards against a slow reply for a topic the reader has already navigated away from. */
const requestSeq = ref(0)

async function load(): Promise<void> {
  const mine = requestSeq.value + 1
  requestSeq.value = mine
  await section.load(async () => {
    // One retry inside the fetcher, so `useSectionState`'s phase contract is untouched. The
    // failure this guards against is a transient timeout, and re-asking costs one request where
    // the alternative is showing an error over content that was actually available.
    for (let attempt = 0; ; attempt += 1) {
      try {
        const r = await getTopicPerspectives(props.id, props.scope)
        if (mine !== requestSeq.value) throw new Error("superseded")
        return r.perspectives
      } catch (err) {
        if (mine !== requestSeq.value) throw err
        /**
         * A 404 here means "this topic HAS no perspectives", not "the request failed" (#2004 item 12).
         *
         * Verified against prod: `GET /api/app/topics/topic:reward-hacking/perspectives` returns
         * `404 {"detail":"No perspectives for this topic."}`. Treating that as an error is what put
         * an unattributed "Couldn't load this right now" on the topic page — the app reporting a
         * fault where the server reported an absence, and offering a Try again that could only ever
         * fail again.
         *
         * The sibling endpoint gets this right: conversation-arc answers `200 {"weeks":[]}` for the
         * same topic and correctly renders nothing. Empty is not broken.
         */
        if (err instanceof ApiError && err.status === 404) return []
        if (attempt >= 1) throw err
        await new Promise((resolve) => setTimeout(resolve, 600))
      }
    }
  })
}

watch(
  () => [props.id, props.scope] as const,
  () => void load(),
  { immediate: true }
)

// Show up to PREVIEW insights per speaker; the rest sit behind a per-speaker toggle.
const PREVIEW = 3
const expanded = ref<Set<string>>(new Set())
function toggle(personId: string): void {
  const next = new Set(expanded.value)
  if (next.has(personId)) next.delete(personId)
  else next.add(personId)
  expanded.value = next
}
</script>

<template>
  <!-- Error says so and offers a retry, via the shared control so every section fails the same
       way. Genuinely-empty still renders nothing — that distinction is the whole point (#1591:
       "hide when the SYSTEM is empty"). No skeleton here: this sits inside an already-loading
       card, so a second shimmer would be noise. -->
  <!--
    The heading stays visible when this FAILS (#2004 item 12).

    The error box used to render alone, with the `<section>` that carries the `<h3>` behind a
    `v-else-if` — so a failure produced an unlabelled "Couldn't load this right now" wedged between
    two unrelated sections. The reader is told something broke without being told what, and the
    retry button retries an unnamed thing. It also made the failure undiagnosable from a screenshot:
    two components render into this exact slot, and nothing distinguished which one had failed.
    Uses the count-free title, because on error the count is precisely what we do not know.
  -->
  <section v-if="section.isError.value" class="mb-4" data-testid="topic-perspectives-error">
    <h3 class="lp-section mb-2">{{ t("ec.perspectivesTitle") }}</h3>
    <SectionStatus :phase="section.phase.value" @retry="load()" />
  </section>

  <section v-else-if="perspectives.length" class="mb-4" data-testid="topic-perspectives">
    <h3 class="lp-section mb-2">
      {{ t("ec.perspectives", perspectives.length, { named: { count: perspectives.length } }) }}
    </h3>
    <ul class="flex flex-col gap-2.5">
      <li
        v-for="p in perspectives"
        :key="p.person_id"
        class="rounded-lg border border-border bg-overlay p-3"
        data-testid="topic-perspective"
      >
        <!-- Avatar in its own left column; everything else (name + count, then the insights and the
             show-more) lives in the right column so it all aligns to where the NAME starts and
             nothing tucks under the avatar. -->
        <div class="flex gap-2.5">
          <ProfileAvatar
            :name="p.person_name"
            :src="p.image_url"
            :size="24"
            class="mt-0.5 shrink-0"
          />
          <div class="min-w-0 flex-1">
            <!-- Name + count share ONE baseline. -->
            <div class="flex flex-wrap items-baseline gap-x-2">
              <button
                type="button"
                class="text-sm font-bold text-person hover:underline"
                @click="emit('open', { kind: 'person', id: p.person_id })"
              >
                {{ p.person_name }}
              </button>
              <span class="lp-kicker">{{
                t("ec.perspectiveInsights", p.insight_count, { named: { count: p.insight_count } })
              }}</span>
            </div>
            <ul class="mt-1 flex flex-col gap-1">
              <li
                v-for="ins in expanded.has(p.person_id) ? p.insights : p.insights.slice(0, PREVIEW)"
                :key="ins.id"
                class="flex items-baseline gap-1.5 text-sm text-canvas-foreground"
              >
                <span aria-hidden="true" class="text-muted">•</span>
                <span class="min-w-0 flex-1">{{ ins.text }}</span>
                <!-- #2032: topic → insight → episode-moment. Grounded insights carry their source
                     episode + the supporting quote's start, so the take jumps into the player AT
                     the moment. Ungrounded/quote-less insights render with no ▶ (stays honest). -->
                <RouterLink
                  v-if="ins.episode_slug && ins.start_ms != null"
                  :to="{
                    name: 'player',
                    params: { slug: ins.episode_slug },
                    query: { t: String(Math.floor(ins.start_ms / 1000)) },
                  }"
                  class="shrink-0 font-mono text-xs font-bold text-accent no-underline"
                  data-testid="perspective-jump"
                  :aria-label="t('kp.jumpToMoment', { time: formatTime(ins.start_ms / 1000) })"
                  :title="t('kp.jumpToMoment', { time: formatTime(ins.start_ms / 1000) })"
                  >▶ {{ formatTime(ins.start_ms / 1000) }}</RouterLink
                >
              </li>
            </ul>
            <button
              v-if="p.insights.length > PREVIEW"
              type="button"
              class="mt-1 text-xs font-semibold text-accent hover:underline"
              @click="toggle(p.person_id)"
            >
              {{
                expanded.has(p.person_id)
                  ? t("ec.perspectiveLess")
                  : t("ec.perspectiveMore", { count: p.insights.length - PREVIEW })
              }}
            </button>
          </div>
        </div>
      </li>
    </ul>
  </section>
</template>
