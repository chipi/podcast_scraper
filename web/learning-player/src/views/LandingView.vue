<script setup lang="ts">
/**
 * Logged-out lure landing (RFC-120). Slim "marketing" shape — NOT a mirror of HomeView:
 * value-prop hero + "Create your free account" CTA, one read-only "Featured" rail (curated
 * teaser), topic chips, a short "how it works", repeat CTA. Every card/chip funnels to signup
 * (no action controls, no auth-gated detail). The only anonymous content it touches is the
 * teaser allow-list (/discover, /corpus/trending-topics) — server-clamped for anon callers.
 *
 * The guard redirects logged-out visitors here with `?redirect=<intended>`; the CTAs thread that
 * through login → OAuth so a shared deep link survives signup.
 */
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink, useRoute } from 'vue-router'
import { getDiscover, getTrendingTopics } from '../services/api'
import type { EpisodeSummary } from '../services/types'
import { safeInternalPath } from '../utils/redirect'

const { t } = useI18n()
const route = useRoute()

const featured = ref<EpisodeSummary[]>([])
const topics = ref<{ topic_id: string; topic_label?: string | null }[]>([])

/** Same-origin redirect target carried by the guard, if any. */
const redirect = computed<string | undefined>(() => safeInternalPath(route.query.redirect) ?? undefined)

/** Route to signup, preserving a deep-link redirect (a featured card sends you to that episode). */
function signupTo(deepLink?: string) {
  const redir = deepLink ?? redirect.value
  return { name: 'login', query: { mode: 'signup', ...(redir ? { redirect: redir } : {}) } }
}
const signInTo = computed(() => ({
  name: 'login',
  query: redirect.value ? { redirect: redirect.value } : {},
}))

function artwork(ep: EpisodeSummary): string | null {
  return ep.artwork_url || ep.episode_image_url || ep.feed_image_url
}

onMounted(async () => {
  // Degrade gracefully — the hero + CTA stand alone if the teaser can't load.
  try {
    // /discover is recency-ordered; dedupe by show and keep the 4 shows with the newest
    // episodes (one card each), so the rail reads as "the latest across 4 different shows".
    const page = await getDiscover(8)
    const seen = new Set<string>()
    const distinct: EpisodeSummary[] = []
    for (const ep of page.items) {
      if (seen.has(ep.feed_id)) continue
      seen.add(ep.feed_id)
      distinct.push(ep)
      if (distinct.length === 4) break
    }
    featured.value = distinct
  } catch {
    /* teaser unavailable — hero + CTA still render */
  }
  try {
    const res = await getTrendingTopics()
    topics.value = res.topics.slice(0, 8)
  } catch {
    /* chips optional */
  }
})
</script>

<template>
  <div class="mx-auto max-w-5xl px-1">
    <!-- Hero -->
    <section class="pb-8 pt-6 text-center sm:pt-10">
      <span class="lp-kicker">{{ t('app.tagline') }}</span>
      <h1 class="mx-auto mb-3 mt-2 max-w-2xl font-display text-4xl font-extrabold tracking-tight sm:text-5xl">
        {{ t('landing.heroTitle') }}
      </h1>
      <p class="mx-auto mb-6 max-w-xl text-base text-muted sm:text-lg">
        {{ t('landing.heroSub') }}
      </p>
      <div class="flex flex-wrap items-center justify-center gap-3">
        <RouterLink
          :to="signupTo()"
          class="rounded-full bg-accent px-7 py-3 font-bold text-accent-foreground no-underline"
          data-testid="landing-cta-primary"
        >
          {{ t('landing.ctaCreate') }}
        </RouterLink>
        <RouterLink
          :to="signInTo"
          class="rounded-full border border-border px-6 py-3 font-bold no-underline hover:bg-surface"
          data-testid="landing-cta-signin"
        >
          {{ t('auth.signIn') }}
        </RouterLink>
      </div>
    </section>

    <!-- Featured teaser rail (read-only; a card funnels to signup for that episode) -->
    <section v-if="featured.length" class="py-6" data-testid="landing-featured">
      <h2 class="mb-3 px-1 font-display text-xl font-bold">{{ t('landing.featured') }}</h2>
      <div class="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <RouterLink
          v-for="ep in featured"
          :key="ep.slug"
          :to="signupTo(`/episode/${ep.slug}`)"
          class="group block rounded-2xl border border-border bg-surface p-3 no-underline"
          data-testid="landing-card"
        >
          <div class="mb-2 aspect-square w-full overflow-hidden rounded-xl bg-canvas">
            <img
              v-if="artwork(ep)"
              :src="artwork(ep) || ''"
              :alt="ep.podcast_title || ep.title"
              class="h-full w-full object-cover transition group-hover:scale-[1.03]"
              loading="lazy"
            />
          </div>
          <p class="line-clamp-2 text-sm font-bold leading-snug">{{ ep.title }}</p>
          <p v-if="ep.podcast_title" class="mt-0.5 line-clamp-1 text-xs text-muted">
            {{ ep.podcast_title }}
          </p>
        </RouterLink>
      </div>
    </section>

    <!-- Topic chips -->
    <section v-if="topics.length" class="py-4">
      <h2 class="mb-2 px-1 font-display text-xl font-bold">{{ t('landing.explore') }}</h2>
      <div class="flex flex-wrap gap-2">
        <RouterLink
          v-for="tp in topics"
          :key="tp.topic_id"
          :to="signupTo(`/topic/${tp.topic_id}`)"
          class="rounded-full border border-topic/40 px-3 py-1.5 text-sm font-semibold text-topic no-underline transition hover:bg-overlay"
          data-testid="landing-chip"
        >{{ tp.topic_label || tp.topic_id }}</RouterLink>
      </div>
    </section>

    <!-- How it works -->
    <section class="py-8">
      <h2 class="mb-4 px-1 font-display text-xl font-bold">{{ t('landing.howTitle') }}</h2>
      <div class="grid gap-4 sm:grid-cols-3">
        <div v-for="n in 3" :key="n" class="rounded-2xl border border-border bg-surface p-5">
          <div class="mb-3 flex items-center gap-3">
            <span class="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-accent text-sm font-bold text-accent-foreground">
              {{ n }}
            </span>
            <h3 class="font-display text-lg font-bold">{{ t(`landing.how${n}Title`) }}</h3>
          </div>
          <p class="text-sm leading-relaxed text-muted">{{ t(`landing.how${n}`) }}</p>
        </div>
      </div>
    </section>

    <!-- Repeat CTA -->
    <section class="pb-12 pt-2 text-center">
      <p class="mb-3 font-display text-xl font-bold">{{ t('landing.closer') }}</p>
      <RouterLink
        :to="signupTo()"
        class="inline-block rounded-full bg-accent px-7 py-3 font-bold text-accent-foreground no-underline"
        data-testid="landing-cta-foot"
      >
        {{ t('landing.ctaCreate') }}
      </RouterLink>
    </section>
  </div>
</template>
