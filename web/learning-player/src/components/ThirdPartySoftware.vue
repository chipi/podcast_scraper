<script setup lang="ts">
/**
 * Third-party software (2026-10-05) — the open-source packages the app ships, with their licences.
 *
 * The list is NOT written by hand: `scripts/third-party.mjs` regenerates `third-party.json` from
 * npm's production dependency tree on every build, so it cannot drift from what is actually in the
 * app. Fetched when the page opens rather than bundled, because the licence texts run to a few
 * hundred KB that no other screen needs.
 */
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'

interface Entry {
  name: string
  version: string
  license: string
  url: string
  text: string | null
}

const { t } = useI18n()
const entries = ref<Entry[]>([])
const generated = ref<string | null>(null)
const state = ref<'loading' | 'ready' | 'failed'>('loading')

onMounted(async () => {
  try {
    const res = await fetch(`${import.meta.env.BASE_URL}third-party.json`, { cache: 'no-cache' })
    if (!res.ok) throw new Error(String(res.status))
    const data = (await res.json()) as { generated?: string; packages?: Entry[] }
    entries.value = data.packages ?? []
    generated.value = data.generated ?? null
    state.value = 'ready'
  } catch {
    state.value = 'failed'
  }
})

const builtOn = computed(() =>
  generated.value ? new Date(generated.value).toLocaleDateString(undefined, { dateStyle: 'medium' }) : '',
)
</script>

<template>
  <div class="space-y-4 text-sm leading-relaxed" data-testid="third-party">
    <p>{{ t('about.thirdPartyIntro') }}</p>
    <p v-if="state === 'loading'" class="text-muted">{{ t('about.thirdPartyLoading') }}</p>
    <p v-else-if="state === 'failed'" class="text-muted" data-testid="third-party-failed">
      {{ t('about.thirdPartyFailed') }}
    </p>
    <template v-else>
      <p class="text-muted" data-testid="third-party-count">
        {{ t('about.thirdPartyCount', { count: entries.length, date: builtOn }) }}
      </p>
      <ul class="divide-y divide-border border-y border-border">
        <li v-for="e in entries" :key="`${e.name}@${e.version}`" data-testid="third-party-entry">
          <details class="group py-2">
            <summary class="flex cursor-pointer list-none items-baseline gap-2 marker:content-none [&::-webkit-details-marker]:hidden">
              <span class="min-w-0 flex-1 truncate font-semibold">{{ e.name }}</span>
              <span v-if="e.version" class="shrink-0 font-mono text-xs text-muted">{{ e.version }}</span>
              <span class="shrink-0 font-mono text-xs text-muted">{{ e.license }}</span>
            </summary>
            <div class="mt-2 space-y-2">
              <a :href="e.url" target="_blank" rel="noopener" class="break-all text-xs">{{ e.url }}</a>
              <pre
                v-if="e.text"
                class="max-h-64 overflow-auto whitespace-pre-wrap rounded-lg bg-overlay p-3 font-mono text-[11px] leading-snug text-muted"
              >{{ e.text }}</pre>
              <p v-else class="text-xs text-muted">{{ t('about.thirdPartyNoText', { license: e.license }) }}</p>
            </div>
          </details>
        </li>
      </ul>
    </template>
  </div>
</template>
