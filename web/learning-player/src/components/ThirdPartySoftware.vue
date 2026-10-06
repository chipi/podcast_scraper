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
  /** An Apache-2.0 NOTICE file, which has to travel with the software. */
  notice?: string | null
  /** Where it ships: the web layer every platform runs, or the iOS / Android binary only. */
  platform?: 'web' | 'ios' | 'android'
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

/**
 * Grouped by NAME PREFIX (operator 2026-10-05): every `@scope/…` npm package sits under its scope
 * (`@babel`, `@capacitor`, `@vue` …) and every Android library under its Maven group
 * (`androidx.core`, `com.google.firebase` …), each expanding to its members; anything else is its
 * own row. Alphabetical throughout, so a package is where its name says it is.
 */
function prefixOf(name: string): string | null {
  if (name.startsWith('@')) return name.split('/')[0]
  if (name.includes(':')) return name.split(':')[0]
  return null
}
function shortName(name: string, prefix: string): string {
  return name.slice(prefix.length + 1)
}
interface Group {
  key: string
  scope: string | null
  members: Entry[]
}
const groups = computed<Group[]>(() => {
  const byScope = new Map<string, Entry[]>()
  const out: Group[] = []
  for (const e of entries.value) {
    const scope = prefixOf(e.name)
    if (!scope) {
      out.push({ key: e.name + '@' + e.version, scope: null, members: [e] })
      continue
    }
    if (!byScope.has(scope)) {
      const members: Entry[] = []
      byScope.set(scope, members)
      out.push({ key: scope, scope, members })
    }
    byScope.get(scope)!.push(e)
  }
  return out.sort((a, b) =>
    (a.scope ?? a.members[0].name).localeCompare(b.scope ?? b.members[0].name, 'en', { sensitivity: 'base' }),
  )
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
        <li v-for="g in groups" :key="g.key">
          <!-- A scope: one row naming it and how many packages it holds, expanding to its members. -->
          <details v-if="g.scope" class="group/scope py-2" data-testid="third-party-group">
            <summary class="flex cursor-pointer list-none items-baseline gap-2 marker:content-none [&::-webkit-details-marker]:hidden">
              <span class="w-3 shrink-0 text-muted transition group-open/scope:rotate-90" aria-hidden="true">›</span>
              <span class="min-w-0 flex-1 truncate font-semibold">{{ g.scope }}</span>
              <span v-if="g.members[0].platform && g.members[0].platform !== 'web'" class="shrink-0 rounded-full bg-overlay px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wide text-muted" data-testid="third-party-platform">{{ t(`about.platform_${g.members[0].platform}`) }}</span>
              <span class="shrink-0 text-xs text-muted">{{ t('about.thirdPartyGroupCount', g.members.length) }}</span>
            </summary>
            <ul class="mt-1 border-l border-border pl-3">
              <li v-for="e in g.members" :key="`${e.name}@${e.version}`" data-testid="third-party-entry">
                <details class="group/entry py-1.5">
                  <summary class="flex cursor-pointer list-none items-baseline gap-2 marker:content-none [&::-webkit-details-marker]:hidden">
                    <span class="w-3 shrink-0 text-muted transition group-open/entry:rotate-90" aria-hidden="true">›</span>
                    <span class="min-w-0 flex-1 truncate">{{ shortName(e.name, g.scope) }}</span>
                    <span v-if="e.version" class="shrink-0 font-mono text-xs text-muted">{{ e.version }}</span>
                    <span class="shrink-0 font-mono text-xs text-muted">{{ e.license }}</span>
                  </summary>
                  <div class="ml-5 mt-2 space-y-1.5">
                    <a :href="e.url" target="_blank" rel="noopener" class="break-all text-xs">{{ e.url }}</a>
                    <p class="font-mono text-[10px] uppercase tracking-wide text-muted">{{ t('about.thirdPartyLicenceText') }}</p>
                    <pre
                      v-if="e.text"
                      class="max-h-64 overflow-auto whitespace-pre-wrap rounded-lg bg-overlay p-3 font-mono text-[11px] leading-snug text-muted"
                    >{{ e.text }}</pre>
                    <p v-else class="text-xs text-muted">{{ t('about.thirdPartyNoText', { license: e.license }) }}</p>
                    <template v-if="e.notice">
                      <p class="font-mono text-[10px] uppercase tracking-wide text-muted">{{ t('about.thirdPartyNotice') }}</p>
                      <pre class="max-h-64 overflow-auto whitespace-pre-wrap rounded-lg bg-overlay p-3 font-mono text-[11px] leading-snug text-muted">{{ e.notice }}</pre>
                    </template>
                  </div>
                </details>
              </li>
            </ul>
          </details>
          <!-- An unscoped package: a row of its own, expanding to its link and licence text. -->
          <details v-else class="group/entry py-2" data-testid="third-party-entry">
            <summary class="flex cursor-pointer list-none items-baseline gap-2 marker:content-none [&::-webkit-details-marker]:hidden">
              <span class="w-3 shrink-0 text-muted transition group-open/entry:rotate-90" aria-hidden="true">›</span>
              <span class="min-w-0 flex-1 truncate font-semibold">{{ g.members[0].name }}</span>
              <span v-if="g.members[0].platform && g.members[0].platform !== 'web'" class="shrink-0 rounded-full bg-overlay px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wide text-muted" data-testid="third-party-platform">{{ t(`about.platform_${g.members[0].platform}`) }}</span>
              <span v-if="g.members[0].version" class="shrink-0 font-mono text-xs text-muted">{{ g.members[0].version }}</span>
              <span class="shrink-0 font-mono text-xs text-muted">{{ g.members[0].license }}</span>
            </summary>
            <div class="ml-5 mt-2 space-y-1.5">
              <a :href="g.members[0].url" target="_blank" rel="noopener" class="break-all text-xs">{{ g.members[0].url }}</a>
              <p class="font-mono text-[10px] uppercase tracking-wide text-muted">{{ t('about.thirdPartyLicenceText') }}</p>
              <pre
                v-if="g.members[0].text"
                class="max-h-64 overflow-auto whitespace-pre-wrap rounded-lg bg-overlay p-3 font-mono text-[11px] leading-snug text-muted"
              >{{ g.members[0].text }}</pre>
              <p v-else class="text-xs text-muted">{{ t('about.thirdPartyNoText', { license: g.members[0].license }) }}</p>
              <template v-if="g.members[0].notice">
                <p class="font-mono text-[10px] uppercase tracking-wide text-muted">{{ t('about.thirdPartyNotice') }}</p>
                <pre class="max-h-64 overflow-auto whitespace-pre-wrap rounded-lg bg-overlay p-3 font-mono text-[11px] leading-snug text-muted">{{ g.members[0].notice }}</pre>
              </template>
            </div>
          </details>
        </li>
      </ul>
    </template>
  </div>
</template>
