<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { FeedApiEntry } from '../../api/feedsApi'
import {
  buildFeedEntry,
  FEED_ADVANCED_FIELDS,
  splitFeedEntry,
  type EpisodeOrder,
  type FeedAdvancedFieldDef,
  type FeedMustFields,
} from '../../utils/feedOverrides'
import CollapsibleSection from '../shared/CollapsibleSection.vue'

/**
 * Per-feed override editor (#694) — the drill-in shown when an operator
 * "Configure"s a feed row. The five selection-controlling fields are structured
 * inputs; rarer tuning is a grouped Advanced block (also structured inputs), and
 * any keys the editor doesn't model round-trip through a raw-JSON escape hatch.
 * Save semantics live in `utils/feedOverrides.ts` (empty = inherit; collapse to
 * a bare URL when clean; preserve unknown keys).
 */
const props = withDefaults(
  defineProps<{
    entry: FeedApiEntry
    globalMaxEpisodes?: number | null
    busy?: boolean
  }>(),
  { globalMaxEpisodes: null, busy: false },
)

const emit = defineEmits<{ save: [FeedApiEntry]; back: [] }>()

const urlKey = ref<'url' | 'rss'>('url')
const url = ref('')
const maxEpisodesStr = ref('')
const episodeOrder = ref<'' | EpisodeOrder>('')
const offsetStr = ref('')
const since = ref('')
const until = ref('')
/** Advanced field values keyed by field key; all held as strings (bool = ''/'true'/'false'). */
const advancedValues = ref<Record<string, string>>({})
const unknownText = ref('')
const error = ref<string | null>(null)

const advancedGroups = computed(() => {
  const groups: { group: string; fields: FeedAdvancedFieldDef[] }[] = []
  for (const f of FEED_ADVANCED_FIELDS) {
    let g = groups.find((x) => x.group === f.group)
    if (!g) {
      g = { group: f.group, fields: [] }
      groups.push(g)
    }
    g.fields.push(f)
  }
  return groups
})

const advancedActiveCount = computed(
  () => Object.values(advancedValues.value).filter((v) => v !== '').length,
)

watch(
  () => props.entry,
  (entry) => {
    const s = splitFeedEntry(entry)
    urlKey.value = s.urlKey
    url.value = s.url
    maxEpisodesStr.value = s.must.max_episodes != null ? String(s.must.max_episodes) : ''
    episodeOrder.value = s.must.episode_order ?? ''
    offsetStr.value = s.must.episode_offset != null ? String(s.must.episode_offset) : ''
    since.value = s.must.episode_since ?? ''
    until.value = s.must.episode_until ?? ''
    const av: Record<string, string> = {}
    for (const f of FEED_ADVANCED_FIELDS) {
      const raw = s.advanced[f.key]
      if (raw === undefined) {
        av[f.key] = ''
      } else if (f.type === 'bool') {
        av[f.key] = raw ? 'true' : 'false'
      } else {
        av[f.key] = String(raw)
      }
    }
    advancedValues.value = av
    unknownText.value = Object.keys(s.extras).length ? JSON.stringify(s.extras, null, 2) : ''
    error.value = null
  },
  { immediate: true },
)

const maxEpisodesOverridesDefault = computed(
  () => String(maxEpisodesStr.value ?? '').trim() !== '',
)

function parseIntField(raw: string, min: number, label: string): number | null | undefined {
  const t = String(raw ?? '').trim()
  if (t === '') return undefined
  const n = Number(t)
  if (!Number.isInteger(n) || n < min) {
    error.value = `${label} must be an integer ≥ ${min}.`
    return null
  }
  return n
}

/** Coerce one advanced field's string into a typed value; null = invalid. */
function coerceAdvanced(field: FeedAdvancedFieldDef, raw: string): unknown | null | undefined {
  const t = String(raw ?? '').trim()
  if (t === '') return undefined
  if (field.type === 'bool') return t === 'true'
  if (field.type === 'string') return t
  const n = Number(t)
  if (!Number.isFinite(n) || (field.type === 'int' && !Number.isInteger(n)) || n < 0) {
    error.value = `${field.label} must be a ${field.type === 'int' ? 'non-negative integer' : 'non-negative number'}.`
    return null
  }
  return n
}

function onSave(): void {
  error.value = null

  const maxEp = parseIntField(maxEpisodesStr.value, 1, 'Max episodes')
  if (maxEp === null) return
  const offset = parseIntField(offsetStr.value, 0, 'Episode offset')
  if (offset === null) return

  const advanced: Record<string, unknown> = {}
  for (const f of FEED_ADVANCED_FIELDS) {
    const v = coerceAdvanced(f, advancedValues.value[f.key] ?? '')
    if (v === null) return
    if (v !== undefined) advanced[f.key] = v
  }

  let unknown: Record<string, unknown> = {}
  const rawUnknown = unknownText.value.trim()
  if (rawUnknown !== '' && rawUnknown !== '{}') {
    let parsed: unknown
    try {
      parsed = JSON.parse(rawUnknown)
    } catch {
      error.value = 'Other fields must be valid JSON.'
      return
    }
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
      error.value = 'Other fields must be a JSON object.'
      return
    }
    unknown = parsed as Record<string, unknown>
  }

  const must: FeedMustFields = {}
  if (maxEp != null) must.max_episodes = maxEp
  if (episodeOrder.value) must.episode_order = episodeOrder.value
  if (offset != null) must.episode_offset = offset
  if (since.value.trim()) must.episode_since = since.value.trim()
  if (until.value.trim()) must.episode_until = until.value.trim()

  // Advanced wins over any same-named unknown key (shouldn't overlap, but be safe).
  emit('save', buildFeedEntry(urlKey.value, url.value, must, { ...unknown, ...advanced }))
}
</script>

<template>
  <div
    class="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto overscroll-contain"
    data-testid="feed-override-editor"
  >
    <div class="flex shrink-0 items-center gap-2">
      <button
        type="button"
        class="rounded border border-border px-2 py-0.5 text-[10px] hover:bg-overlay"
        data-testid="feed-override-back"
        @click="emit('back')"
      >
        ← Feeds
      </button>
      <span
        class="min-w-0 flex-1 truncate font-mono text-[11px] text-surface-foreground"
        data-testid="feed-override-url"
        :title="url"
      >{{ url }}</span>
    </div>

    <p class="shrink-0 text-[10px] text-muted leading-snug">
      Per-feed overrides win over the <strong class="text-surface-foreground">viewer_operator.yaml</strong>
      defaults. Leave a field blank to inherit the global value.
    </p>

    <div class="grid shrink-0 grid-cols-2 gap-2">
      <label class="flex flex-col gap-0.5 text-[10px] text-muted">
        <span>Max episodes</span>
        <input
          v-model="maxEpisodesStr"
          type="number"
          min="1"
          inputmode="numeric"
          class="rounded border border-border bg-elevated px-2 py-1 text-[11px] text-elevated-foreground"
          data-testid="feed-override-max-episodes"
          placeholder="inherit"
        >
        <span
          v-if="maxEpisodesOverridesDefault"
          class="text-[9px] text-warning"
          data-testid="feed-override-maxep-hint"
        >{{
          globalMaxEpisodes != null
            ? `overriding viewer_operator default = ${globalMaxEpisodes}`
            : 'overrides viewer_operator default'
        }}</span>
      </label>

      <label class="flex flex-col gap-0.5 text-[10px] text-muted">
        <span>Episode order</span>
        <select
          v-model="episodeOrder"
          class="rounded border border-border bg-elevated px-2 py-1 text-[11px] text-elevated-foreground"
          data-testid="feed-override-order"
        >
          <option value="">
            Inherit
          </option>
          <option value="newest">
            Newest first
          </option>
          <option value="oldest">
            Oldest first
          </option>
        </select>
      </label>

      <label class="flex flex-col gap-0.5 text-[10px] text-muted">
        <span>Episode offset</span>
        <input
          v-model="offsetStr"
          type="number"
          min="0"
          inputmode="numeric"
          class="rounded border border-border bg-elevated px-2 py-1 text-[11px] text-elevated-foreground"
          data-testid="feed-override-offset"
          placeholder="inherit"
        >
      </label>

      <div class="flex flex-col gap-0.5">
        <label class="flex flex-col gap-0.5 text-[10px] text-muted">
          <span>Episode since</span>
          <input
            v-model="since"
            type="date"
            class="rounded border border-border bg-elevated px-2 py-1 text-[11px] text-elevated-foreground"
            data-testid="feed-override-since"
          >
        </label>
        <label class="flex flex-col gap-0.5 text-[10px] text-muted">
          <span>Episode until</span>
          <input
            v-model="until"
            type="date"
            class="rounded border border-border bg-elevated px-2 py-1 text-[11px] text-elevated-foreground"
            data-testid="feed-override-until"
          >
        </label>
      </div>
    </div>

    <CollapsibleSection
      :title="`Advanced${advancedActiveCount ? ` (${advancedActiveCount} set)` : ''}`"
      summary="retry / delay / circuit-breaker / user_agent"
      :default-open="false"
      class="shrink-0"
    >
      <div class="space-y-2">
        <fieldset
          v-for="grp in advancedGroups"
          :key="grp.group"
          class="rounded border border-border/60 p-1.5"
        >
          <legend class="px-1 text-[9px] font-medium uppercase tracking-wide text-muted">
            {{ grp.group }}
          </legend>
          <div class="grid grid-cols-2 gap-1.5">
            <label
              v-for="f in grp.fields"
              :key="f.key"
              class="flex flex-col gap-0.5 text-[10px] text-muted"
            >
              <span>{{ f.label }}</span>
              <select
                v-if="f.type === 'bool'"
                v-model="advancedValues[f.key]"
                class="rounded border border-border bg-elevated px-1.5 py-0.5 text-[11px] text-elevated-foreground"
                :data-testid="`feed-override-adv-${f.key}`"
              >
                <option value="">
                  Inherit
                </option>
                <option value="true">
                  On
                </option>
                <option value="false">
                  Off
                </option>
              </select>
              <input
                v-else
                v-model="advancedValues[f.key]"
                :type="f.type === 'string' ? 'text' : 'number'"
                :inputmode="f.type === 'int' ? 'numeric' : f.type === 'float' ? 'decimal' : undefined"
                class="rounded border border-border bg-elevated px-1.5 py-0.5 text-[11px] text-elevated-foreground"
                :data-testid="`feed-override-adv-${f.key}`"
                placeholder="inherit"
              >
            </label>
          </div>
        </fieldset>

        <label class="flex flex-col gap-0.5 text-[10px] text-muted">
          <span>Other fields (raw JSON)</span>
          <textarea
            v-model="unknownText"
            spellcheck="false"
            rows="3"
            class="w-full resize-y rounded border border-border bg-elevated p-2 font-mono text-[11px] text-elevated-foreground"
            data-testid="feed-override-extras"
            placeholder="{}"
          />
        </label>
      </div>
    </CollapsibleSection>

    <p
      v-if="error"
      class="shrink-0 rounded border border-danger/40 bg-danger/10 px-2 py-1 text-[10px] text-danger"
      data-testid="feed-override-error"
    >
      {{ error }}
    </p>

    <div class="shrink-0 border-t border-border pt-2">
      <button
        type="button"
        class="rounded border border-border px-2 py-1 text-[10px] hover:bg-overlay disabled:opacity-40"
        :disabled="busy"
        data-testid="feed-override-save"
        @click="onSave"
      >
        Save feed overrides
      </button>
    </div>
  </div>
</template>
