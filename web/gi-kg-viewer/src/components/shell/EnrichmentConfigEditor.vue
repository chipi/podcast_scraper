<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import {
  getEnrichmentConfig,
  getEnrichmentConfigSchema,
  getEnrichmentProviderTypes,
  putEnrichmentConfig,
  type EnrichmentConfigResponse,
  type JsonSchemaFragment,
  type ProviderTypeInfo,
} from '../../api/enrichmentConfigApi'

/**
 * RFC-088 v2 enrichment-config editor — embedded in EnrichmentPanel.vue
 * under a "Configuration" section. Two-tier toggle (global on/off +
 * per-enricher on/off), per-enricher knob fields generated from the
 * server-fetched schema, and a per-row provider dropdown for ML
 * enrichers populated from /api/enrichment/provider-types.
 *
 * Edits land in ``viewer_operator.yaml`` via PUT /api/enrichment/config.
 * Profile defaults shine through on every key the operator hasn't
 * explicitly overridden.
 */

interface Props {
  corpusPath: string
}
const props = defineProps<Props>()

interface KnobSchemaProperty {
  type?: string
  enum?: string[]
  minimum?: number
  maximum?: number
  default?: unknown
  description?: string
}
type EnricherSchemaBlock = {
  properties?: Record<string, KnobSchemaProperty | Record<string, unknown>>
}
type SchemaEnrichers = Record<string, EnricherSchemaBlock>

const loading = ref(false)
const saving = ref(false)
const dirty = ref(false)
const error = ref<string | null>(null)
const saveNotice = ref<string | null>(null)

const remote = ref<EnrichmentConfigResponse | null>(null)
const schema = ref<JsonSchemaFragment | null>(null)
const providerTypes = ref<Record<string, ProviderTypeInfo[]>>({})

/** Local working copy that the form binds to. Mirrors the operator block. */
interface EnricherDraft {
  enabled: boolean
  knobs: Record<string, unknown>
  provider: { type: string; params: Record<string, unknown> } | null
}
const draft = ref<{ enabled: boolean; enrichers: Record<string, EnricherDraft> }>({
  enabled: true,
  enrichers: {},
})

const schemaEnrichers = computed<SchemaEnrichers>(() => {
  const props_ = (schema.value?.properties as Record<string, unknown>) ?? {}
  const enr = (props_['enrichers'] as Record<string, unknown>) ?? {}
  return ((enr['properties'] as Record<string, unknown>) ?? {}) as SchemaEnrichers
})

/** Sorted list of enricher ids — order is stable for the operator. */
const enricherIds = computed<string[]>(() => Object.keys(schemaEnrichers.value).sort())

function isPlainObject(x: unknown): x is Record<string, unknown> {
  return typeof x === 'object' && x !== null && !Array.isArray(x)
}

function loadDraftFromOperator(op: Record<string, unknown>): void {
  const enabled = typeof op.enabled === 'boolean' ? op.enabled : true
  const rawEnrichers = isPlainObject(op.enrichers) ? op.enrichers : {}
  const next: Record<string, EnricherDraft> = {}
  for (const id of enricherIds.value) {
    const blk = rawEnrichers[id]
    if (!isPlainObject(blk)) {
      next[id] = { enabled: id in rawEnrichers, knobs: {}, provider: null }
      continue
    }
    const enabledKey = blk['enabled']
    const knobs: Record<string, unknown> = {}
    let provider: EnricherDraft['provider'] = null
    for (const [k, v] of Object.entries(blk)) {
      if (k === 'enabled') continue
      if (k === 'provider' && isPlainObject(v)) {
        const ptype = typeof v['type'] === 'string' ? (v['type'] as string) : ''
        const params: Record<string, unknown> = {}
        for (const [pk, pv] of Object.entries(v)) {
          if (pk === 'type') continue
          params[pk] = pv
        }
        provider = { type: ptype, params }
        continue
      }
      knobs[k] = v
    }
    next[id] = {
      enabled: typeof enabledKey === 'boolean' ? enabledKey : true,
      knobs,
      provider,
    }
  }
  draft.value = { enabled, enrichers: next }
}

function serializeDraft(): Record<string, unknown> {
  const out: Record<string, unknown> = { enabled: draft.value.enabled, enrichers: {} }
  const enrichers = out.enrichers as Record<string, unknown>
  for (const [id, ed] of Object.entries(draft.value.enrichers)) {
    if (!ed.enabled && Object.keys(ed.knobs).length === 0 && !ed.provider) {
      // Not enabled and no edits — omit entirely (profile defaults apply
      // if this id is in the profile's set).
      continue
    }
    const blk: Record<string, unknown> = {}
    if (!ed.enabled) blk.enabled = false
    for (const [k, v] of Object.entries(ed.knobs)) blk[k] = v
    if (ed.provider && ed.provider.type) {
      blk.provider = { type: ed.provider.type, ...ed.provider.params }
    }
    enrichers[id] = blk
  }
  return out
}

async function refresh(): Promise<void> {
  if (!props.corpusPath?.trim()) {
    error.value = 'No corpus path set.'
    return
  }
  loading.value = true
  error.value = null
  try {
    const [cfg, sch, pts] = await Promise.all([
      getEnrichmentConfig(props.corpusPath),
      getEnrichmentConfigSchema(),
      getEnrichmentProviderTypes(),
    ])
    remote.value = cfg
    schema.value = sch
    providerTypes.value = pts.by_protocol
    loadDraftFromOperator(cfg.operator_block)
    dirty.value = false
  } catch (exc) {
    error.value = exc instanceof Error ? exc.message : String(exc)
  } finally {
    loading.value = false
  }
}

async function save(): Promise<void> {
  if (!props.corpusPath?.trim()) {
    error.value = 'No corpus path set.'
    return
  }
  saving.value = true
  error.value = null
  saveNotice.value = null
  try {
    const block = serializeDraft()
    const fresh = await putEnrichmentConfig(props.corpusPath, block)
    remote.value = fresh
    loadDraftFromOperator(fresh.operator_block)
    dirty.value = false
    saveNotice.value = 'Saved.'
  } catch (exc) {
    error.value = exc instanceof Error ? exc.message : String(exc)
  } finally {
    saving.value = false
  }
}

function discard(): void {
  if (remote.value) {
    loadDraftFromOperator(remote.value.operator_block)
    dirty.value = false
    saveNotice.value = null
  }
}

function resetToProfileDefaults(): void {
  // Clear the operator-side block entirely. Profile shines through.
  draft.value = { enabled: true, enrichers: {} }
  for (const id of enricherIds.value) {
    draft.value.enrichers[id] = { enabled: true, knobs: {}, provider: null }
  }
  dirty.value = true
}

function manifestPropFor(enricherId: string, knob: string): KnobSchemaProperty | null {
  const props_ = schemaEnrichers.value[enricherId]?.properties ?? {}
  const v = props_[knob]
  return v && typeof v === 'object' ? (v as KnobSchemaProperty) : null
}

function providerTypesFor(enricherId: string): ProviderTypeInfo[] {
  const block = schemaEnrichers.value[enricherId]
  if (!block || !isPlainObject(block.properties)) return []
  const provider = (block.properties as Record<string, unknown>)['provider']
  if (!isPlainObject(provider)) return []
  const oneOf = provider['oneOf']
  if (!Array.isArray(oneOf)) return []
  // Each oneOf entry has properties.type.const = type name. Look up in providerTypes.
  const names = new Set<string>()
  for (const alt of oneOf) {
    if (!isPlainObject(alt)) continue
    const props_ = alt['properties']
    if (!isPlainObject(props_)) continue
    const typeProp = props_['type']
    if (!isPlainObject(typeProp)) continue
    const constVal = typeProp['const']
    if (typeof constVal === 'string') names.add(constVal)
  }
  return Object.values(providerTypes.value)
    .flat()
    .filter((pt) => names.has(pt.name))
}

function selectedProviderType(enricherId: string): ProviderTypeInfo | null {
  const sel = draft.value.enrichers[enricherId]?.provider?.type
  if (!sel) return null
  for (const list of Object.values(providerTypes.value)) {
    for (const pt of list) if (pt.name === sel) return pt
  }
  return null
}

function providerParamSchema(enricherId: string): Record<string, KnobSchemaProperty> {
  const pt = selectedProviderType(enricherId)
  if (!pt) return {}
  const props_ = (pt.params_schema['properties'] as Record<string, KnobSchemaProperty>) ?? {}
  return props_
}

function markDirty(): void {
  dirty.value = true
  saveNotice.value = null
}

watch(() => props.corpusPath, () => void refresh())
onMounted(refresh)
</script>

<template>
  <section data-testid="enrichment-config-editor" class="flex flex-col gap-3 text-[11px]">
    <header class="flex items-center justify-between gap-2">
      <div>
        <h3 class="text-sm font-semibold">Enrichment configuration</h3>
        <p class="text-muted text-[10px]">
          Edit ``viewer_operator.yaml``'s enrichment block. Profile defaults shine through
          unless overridden. Saved settings persist across runs.
        </p>
      </div>
      <div class="flex items-center gap-2">
        <button
          type="button"
          class="rounded border border-default bg-overlay px-2 py-1 hover:bg-overlay-2 disabled:opacity-50"
          data-testid="enrichment-config-refresh-btn"
          :disabled="loading"
          @click="void refresh()"
        >
          {{ loading ? 'Loading…' : 'Reload' }}
        </button>
        <button
          type="button"
          class="rounded border border-default bg-overlay px-2 py-1 hover:bg-overlay-2 disabled:opacity-50"
          data-testid="enrichment-config-reset-btn"
          :disabled="loading || saving"
          @click="resetToProfileDefaults()"
        >
          Reset to profile
        </button>
        <button
          type="button"
          class="rounded border border-default bg-overlay px-2 py-1 hover:bg-overlay-2 disabled:opacity-50"
          data-testid="enrichment-config-discard-btn"
          :disabled="!dirty || saving"
          @click="discard()"
        >
          Discard
        </button>
        <button
          type="button"
          class="rounded border border-emerald-700 bg-emerald-700/30 px-2 py-1 hover:bg-emerald-700/40 disabled:opacity-50"
          data-testid="enrichment-config-save-btn"
          :disabled="!dirty || saving"
          @click="void save()"
        >
          {{ saving ? 'Saving…' : 'Save' }}
        </button>
      </div>
    </header>

    <div v-if="error" class="rounded border border-rose-700 bg-rose-900/30 p-2 text-rose-200">
      {{ error }}
    </div>
    <div
      v-if="saveNotice"
      class="rounded border border-emerald-700 bg-emerald-900/30 p-2 text-emerald-200"
      data-testid="enrichment-config-save-notice"
    >
      {{ saveNotice }}
    </div>

    <!-- Tier 1 — global enable -->
    <label
      class="flex items-center gap-2 rounded border border-default bg-overlay p-2"
      data-testid="enrichment-config-global-enabled"
    >
      <input
        type="checkbox"
        :checked="draft.enabled"
        @change="draft.enabled = (($event.target as HTMLInputElement).checked); markDirty()"
      />
      <span class="font-semibold">Enrichment enabled</span>
      <span class="text-muted">(master switch; unchecking disables the whole pass)</span>
    </label>

    <!-- Tier 2 — per-enricher rows -->
    <div v-if="enricherIds.length === 0" class="text-muted">
      No enrichers known. Reload the page or check the server.
    </div>
    <ul v-else class="flex flex-col gap-2" data-testid="enrichment-config-enricher-list">
      <li
        v-for="id in enricherIds"
        :key="id"
        class="rounded border border-default bg-overlay p-2"
        :data-testid="`enrichment-config-row-${id}`"
      >
        <label class="flex items-center gap-2">
          <input
            type="checkbox"
            :checked="draft.enrichers[id]?.enabled ?? true"
            :data-testid="`enrichment-config-enabled-${id}`"
            @change="
              draft.enrichers[id] = {
                ...(draft.enrichers[id] ?? { enabled: true, knobs: {}, provider: null }),
                enabled: ($event.target as HTMLInputElement).checked,
              };
              markDirty()
            "
          />
          <span class="font-mono">{{ id }}</span>
        </label>

        <!-- Knob fields, generated from manifest.config_schema -->
        <div
          v-if="Object.keys(schemaEnrichers[id]?.properties ?? {}).length > 0"
          class="mt-2 grid grid-cols-2 gap-2 pl-6"
        >
          <template
            v-for="(_propVal, knob) in schemaEnrichers[id]?.properties ?? {}"
            :key="knob"
          >
            <div
              v-if="!['enabled', 'opt_in', 'max_cost_usd_per_run', 'expected_duration_s', 'provider'].includes(knob)"
              class="flex flex-col gap-1"
            >
              <label class="text-muted text-[10px]" :for="`knob-${id}-${knob}`">
                {{ knob }}
                <span
                  v-if="manifestPropFor(id, knob)?.description"
                  class="block text-[9px] opacity-70"
                >{{ manifestPropFor(id, knob)?.description }}</span>
              </label>
              <input
                :id="`knob-${id}-${knob}`"
                :data-testid="`enrichment-config-knob-${id}-${knob}`"
                :type="manifestPropFor(id, knob)?.type === 'integer' || manifestPropFor(id, knob)?.type === 'number' ? 'number' : 'text'"
                :step="manifestPropFor(id, knob)?.type === 'integer' ? 1 : 'any'"
                :min="manifestPropFor(id, knob)?.minimum"
                :max="manifestPropFor(id, knob)?.maximum"
                :value="String(draft.enrichers[id]?.knobs?.[knob] ?? '')"
                :placeholder="
                  manifestPropFor(id, knob)?.default !== undefined
                    ? `default: ${String(manifestPropFor(id, knob)?.default)}`
                    : ''
                "
                class="rounded border border-default bg-base px-2 py-1"
                @input="
                  ;(draft.enrichers[id] = {
                    ...(draft.enrichers[id] ?? { enabled: true, knobs: {}, provider: null }),
                    knobs: {
                      ...(draft.enrichers[id]?.knobs ?? {}),
                      [knob]:
                        ($event.target as HTMLInputElement).value === ''
                          ? undefined
                          : (manifestPropFor(id, knob)?.type === 'integer' ||
                              manifestPropFor(id, knob)?.type === 'number'
                                ? Number(($event.target as HTMLInputElement).value)
                                : ($event.target as HTMLInputElement).value),
                    },
                  });
                  markDirty()
                "
              />
            </div>
          </template>
        </div>

        <!-- Provider section — only when this enricher has provider_requirement -->
        <div
          v-if="providerTypesFor(id).length > 0"
          class="mt-2 rounded border border-default bg-base/40 p-2"
          :data-testid="`enrichment-config-provider-${id}`"
        >
          <p class="mb-1 text-[10px] text-muted">Provider</p>
          <select
            :data-testid="`enrichment-config-provider-type-${id}`"
            :value="draft.enrichers[id]?.provider?.type ?? ''"
            class="w-full rounded border border-default bg-base px-2 py-1"
            @change="
              draft.enrichers[id] = {
                ...(draft.enrichers[id] ?? { enabled: true, knobs: {}, provider: null }),
                provider: ($event.target as HTMLSelectElement).value
                  ? { type: ($event.target as HTMLSelectElement).value, params: {} }
                  : null,
              };
              markDirty()
            "
          >
            <option value="">— inherit / none —</option>
            <option v-for="pt in providerTypesFor(id)" :key="pt.name" :value="pt.name">
              {{ pt.name }} — {{ pt.description }}
            </option>
          </select>

          <div
            v-if="draft.enrichers[id]?.provider"
            class="mt-1 grid grid-cols-2 gap-2 pl-2"
          >
            <template
              v-for="(propSchema, paramKey) in providerParamSchema(id)"
              :key="paramKey"
            >
              <div class="flex flex-col gap-1">
                <label class="text-muted text-[10px]">{{ paramKey }}</label>
                <input
                  :data-testid="`enrichment-config-provider-param-${id}-${paramKey}`"
                  :type="propSchema?.type === 'integer' || propSchema?.type === 'number' ? 'number' : 'text'"
                  :value="String(draft.enrichers[id]?.provider?.params?.[paramKey] ?? '')"
                  :placeholder="
                    propSchema?.default !== undefined
                      ? `default: ${String(propSchema?.default)}`
                      : ''
                  "
                  class="rounded border border-default bg-base px-2 py-1"
                  @input="
                    ;(draft.enrichers[id] = {
                      ...(draft.enrichers[id] ?? { enabled: true, knobs: {}, provider: null }),
                      provider: {
                        type: draft.enrichers[id]?.provider?.type ?? '',
                        params: {
                          ...(draft.enrichers[id]?.provider?.params ?? {}),
                          [paramKey]:
                            ($event.target as HTMLInputElement).value === ''
                              ? undefined
                              : (propSchema?.type === 'integer' || propSchema?.type === 'number'
                                  ? Number(($event.target as HTMLInputElement).value)
                                  : ($event.target as HTMLInputElement).value),
                        },
                      },
                    });
                    markDirty()
                  "
                />
              </div>
            </template>
          </div>
        </div>
      </li>
    </ul>
  </section>
</template>
