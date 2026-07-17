<script setup lang="ts">
/**
 * graph-v3 Tier 5A-1 → Tier 7-1 — floating legend for theme-cluster regions.
 *
 * Renders only when the `themeClusterRegions` lens is on AND the
 * `topic_theme_clusters.json` artifact is loaded.
 *
 * Two-level tree (graph-v3 tier 7-1): rows are grouped under the super-theme
 * emitted by the `topic_theme_clusters` enricher v1.1.0
 * (`super_theme_id` / `super_theme_label`). A super-theme with a single child
 * is auto-flattened — no expand chevron, one row that behaves the same as
 * v1.0 output. Multi-child super-themes expand/collapse under a header row.
 *
 * State — per-super-theme expand state + whole-legend collapse state — both
 * sync via USERPREFS-1 (`graphThemeLegendCollapsed`,
 * `graphThemeLegendExpandedSupers`). Backwards-compat: clusters without
 * `super_theme_id` land under a synthetic per-cluster group (renders flat).
 */
import { computed, onMounted, ref, watch } from 'vue'
import type { TopicClustersCluster } from '../../api/corpusTopicClustersApi'
import { useArtifactsStore } from '../../stores/artifacts'
import { useGraphLensesStore } from '../../stores/graphLenses'
import { useUserPreferencesStore } from '../../stores/userPreferences'
import { useGraphThemeFocusStore } from '../../stores/graphThemeFocus'
import { themeRegionColor } from '../../utils/themeRegionPalette'

const artifacts = useArtifactsStore()
const lenses = useGraphLensesStore()
const userPrefs = useUserPreferencesStore()
const themeFocus = useGraphThemeFocusStore()

const COLLAPSED_KEY = 'ps_graph_theme_legend_collapsed'
const EXPANDED_SUPERS_KEY = 'ps_graph_theme_legend_expanded_supers'
/** USERPREFS-1 cross-device sync keys. */
const PREF_COLLAPSED = 'graphThemeLegendCollapsed'
const PREF_EXPANDED_SUPERS = 'graphThemeLegendExpandedSupers'

const collapsed = ref(false)
/** Set of super_theme_ids currently expanded. Multi-child supers default
 *  collapsed; single-child supers ignore this state. */
const expandedSupers = ref<Set<string>>(new Set())
let applyingRemoteCollapsed = false
let applyingRemoteExpanded = false

onMounted(() => {
  try {
    if (typeof localStorage === 'undefined') return
    collapsed.value = localStorage.getItem(COLLAPSED_KEY) === '1'
    const raw = localStorage.getItem(EXPANDED_SUPERS_KEY)
    if (raw) {
      const parsed = JSON.parse(raw) as unknown
      if (Array.isArray(parsed)) {
        expandedSupers.value = new Set(
          parsed.filter((v): v is string => typeof v === 'string'),
        )
      }
    }
  } catch {
    /* ignore quota / private mode / bad JSON */
  }
})

function toggleCollapsed(): void {
  collapsed.value = !collapsed.value
  try {
    if (typeof localStorage === 'undefined') return
    if (collapsed.value) localStorage.setItem(COLLAPSED_KEY, '1')
    else localStorage.removeItem(COLLAPSED_KEY)
  } catch {
    /* ignore */
  }
  if (applyingRemoteCollapsed) return
  void userPrefs.set(PREF_COLLAPSED, collapsed.value)
}

function toggleSuper(sid: string): void {
  const next = new Set(expandedSupers.value)
  if (next.has(sid)) next.delete(sid)
  else next.add(sid)
  expandedSupers.value = next
  try {
    if (typeof localStorage === 'undefined') return
    if (next.size > 0) {
      localStorage.setItem(EXPANDED_SUPERS_KEY, JSON.stringify(Array.from(next)))
    } else {
      localStorage.removeItem(EXPANDED_SUPERS_KEY)
    }
  } catch {
    /* ignore */
  }
  if (applyingRemoteExpanded) return
  void userPrefs.set(PREF_EXPANDED_SUPERS, Array.from(next))
}

/* USERPREFS-1 — apply server-hydrated state (+ cross-tab BroadcastChannel
   updates via the same reactive path). */
watch(
  () => userPrefs.get<boolean>(PREF_COLLAPSED),
  (v) => {
    if (typeof v !== 'boolean' || v === collapsed.value) return
    applyingRemoteCollapsed = true
    try { collapsed.value = v } finally { applyingRemoteCollapsed = false }
  },
  { immediate: true },
)
watch(
  () => userPrefs.get<string[]>(PREF_EXPANDED_SUPERS),
  (v) => {
    if (!Array.isArray(v)) return
    const remote = new Set(v.filter((x): x is string => typeof x === 'string'))
    // Skip a no-op update — cheap set-equality check.
    if (
      remote.size === expandedSupers.value.size &&
      Array.from(remote).every((x) => expandedSupers.value.has(x))
    ) {
      return
    }
    applyingRemoteExpanded = true
    try { expandedSupers.value = remote } finally { applyingRemoteExpanded = false }
  },
  { immediate: true },
)

const visible = computed(() => {
  return lenses.themeClusterRegions && artifacts.themeClustersDoc != null
})

interface ClusterRow {
  id: string
  label: string
  colour: string
  memberCount: number
}

interface SuperGroup {
  superId: string
  superLabel: string
  children: ClusterRow[]
  /** Aggregate colour for the group swatch — matches the first child's tint
   *  so users can link legend header ↔ canvas at a glance. */
  colour: string
  totalMembers: number
}

/** graph-v3 tier 7-2 — typeahead filter over super-theme + child labels.
 *  Ephemeral (per-tab); doesn't sync via USERPREFS-1 — search state doesn't
 *  really belong on other devices. */
const filterQuery = ref('')
const filterInputRef = ref<HTMLInputElement | null>(null)

function clearFilter(): void {
  filterQuery.value = ''
  filterInputRef.value?.focus()
}

const allGroups = computed<SuperGroup[]>(() => {
  const doc = artifacts.themeClustersDoc
  const clusters = (doc?.clusters ?? []) as TopicClustersCluster[]
  const byId = new Map<string, SuperGroup>()
  const order: string[] = []
  for (const cl of clusters) {
    const cid =
      typeof cl?.graph_compound_parent_id === 'string' ? cl.graph_compound_parent_id.trim() : ''
    if (!cid) continue
    const clabel =
      typeof cl?.canonical_label === 'string' && cl.canonical_label.trim()
        ? cl.canonical_label.trim()
        : cid
    const memberCount = typeof cl?.member_count === 'number' ? cl.member_count : 0
    /* Backwards compat with enricher v1.0.x — no super_theme_* → each cluster
       is its own group so the render collapses to the historical flat list. */
    const sid =
      typeof cl?.super_theme_id === 'string' && cl.super_theme_id.trim()
        ? cl.super_theme_id.trim()
        : cid
    const slabel =
      typeof cl?.super_theme_label === 'string' && cl.super_theme_label.trim()
        ? cl.super_theme_label.trim()
        : clabel
    let g = byId.get(sid)
    if (!g) {
      g = {
        superId: sid,
        superLabel: slabel,
        children: [],
        colour: themeRegionColor(cid),
        totalMembers: 0,
      }
      byId.set(sid, g)
      order.push(sid)
    }
    g.children.push({ id: cid, label: clabel, colour: themeRegionColor(cid), memberCount })
    g.totalMembers += memberCount
  }
  return order.map((sid) => byId.get(sid)!)
})

/** Filtered view: substring match against super-theme label AND any child
 *  label. Match on super-header keeps ALL children; match on a child keeps
 *  just the matches. Empty query = all groups. Match is case-insensitive. */
const groups = computed<SuperGroup[]>(() => {
  const q = filterQuery.value.trim().toLowerCase()
  if (!q) return allGroups.value
  const out: SuperGroup[] = []
  for (const g of allGroups.value) {
    if (g.superLabel.toLowerCase().includes(q)) {
      out.push(g)
      continue
    }
    const matched = g.children.filter((c) => c.label.toLowerCase().includes(q))
    if (matched.length > 0) {
      out.push({ ...g, children: matched })
    }
  }
  return out
})

/** graph-v3 tier 7-3 — click a row → focus on the graph canvas.
 *  Super-header click focuses every child cluster; cluster row click focuses
 *  just that one. Re-clicking the same target clears focus (toggle). */
function focusSuper(g: SuperGroup): void {
  const ids = new Set(g.children.map((c) => c.id))
  if (isSuperFocused(g)) {
    themeFocus.clearFocus()
  } else {
    themeFocus.setFocus(ids)
  }
}

function focusCluster(clusterId: string): void {
  if (isClusterFocused(clusterId) && themeFocus.focusedThemeIds.size === 1) {
    themeFocus.clearFocus()
  } else {
    themeFocus.setFocus([clusterId])
  }
}

function isSuperFocused(g: SuperGroup): boolean {
  if (!themeFocus.hasFocus() || g.children.length === 0) return false
  const focus = themeFocus.focusedThemeIds
  return g.children.every((c) => focus.has(c.id)) && focus.size === g.children.length
}

function isClusterFocused(clusterId: string): boolean {
  return themeFocus.isFocused(clusterId)
}

function isExpanded(g: SuperGroup): boolean {
  // Single-child groups are always "expanded" (flattened) — no chevron.
  if (g.children.length <= 1) return true
  // Active filter → always expand so matched children are visible even if
  // the user hasn't manually expanded this super-theme.
  if (filterQuery.value.trim()) return true
  return expandedSupers.value.has(g.superId)
}
</script>

<template>
  <div
    v-if="visible"
    class="pointer-events-auto absolute bottom-2 right-2 z-10 max-w-[14rem] rounded border border-border bg-surface/95 text-surface-foreground shadow-md backdrop-blur-[1px]"
    data-testid="graph-theme-legend"
  >
    <div class="flex items-center justify-between border-b border-border/60 px-2 py-1">
      <span class="text-[10px] uppercase tracking-wide text-muted">Theme regions</span>
      <button
        type="button"
        class="rounded px-1 text-[10px] leading-none text-muted hover:bg-overlay hover:text-surface-foreground"
        data-testid="graph-theme-legend-toggle"
        :aria-label="collapsed ? 'Expand theme legend' : 'Collapse theme legend'"
        @click="toggleCollapsed"
      >
        {{ collapsed ? '▸' : '▾' }}
      </button>
    </div>
    <div v-if="!collapsed" class="flex flex-col">
      <!-- graph-v3 tier 7-2 — typeahead filter. Hidden when there are <=6
           super-themes because scrolling isn't a problem at that size. -->
      <div
        v-if="allGroups.length > 6"
        class="border-b border-border/60 px-2 py-1"
      >
        <div class="relative">
          <input
            ref="filterInputRef"
            v-model="filterQuery"
            type="text"
            class="w-full rounded border border-border bg-surface px-1.5 py-0.5 text-[11px] leading-tight focus:outline-none focus:ring-1 focus:ring-primary"
            placeholder="Filter themes…"
            data-testid="graph-theme-legend-filter"
            aria-label="Filter theme legend"
          />
          <button
            v-if="filterQuery"
            type="button"
            class="absolute right-0.5 top-1/2 -translate-y-1/2 rounded px-1 text-[10px] leading-none text-muted hover:bg-overlay hover:text-surface-foreground"
            aria-label="Clear filter"
            data-testid="graph-theme-legend-filter-clear"
            @click="clearFilter"
          >
            ✕
          </button>
        </div>
      </div>
      <div class="max-h-[16rem] overflow-y-auto px-2 py-1">
      <template v-for="g in groups" :key="g.superId">
        <!-- Multi-child super-theme header. Chevron toggles expand; label
             toggles graph focus (tier 7-3). -->
        <div
          v-if="g.children.length > 1"
          class="flex items-center gap-0.5 py-0.5 text-[11px] font-medium"
          :class="isSuperFocused(g) ? 'bg-overlay/60' : ''"
          :data-testid="`graph-theme-legend-super-${g.superId}`"
        >
          <button
            type="button"
            class="rounded px-0.5 text-[9px] leading-none text-muted hover:bg-overlay hover:text-surface-foreground"
            :data-testid="`graph-theme-legend-super-toggle-${g.superId}`"
            :aria-expanded="isExpanded(g)"
            :aria-label="isExpanded(g) ? `Collapse ${g.superLabel}` : `Expand ${g.superLabel}`"
            @click="toggleSuper(g.superId)"
          >
            {{ isExpanded(g) ? '▾' : '▸' }}
          </button>
          <button
            type="button"
            class="flex flex-1 items-center gap-1.5 rounded px-0.5 text-left leading-none hover:bg-overlay"
            :data-testid="`graph-theme-legend-super-focus-${g.superId}`"
            :aria-pressed="isSuperFocused(g)"
            :aria-label="isSuperFocused(g) ? `Clear focus on ${g.superLabel}` : `Focus graph on ${g.superLabel}`"
            @click="focusSuper(g)"
          >
            <span
              class="inline-block h-3 w-3 shrink-0 rounded-sm border border-border/40"
              :style="{ backgroundColor: g.colour }"
              aria-hidden="true"
            />
            <span class="min-w-0 flex-1 truncate" :title="g.superLabel">
              {{ g.superLabel }}
            </span>
            <span class="shrink-0 text-[9px] font-mono text-muted">
              {{ g.children.length }}
            </span>
          </button>
        </div>
        <!-- Child cluster rows — flattened row when single-child, indented under
             header when multi-child + expanded. Whole row is a focus button. -->
        <ul
          v-if="isExpanded(g)"
          class="mb-0.5"
          :class="g.children.length > 1 ? 'ml-3 border-l border-border/40 pl-1.5' : ''"
        >
          <li
            v-for="r in g.children"
            :key="r.id"
            :data-testid="`graph-theme-legend-row-${r.id}`"
          >
            <button
              type="button"
              class="flex w-full items-center gap-1.5 rounded px-0.5 py-0.5 text-left text-[11px] leading-none hover:bg-overlay"
              :class="isClusterFocused(r.id) ? 'bg-overlay/60' : ''"
              :aria-pressed="isClusterFocused(r.id)"
              :aria-label="isClusterFocused(r.id) ? `Clear focus on ${r.label}` : `Focus graph on ${r.label}`"
              :data-testid="`graph-theme-legend-row-focus-${r.id}`"
              @click="focusCluster(r.id)"
            >
              <span
                class="inline-block h-3 w-3 shrink-0 rounded-sm border border-border/40"
                :style="{ backgroundColor: r.colour }"
                aria-hidden="true"
              />
              <span class="min-w-0 flex-1 truncate" :title="r.label">{{ r.label }}</span>
              <span class="shrink-0 text-[9px] font-mono text-muted">{{ r.memberCount }}</span>
            </button>
          </li>
        </ul>
      </template>
      <p v-if="groups.length === 0" class="py-1 text-[10px] text-muted">
        {{ filterQuery ? 'No matches.' : 'No clusters in this corpus.' }}
      </p>
      </div>
    </div>
  </div>
</template>
