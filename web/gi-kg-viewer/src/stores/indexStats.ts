import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'
import type { IndexStatsEnvelope } from '../api/indexStatsApi'
import { fetchIndexStats, postIndexRebuild } from '../api/indexStatsApi'
import { formatBytes } from '../utils/formatting'
import type { MetricRow } from '../utils/metrics'
import { useShellStore } from './shell'

const INDEX_REASON_LABELS: Record<string, string> = {
  artifacts_newer_than_index:
    'Index-related files on disk are newer than the last index build. Run `podcast index` on the corpus root (use `--rebuild` after model or chunking changes).',
  no_index_but_metadata:
    'Episode metadata exists but there is no usable FAISS index at the expected path.',
  embedding_model_mismatch:
    'Embedding model differs from the default (or from the model you passed in the API). Rebuild the index so embeddings stay consistent.',
  multi_feed_batch_incomplete:
    'The last multi-feed batch reported failures; confirm corpus state before trusting search results.',
}

function expandIndexReasonLines(env: IndexStatsEnvelope): string[] {
  const lines: string[] = []
  const codes = env.reindex_reasons ?? []
  for (const code of codes) {
    if (code === 'corpus_search_parent_hint') {
      for (const h of env.search_root_hints ?? []) {
        if (!lines.includes(h)) {
          lines.push(h)
        }
      }
      continue
    }
    const label = INDEX_REASON_LABELS[code] ?? code
    if (!lines.includes(label)) {
      lines.push(label)
    }
  }
  return lines
}

export const useIndexStatsStore = defineStore('indexStats', () => {
  const shell = useShellStore()

  const indexEnvelope = ref<IndexStatsEnvelope | null>(null)
  const indexLoading = ref(false)
  const indexError = ref<string | null>(null)
  const rebuildSubmitting = ref(false)
  let rebuildPollTimer: ReturnType<typeof setInterval> | null = null

  function stopRebuildPoll(): void {
    if (rebuildPollTimer !== null) {
      clearInterval(rebuildPollTimer)
      rebuildPollTimer = null
    }
  }

  function startRebuildPoll(): void {
    stopRebuildPoll()
    let ticks = 0
    rebuildPollTimer = setInterval(() => {
      ticks += 1
      void refreshIndexStats().then(() => {
        const env = indexEnvelope.value
        if (!env?.rebuild_in_progress || ticks >= 80) {
          stopRebuildPoll()
        }
      })
    }, 2500)
  }

  const indexRows = computed((): MetricRow[] => {
    const env = indexEnvelope.value
    if (!env?.available || !env.stats) return []
    const s = env.stats
    const feeds =
      s.feeds_indexed.length > 0 ? s.feeds_indexed.join(', ') : '—'
    return [
      { k: 'Index path', v: env.index_path || '—' },
      { k: 'Total vectors', v: String(s.total_vectors) },
      { k: 'Embedding model', v: s.embedding_model || '—' },
      { k: 'Embedding dim', v: String(s.embedding_dim) },
      { k: 'Last updated', v: s.last_updated || '—' },
      { k: 'On-disk size', v: formatBytes(s.index_size_bytes) },
      { k: 'Feeds indexed', v: feeds },
    ]
  })

  const indexHealthBanner = computed((): {
    kind: 'warn' | 'info'
    lines: string[]
  } | null => {
    const env = indexEnvelope.value
    if (!env || !shell.healthStatus) {
      return null
    }
    const lines = expandIndexReasonLines(env)
    if (!lines.length) {
      return null
    }
    if (env.reindex_recommended) {
      return { kind: 'warn', lines }
    }
    return { kind: 'info', lines }
  })

  const rebuildActionsDisabled = computed(
    () =>
      !shell.healthStatus ||
      indexLoading.value ||
      rebuildSubmitting.value ||
      indexEnvelope.value?.reason === 'faiss_unavailable' ||
      indexEnvelope.value?.rebuild_in_progress === true,
  )

  async function requestIndexRebuild(full: boolean): Promise<void> {
    indexError.value = null
    rebuildSubmitting.value = true
    try {
      const path = shell.hasCorpusPath ? shell.corpusPath.trim() : undefined
      await postIndexRebuild({ corpusPath: path, rebuild: full })
      await refreshIndexStats()
      if (indexEnvelope.value?.rebuild_in_progress) {
        startRebuildPoll()
      }
    } catch (e) {
      indexError.value = e instanceof Error ? e.message : String(e)
    } finally {
      rebuildSubmitting.value = false
    }
  }

  async function refreshIndexStats(): Promise<void> {
    indexError.value = null
    if (!shell.healthStatus) {
      indexEnvelope.value = null
      return
    }
    indexLoading.value = true
    try {
      const path = shell.hasCorpusPath ? shell.corpusPath.trim() : undefined
      indexEnvelope.value = await fetchIndexStats(path)
    } catch (e) {
      indexEnvelope.value = null
      indexError.value = e instanceof Error ? e.message : String(e)
    } finally {
      indexLoading.value = false
    }
  }

  watch(
    () => [shell.corpusPath, shell.healthStatus] as const,
    () => {
      void refreshIndexStats()
    },
    { immediate: true },
  )

  watch(
    () => indexEnvelope.value?.rebuild_in_progress,
    (on) => {
      if (on) {
        startRebuildPoll()
      }
    },
  )

  return {
    indexEnvelope,
    indexLoading,
    indexError,
    rebuildSubmitting,
    indexRows,
    indexHealthBanner,
    rebuildActionsDisabled,
    refreshIndexStats,
    requestIndexRebuild,
  }
})
