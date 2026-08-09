// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import TopicClustersStatusBlock from './TopicClustersStatusBlock.vue'
import { useShellStore } from '../../stores/shell'
import { useArtifactsStore } from '../../stores/artifacts'

function mountWith(loadState: string) {
  const shell = useShellStore()
  shell.corpusPath = '/corpus'
  shell.healthStatus = 'ok'
  const artifacts = useArtifactsStore()
  artifacts.$patch({ topicClustersLoadState: loadState })
  const spy = vi.spyOn(artifacts, 'rebuildTopicClusters').mockResolvedValue()
  const w = mount(TopicClustersStatusBlock)
  return { w, artifacts, spy }
}

describe('TopicClustersStatusBlock — Rebuild button (task-#14)', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it.each(['missing', 'error'])('shows the Rebuild button when clusters are %s', (state) => {
    const { w } = mountWith(state)
    expect(w.find('[data-testid="topic-clusters-rebuild"]').exists()).toBe(true)
  })

  it.each(['ok', 'idle', 'local_files'])('hides the Rebuild button when %s', (state) => {
    const { w } = mountWith(state)
    expect(w.find('[data-testid="topic-clusters-rebuild"]').exists()).toBe(false)
  })

  it('triggers rebuildTopicClusters on click', async () => {
    const { w, spy } = mountWith('missing')
    await w.find('[data-testid="topic-clusters-rebuild"]').trigger('click')
    expect(spy).toHaveBeenCalledOnce()
  })

  it('disables the button and shows a busy label while rebuilding', async () => {
    const { w, artifacts } = mountWith('missing')
    artifacts.$patch({ topicClustersRebuilding: true })
    await w.vm.$nextTick()
    const btn = w.find('[data-testid="topic-clusters-rebuild"]')
    expect(btn.attributes('disabled')).toBeDefined()
    expect(btn.text()).toBe('Rebuilding…')
  })
})
