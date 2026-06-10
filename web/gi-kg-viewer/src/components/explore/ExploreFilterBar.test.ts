// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { useExploreStore } from '../../stores/explore'
import ExploreFilterBar from './ExploreFilterBar.vue'

const BAR = '[data-testid="explore-filter-bar"]'

/**
 * Stub the child chips so the bar renders standalone. ExploreTextChip is a
 * v-model component: the stub surfaces its bound props as JSON and re-emits
 * update:modelValue / submit so we can drive the parent's logic. ExploreMoreChip
 * re-emits open.
 */
const TextChipStub = {
  name: 'ExploreTextChip',
  props: [
    'modelValue',
    'label',
    'chipTestid',
    'popoverTestid',
    'enabled',
    'disabledTitle',
  ],
  emits: ['update:modelValue', 'submit'],
  template:
    '<div :data-testid="`stub-text-${label}`" :data-props="JSON.stringify($props)" />',
}

const MoreChipStub = {
  name: 'ExploreMoreChip',
  emits: ['open'],
  template: '<div data-testid="stub-more" />',
}

const STUBS = {
  ExploreTextChip: TextChipStub,
  ExploreMoreChip: MoreChipStub,
}

const mountBar = (props: { enabled?: boolean; disabledTitle?: string } = {}) =>
  mount(ExploreFilterBar, {
    props: { enabled: true, ...props },
    attachTo: document.body,
    global: { stubs: STUBS },
  })

const propsOf = (w: ReturnType<typeof mountBar>, label: string) =>
  JSON.parse(w.get(`[data-testid="stub-text-${label}"]`).attributes('data-props')!)

describe('ExploreFilterBar', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('renders the region wrapper with an accessible label', () => {
    const w = mountBar()
    const bar = w.get(BAR)
    expect(bar.attributes('role')).toBe('region')
    expect(bar.attributes('aria-label')).toBe('Explore filters')
  })

  it('renders both text chips plus the More chip', () => {
    const w = mountBar()
    expect(w.find('[data-testid="stub-text-Topic"]').exists()).toBe(true)
    expect(w.find('[data-testid="stub-text-Speaker"]').exists()).toBe(true)
    expect(w.find('[data-testid="stub-more"]').exists()).toBe(true)
  })

  it('wires the Topic chip with the right static props', () => {
    const w = mountBar()
    const p = propsOf(w, 'Topic')
    expect(p.label).toBe('Topic')
    expect(p.chipTestid).toBe('explore-chip-topic')
    expect(p.popoverTestid).toBe('explore-popover-topic')
  })

  it('wires the Speaker chip with the right static props', () => {
    const w = mountBar()
    const p = propsOf(w, 'Speaker')
    expect(p.label).toBe('Speaker')
    expect(p.chipTestid).toBe('explore-chip-speaker')
    expect(p.popoverTestid).toBe('explore-popover-speaker')
  })

  it('binds the chips v-model to the store filters (get)', () => {
    const ex = useExploreStore()
    ex.filters.topic = 'rust'
    ex.filters.speaker = 'alice'
    const w = mountBar()
    expect(propsOf(w, 'Topic').modelValue).toBe('rust')
    expect(propsOf(w, 'Speaker').modelValue).toBe('alice')
  })

  it('writes the Topic chip update back into the store (set)', async () => {
    const ex = useExploreStore()
    const w = mountBar()
    w.getComponent(TextChipStub).vm.$emit('update:modelValue', 'graphs')
    await w.vm.$nextTick()
    expect(ex.filters.topic).toBe('graphs')
  })

  it('writes the Speaker chip update back into the store (set)', async () => {
    const ex = useExploreStore()
    const w = mountBar()
    // Second ExploreTextChip is the Speaker chip.
    const chips = w.findAllComponents(TextChipStub)
    chips[1].vm.$emit('update:modelValue', 'bob')
    await w.vm.$nextTick()
    expect(ex.filters.speaker).toBe('bob')
  })

  it('re-emits submit from either text chip', async () => {
    const w = mountBar()
    const chips = w.findAllComponents(TextChipStub)
    chips[0].vm.$emit('submit')
    chips[1].vm.$emit('submit')
    expect(w.emitted('submit')).toHaveLength(2)
  })

  it('re-emits open-more when the More chip emits open', () => {
    const w = mountBar()
    w.getComponent(MoreChipStub).vm.$emit('open')
    expect(w.emitted('open-more')).toHaveLength(1)
  })

  it('forwards enabled + disabledTitle down to the text chips', () => {
    const w = mountBar({ enabled: false, disabledTitle: 'load a corpus' })
    const p = propsOf(w, 'Topic')
    expect(p.enabled).toBe(false)
    expect(p.disabledTitle).toBe('load a corpus')
  })

  it('does not dim the More chip wrapper when enabled', () => {
    const w = mountBar({ enabled: true })
    const wrapper = w.get('[data-testid="stub-more"]').element.parentElement!
    expect(wrapper.className).not.toContain('opacity-50')
    expect(wrapper.className).not.toContain('pointer-events-none')
  })

  it('dims + disables the More chip wrapper when disabled', () => {
    const w = mountBar({ enabled: false })
    const wrapper = w.get('[data-testid="stub-more"]').element.parentElement!
    expect(wrapper.className).toContain('opacity-50')
    expect(wrapper.className).toContain('pointer-events-none')
  })
})
