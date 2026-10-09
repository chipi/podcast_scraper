import { mount, type VueWrapper } from '@vue/test-utils'
import { defineComponent, h, KeepAlive, ref, type Component } from 'vue'

/**
 * Mount `comp` the way a kept-alive tab holds it (App.vue `KEEP_ALIVE_TABS`), so a test can LEAVE
 * the tab and COME BACK — a return is what `onMounted` never sees (2026-10-09).
 *
 * `late: true` mounts the component only after its tab is already showing, as a lazily-rendered
 * section or a tab panel on its first visit does. That component gets NO activation for its mount,
 * so its first `onActivated` is the first return — the case a "skip the first activation" guard
 * gets wrong (it shipped that way, and only the e2e cross-surface spec caught it).
 *
 *   const tab = await keptAlive(YourWeek, { plugins: [i18n, router], late: true })
 *   await tab.leave(); await tab.back()
 */
export async function keptAlive(
  comp: Component,
  opts: { plugins?: unknown[]; props?: Record<string, unknown>; late?: boolean } = {},
): Promise<{ wrapper: VueWrapper; leave: () => Promise<void>; back: () => Promise<void> }> {
  const { flushPromises } = await import('@vue/test-utils')
  const shown = ref(true)
  const inner = ref(!opts.late)
  const Tab = defineComponent({
    setup: () => () => h('div', inner.value ? [h(comp, opts.props ?? {})] : []),
  })
  const Host = defineComponent({
    setup: () => () => h(KeepAlive, null, shown.value ? [h(Tab)] : []),
  })
  const wrapper = mount(Host, { global: { plugins: (opts.plugins ?? []) as never } })
  await flushPromises()
  if (opts.late) {
    inner.value = true
    await flushPromises()
  }
  return {
    wrapper,
    leave: async () => {
      shown.value = false
      await flushPromises()
    },
    back: async () => {
      shown.value = true
      await flushPromises()
    },
  }
}
