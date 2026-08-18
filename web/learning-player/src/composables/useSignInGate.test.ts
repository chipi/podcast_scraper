import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { defineComponent } from 'vue'
import { createMemoryHistory, createRouter } from 'vue-router'
import { useAuthStore } from '../stores/auth'
import { useSignInGate } from './useSignInGate'

const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', name: 'home', component: { template: '<div/>' } },
    { path: '/login', name: 'login', component: { template: '<div/>' } },
    { path: '/episode/:slug', name: 'player', component: { template: '<div/>' } },
  ],
})

const Host = defineComponent({
  setup() {
    return useSignInGate()
  },
  template: '<div/>',
})

async function mountAt(path: string) {
  await router.push(path)
  await router.isReady()
  return mount(Host, { global: { plugins: [router] } })
}

beforeEach(() => setActivePinia(createPinia()))

describe('useSignInGate (#1590)', () => {
  it('runs the action directly when signed in', async () => {
    useAuthStore().user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
    const push = vi.spyOn(router, 'push')
    const action = vi.fn()
    const w = await mountAt('/')
    push.mockClear()

    w.vm.gated(action)()
    await flushPromises()

    expect(action).toHaveBeenCalledOnce()
    expect(push).not.toHaveBeenCalled()
    expect(w.vm.isGated).toBe(false)
  })

  it('sends a signed-out visitor to sign-in instead of running the action', async () => {
    const action = vi.fn()
    const w = await mountAt('/')
    const push = vi.spyOn(router, 'push')

    w.vm.gated(action)()
    await flushPromises()

    expect(action).not.toHaveBeenCalled()
    expect(push).toHaveBeenCalledWith({ name: 'login', query: { redirect: '/' } })
    expect(w.vm.isGated).toBe(true)
  })

  it('preserves query and hash in the redirect, so the exact moment survives', async () => {
    // fullPath, not path: the point of the teaser is that the action they wanted is one step away.
    // Landing back on /episode/x instead of /episode/x?t=1830 loses the thing they were reaching for.
    const w = await mountAt('/episode/ep-1?t=1830')
    const push = vi.spyOn(router, 'push')

    w.vm.gated(vi.fn())()
    await flushPromises()

    expect(push).toHaveBeenCalledWith({
      name: 'login',
      query: { redirect: '/episode/ep-1?t=1830' },
    })
  })

  it('waits for the session to resolve before deciding (a fast tap is not a signed-out tap)', async () => {
    // `isAuthenticated` is false until App.vue's onMounted refresh lands. Without awaiting that, a
    // user who taps in the first moments after load is sent to sign in — discarding the session
    // they already had. Caught by follow-show.spec, which signs in and then clicks immediately.
    const auth = useAuthStore()
    auth.loaded = false
    const refresh = vi.spyOn(auth, 'refresh').mockImplementation(async () => {
      auth.user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
      auth.loaded = true
    })
    const push = vi.spyOn(router, 'push')
    const action = vi.fn()
    const w = await mountAt('/')
    push.mockClear()

    w.vm.gated(action)()
    await flushPromises()

    expect(refresh).toHaveBeenCalled()
    expect(action).toHaveBeenCalledOnce()
    expect(push).not.toHaveBeenCalled()
  })
})
