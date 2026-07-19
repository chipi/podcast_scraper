import { createApp } from 'vue'
import { createPinia } from 'pinia'
import './style.css'
import App from './App.vue'
import { router } from './router'
import { i18n } from './i18n'
import { applyTheme } from './theme/theme'

applyTheme('dark')

// Expose build identity for update-path debugging. When a user reports
// "the PWA isn't updating", the running client's sha + time can be read
// from window.__buildInfo (DevTools console or a support form) to
// distinguish "stuck client" from "cache never invalidated". See
// vite.config.ts `define:` block for the injection.
window.__buildInfo = { sha: __BUILD_SHA__, time: __BUILD_TIME__ }

console.info(`[app] Learning Player build=${__BUILD_SHA__} time=${__BUILD_TIME__}`)

createApp(App).use(createPinia()).use(router).use(i18n).mount('#app')

// USERPREFS-1 (#1213) — hydrate the user preferences payload once at app
// init. Consumers (HomeView, PlayerView, future adopters) read via
// ``useUserPreferencesStore().get(key)`` and get server values when
// available, undefined when not. Fire-and-forget so mount doesn't wait
// on the network round-trip; consuming stores react when the promise
// resolves. Silent-degrade on 401 / offline.
void (async () => {
  const { useUserPreferencesStore } = await import('./stores/userPreferences')
  await useUserPreferencesStore().hydrate()
})()
