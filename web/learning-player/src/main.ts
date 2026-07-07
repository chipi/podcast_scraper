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
