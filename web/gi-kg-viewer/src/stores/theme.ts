import { defineStore } from 'pinia'
import { ref, watch } from 'vue'

export type ThemeChoice = 'light' | 'dark' | 'auto'

const STORAGE_KEY = 'gi-kg-viewer-theme'

function applyTheme(choice: ThemeChoice): void {
  const root = document.documentElement
  if (choice === 'auto') {
    root.removeAttribute('data-theme')
  } else {
    root.setAttribute('data-theme', choice)
  }
}

function loadSaved(): ThemeChoice {
  try {
    const v = localStorage.getItem(STORAGE_KEY)
    if (v === 'light' || v === 'dark' || v === 'auto') return v
  } catch {
    /* ignore */
  }
  return 'dark'
}

export const useThemeStore = defineStore('theme', () => {
  const choice = ref<ThemeChoice>(loadSaved())

  applyTheme(choice.value)

  watch(choice, (v) => {
    applyTheme(v)
    try {
      localStorage.setItem(STORAGE_KEY, v)
    } catch {
      /* ignore */
    }
  })

  function cycle(): void {
    const order: ThemeChoice[] = ['light', 'dark', 'auto']
    const idx = order.indexOf(choice.value)
    choice.value = order[(idx + 1) % order.length]
  }

  return { choice, cycle }
})
