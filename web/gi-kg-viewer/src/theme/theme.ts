import { DEFAULT_PRESET_ID, type PresetId } from './presets/default'

const DATA_PRESET = 'data-ps-preset'

/** Apply a named preset on :root (future: swap token sets without touching components). */
export function applyPreset(preset: PresetId = DEFAULT_PRESET_ID): void {
  document.documentElement.setAttribute(DATA_PRESET, preset)
}

export function currentPreset(): PresetId {
  const v = document.documentElement.getAttribute(DATA_PRESET)
  return v === DEFAULT_PRESET_ID ? DEFAULT_PRESET_ID : DEFAULT_PRESET_ID
}

/** Read a semantic token from computed styles (for canvas / Chart.js bridges in later milestones). */
export function getTokenVar(name: `--ps-${string}`): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}
