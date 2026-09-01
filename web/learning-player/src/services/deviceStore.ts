/**
 * Device-local key/value storage (Capacitor Preferences → iOS `UserDefaults` / Android
 * `SharedPreferences`; `localStorage` under the web fallback).
 *
 * Deliberately NOT `stores/userPreferences.ts`. That store syncs through
 * `/api/app/preferences`, so a value follows the *account* across devices. Everything stored
 * here is the opposite: state that belongs to ONE device and must not travel (#1905) — which
 * episodes sit on *this* phone's disk, and whether *this* phone may download over cellular.
 * A metered phone and an unmetered tablet must be able to disagree.
 */

import { Preferences } from '@capacitor/preferences'

export async function getDeviceJson<T>(key: string): Promise<T | null> {
  const { value } = await Preferences.get({ key })
  if (value == null) return null
  try {
    return JSON.parse(value) as T
  } catch {
    // A corrupt value is not worth crashing a launch over: report absent so the caller
    // rebuilds from empty instead of throwing on every read.
    return null
  }
}

export async function setDeviceJson(key: string, value: unknown): Promise<void> {
  await Preferences.set({ key, value: JSON.stringify(value) })
}

export async function removeDeviceKey(key: string): Promise<void> {
  await Preferences.remove({ key })
}
