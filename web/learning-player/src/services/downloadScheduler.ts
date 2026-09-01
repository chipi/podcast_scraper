/**
 * When a queued download is allowed to start (#1905, slice 3).
 *
 * This is the **L1** design recorded on the issue: transfers run in the app process, so a queued
 * episode starts when the user flags it, when the network changes *while the app is foregrounded*,
 * or when the app resumes. It deliberately does NOT download while the app is closed — that is L2,
 * and it needs iOS background `URLSession` / Android `WorkManager`, neither of which
 * `@capacitor/filesystem` exposes.
 */

import { App } from '@capacitor/app'
import { Network } from '@capacitor/network'
import { useDownloadsStore } from '../stores/downloads'
import { getDeviceJson, setDeviceJson } from './deviceStore'
import { downloadEpisode } from './downloads'
import { isNative } from './native'

export type NetworkPolicy = 'wifi-only' | 'any'

/** Per-device, not per-account: one phone may be metered while a tablet is not (#1905). */
export const POLICY_KEY = 'downloads.networkPolicy'
export const DEFAULT_POLICY: NetworkPolicy = 'wifi-only'

interface Connection {
  connected: boolean
  connectionType: string
}

/**
 * Pure policy table:
 *
 * | connectionType   | wifi-only | any   |
 * | ---------------- | --------- | ----- |
 * | wifi             | start     | start |
 * | cellular         | queue     | start |
 * | none / unknown   | queue     | queue |
 *
 * `unknown` never starts: guessing wrong costs the user real money on a metered link.
 */
export function allows(policy: NetworkPolicy, status: Connection): boolean {
  if (!status.connected) return false
  if (status.connectionType === 'wifi') return true
  if (status.connectionType === 'cellular') return policy === 'any'
  return false
}

export async function getNetworkPolicy(): Promise<NetworkPolicy> {
  return (await getDeviceJson<NetworkPolicy>(POLICY_KEY)) ?? DEFAULT_POLICY
}

export async function setNetworkPolicy(policy: NetworkPolicy): Promise<void> {
  await setDeviceJson(POLICY_KEY, policy)
  // Relaxing the policy should start what was waiting, without making the user hunt for it.
  void drainQueue()
}

// One drain at a time. The triggers overlap by nature — a resume often arrives together with a
// network change — and two drains would race into the same slug.
let draining = false

/**
 * Start every waiting download the current connection permits, oldest first.
 *
 * Sequential on purpose: several large transfers over one link finish no sooner in aggregate and
 * make each individual one look stalled.
 */
export async function drainQueue(): Promise<void> {
  if (!isNative() || draining) return
  draining = true
  try {
    const store = useDownloadsStore()
    await store.ensureLoaded()
    const policy = await getNetworkPolicy()

    // A transient failure earns another attempt now that the network moved. A permanent one
    // (corpus removal → 404) must not be retried on every change, forever.
    for (const entry of Object.values(store.entries)) {
      if (entry.state === 'failed' && entry.errorKind !== 'permanent') {
        await store.mark(entry.slug)
      }
    }

    for (const slug of store.queued) {
      // Re-checked per item: the link can drop or switch to cellular mid-drain.
      if (!allows(policy, await Network.getStatus())) return
      await downloadEpisode(slug)
    }
  } finally {
    draining = false
  }
}

/** Flag an episode and start it immediately if the connection already allows it. */
export async function markForOffline(slug: string): Promise<boolean> {
  const store = useDownloadsStore()
  await store.ensureLoaded()
  const changed = await store.mark(slug)
  void drainQueue()
  return changed
}

/** Wire the L1 triggers. Call once at boot. */
export async function startDownloadScheduler(): Promise<void> {
  if (!isNative()) return
  await Network.addListener('networkStatusChange', () => void drainQueue())
  await App.addListener('resume', () => void drainQueue())
  void drainQueue()
}
