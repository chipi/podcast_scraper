/**
 * Settings saved under the app's earlier name (the GI/KG viewer) move to the Workbench keys once,
 * so a rename does not reset a browser's saved choices. The old key's value is copied only when
 * the new key is absent, then the old key is removed.
 */
export function migrateStorageKey(newKey: string, oldKey: string): void {
  try {
    if (typeof localStorage === 'undefined') return
    const old = localStorage.getItem(oldKey)
    if (old === null) return
    if (localStorage.getItem(newKey) === null) {
      localStorage.setItem(newKey, old)
    }
    localStorage.removeItem(oldKey)
  } catch {
    /* ignore quota / private mode */
  }
}
