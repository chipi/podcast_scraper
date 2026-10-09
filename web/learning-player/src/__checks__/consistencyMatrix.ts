/**
 * Every WRITE the app makes, the surfaces that must reflect it, and the test that proves they do
 * (2026-10-09). Guarded by `consistency-matrix.test.ts`: a write function added to `services/api.ts`
 * without a row here fails the build, so where a change must show — and how that is tested — is
 * decided when the write is written, not discovered on a device.
 *
 * Why: ten bugs on 2026-10-09 were one shape — a write on one surface, and another surface (kept
 * alive, or reading a store nobody told) still showing what it had before. Nothing enumerated the
 * writes, so nothing could say a pair was untested. docs/wip/e2e-cross-surface-gaps-2026-10-09.md
 *
 * `readers`: surfaces OTHER than the one making the write. Empty means the write shows nowhere else
 * (say why in `note`). `proof`: repo-relative test files (under web/learning-player) that cover the
 * readers; an `e2e/cross-surface.spec.ts` row is the strongest (warm tab, in-app navigation).
 */
export interface ConsistencyRow {
  write: string
  readers: string[]
  proof: string[]
  note?: string
}

export const CONSISTENCY_MATRIX: ConsistencyRow[] = [
  // --- favourites ---------------------------------------------------------------------------------
  { write: 'addFavorite', readers: ['Library › Saved', 'every heart'], proof: ['src/composables/useFavoritesPage.test.ts', 'e2e/library-saved.spec.ts'] },
  { write: 'removeFavorite', readers: ['Library › Saved', 'every heart'], proof: ['src/composables/useFavoritesPage.test.ts', 'e2e/library-saved.spec.ts'] },
  { write: 'setFavoriteColor', readers: ['Library › Saved colour filter'], proof: ['src/composables/useFavoritesPage.test.ts'] },

  // --- interests ----------------------------------------------------------------------------------
  { write: 'addInterest', readers: ['Discover › Your trends', 'Profile › Interests', 'Home recommended'], proof: ['e2e/cross-surface.spec.ts', 'src/stores/interests.test.ts'] },
  { write: 'removeInterest', readers: ['Discover › Your trends', 'Profile › Interests'], proof: ['e2e/cross-surface.spec.ts', 'src/stores/interests.test.ts'] },
  { write: 'putUserInterests', readers: ['every interests reader (store replaceAll)'], proof: ['src/components/InterestsPicker.test.ts'] },

  // --- follows ------------------------------------------------------------------------------------
  { write: 'followShow', readers: ['Library › Following', "Home › What's new"], proof: ['e2e/follow-show.spec.ts', 'e2e/welcome-follow-shows.spec.ts'] },
  { write: 'unfollowShow', readers: ['Library › Following', "Home › What's new"], proof: ['e2e/follow-show.spec.ts', 'src/stores/library.test.ts'] },

  // --- listening ----------------------------------------------------------------------------------
  { write: 'putPlayback', readers: ['Home › Continue listening', 'Home › Your Week', 'Queue › Recently played'], proof: ['e2e/cross-surface.spec.ts'] },
  { write: 'logListen', readers: ['Home › Your Week', 'Profile › Stats'], proof: ['src/services/listenLog.test.ts', 'src/views/ProfileView.test.ts'] },
  { write: 'logPlaybackProgress', readers: ['Profile › Stats'], proof: ['src/services/listenLog.test.ts'] },
  { write: 'markCompleted', readers: ['played mark on every episode row'], proof: ['src/stores/completed.test.ts', 'e2e/offline-played-sync.spec.ts'] },
  { write: 'unmarkCompleted', readers: ['played mark on every episode row'], proof: ['src/stores/completed.test.ts'] },
  { write: 'clearListeningHistory', readers: ['played marks', 'Queue › Recently played', 'Profile › Stats'], proof: ['e2e/cross-surface.spec.ts', 'src/views/ProfileView.test.ts'] },

  // --- queue --------------------------------------------------------------------------------------
  { write: 'putQueue', readers: ['masthead queue badge', 'Queue'], proof: ['src/stores/queue.test.ts', 'e2e/queue-panel.spec.ts'] },
  { write: 'addQueueItem', readers: ['masthead queue badge', 'Queue'], proof: ['src/stores/queue.test.ts', 'e2e/queue-panel.spec.ts'] },
  { write: 'removeQueueItem', readers: ['masthead queue badge', 'Queue'], proof: ['src/stores/queue.test.ts', 'e2e/queue-panel.spec.ts'] },

  // --- capture ------------------------------------------------------------------------------------
  { write: 'createHighlight', readers: ['Library › Saved highlights', 'Revisit'], proof: ['src/composables/useHighlightsPage.test.ts', 'e2e/capture.spec.ts'] },
  { write: 'patchHighlight', readers: ['Library › Saved highlights'], proof: ['src/composables/useHighlightsPage.test.ts'] },
  { write: 'deleteHighlight', readers: ['Library › Saved highlights', 'Revisit'], proof: ['src/composables/useHighlightsPage.test.ts', 'src/stores/capture.test.ts'] },
  { write: 'unretireHighlight', readers: ['Revisit'], proof: ['src/stores/capture.test.ts'] },
  { write: 'createNote', readers: ['Boards › notes', 'Search › Your notes'], proof: ['src/stores/capture.test.ts', 'e2e/capture.spec.ts'] },
  { write: 'patchNote', readers: ['Boards › notes', 'Search › Your notes'], proof: ['src/stores/capture.test.ts'] },
  { write: 'deleteNote', readers: ['Boards › notes', 'Search › Your notes'], proof: ['src/stores/capture.test.ts'] },

  // --- resurfacing --------------------------------------------------------------------------------
  { write: 'retireHighlight', readers: ['Revisit count', 'Library › Revisit'], proof: ['src/views/ResurfacingInbox.test.ts', 'src/stores/resurfacing.test.ts'] },
  { write: 'markSurfaced', readers: ['Revisit count', 'Library › Revisit'], proof: ['src/views/PlayerView.test.ts', 'src/views/ResurfacingInbox.test.ts'] },
  { write: 'putResurfacingSettings', readers: ['Revisit count (paused hides it)'], proof: ['src/views/ResurfacingInbox.test.ts'] },

  // --- boards -------------------------------------------------------------------------------------
  { write: 'createCollection', readers: ['Library › Boards', 'Home boards teaser', 'Save sheet'], proof: ['e2e/cross-surface.spec.ts', 'src/components/AddToCollectionButton.test.ts'] },
  { write: 'addToCollection', readers: ['Library › Boards (count)', 'Home boards teaser'], proof: ['e2e/cross-surface.spec.ts', 'src/views/CollectionsView.test.ts'] },
  { write: 'removeFromCollection', readers: ['Library › Boards (count)', 'Home boards teaser'], proof: ['src/views/CollectionsView.test.ts'] },
  { write: 'deleteCollection', readers: ['Home boards teaser', 'Save sheet'], proof: ['e2e/cross-surface.spec.ts', 'src/views/CollectionsView.test.ts'] },
  { write: 'reorderCollections', readers: ['Home boards teaser order'], proof: ['e2e/collections-reorder.spec.ts'] },

  // --- notifications ------------------------------------------------------------------------------
  { write: 'markNotificationRead', readers: ['masthead bell badge'], proof: ['src/stores/notifications.test.ts'] },
  { write: 'markAllNotificationsRead', readers: ['masthead bell badge'], proof: ['e2e/notifications-mark-all.spec.ts'] },

  // --- account ------------------------------------------------------------------------------------
  { write: 'setProfileName', readers: ['masthead avatar initials'], proof: ['src/views/ProfileView.test.ts'] },
  { write: 'uploadAvatar', readers: ['masthead avatar'], proof: ['src/views/ProfileView.test.ts'] },
  { write: 'putComms', readers: [], proof: ['src/views/ProfileView.test.ts'], note: 'Profile › Account only.' },
  { write: 'logout', readers: ['every per-account surface (cleared)'], proof: ['e2e/account-switch.spec.ts'] },
  { write: 'deleteAccount', readers: ['every per-account surface (cleared)'], proof: ['e2e/delete-account.spec.ts'] },
  { write: 'requestMagicLink', readers: [], proof: ['e2e/magic-link-welcome.spec.ts'], note: 'Sign-in only; no signed-in surface reads it.' },
  { write: 'subscribePush', readers: [], proof: ['src/composables/usePushSubscription.test.ts'], note: 'Device push registration; shown only where it is toggled.' },
  { write: 'unsubscribePush', readers: [], proof: ['src/composables/usePushSubscription.test.ts'], note: 'Device push registration; shown only where it is toggled.' },
  { write: 'createMcpToken', readers: [], proof: ['src/components/ConnectedAgents.test.ts'], note: 'Settings › Connected agents only.' },
  { write: 'revokeMcpToken', readers: [], proof: ['src/components/ConnectedAgents.test.ts'], note: 'Settings › Connected agents only.' },
  { write: 'revokeMcpConnection', readers: [], proof: ['src/components/ConnectedAgents.test.ts'], note: 'Settings › Connected agents only.' },

  // --- telemetry ----------------------------------------------------------------------------------
  { write: 'recordDiscoverClick', readers: [], proof: [], note: 'Ranking telemetry; no surface displays it.' },
  { write: 'postAppExits', readers: [], proof: ['src/services/lifecycle.test.ts'], note: 'Lifecycle telemetry; no surface displays it.' },
]
