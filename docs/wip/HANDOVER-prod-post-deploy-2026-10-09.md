# Handover — prod, after the deploy of main (2026-10-09)

For the agent deploying main to prod. **Supersedes nothing**: the plan in
`docs/wip/HANDOVER-player-memory-prod-2026-10-08.md` still stands in full, and this adds what landed
on main on 2026-10-09. Every step that changes prod needs the operator's go, one by one.

## 0. Prod as of 2026-10-09 ~11:15 (read-only, checked over ssh)

- Every service on `sha-0f63257` (api, compose-api, operator-api, mcp, digest-scheduler, obs,
  learning-app). Nothing from the 2026-10-08 handover has been deployed yet.
- Corpus **2.7.17**, migrations through `0023`; `0024`–`0027` pending.
- main is at `2ca8dac75` (or later). No new migrations since `0027`.

## 1. Order (unchanged from 2026-10-08)

1. **1.0.3** on TestFlight and Play internal, and confirmed installable on a tester's Play page —
   the operator and the mobile session do this once main is green.
2. Deploy main — only after 1.
3. `0024` → `0025` → `0026` → `0027`, strict id order, each with the operator's OK. Commands,
   expected dry-run output and undo: 2026-10-08 handover §0, §2, §2b.
4. Set the released version to **1.0.3** in the operator viewer.
5. The operator tells testers to update.

The homelab delivery-template change (§3 below) is a separate deploy by the homelab session. The two
can go in **either order**: the template falls back to the first episode when the app sends no
`open_url`, and the app's new field is optional in the schema.

## 2. What landed on main on 2026-10-09 — checks after the deploy

Server (reaches prod with the deploy):

| Commit | Change | Check after deploy |
|---|---|---|
| `dad37ea5a` | The new-episodes push payload carries `open_url`: the episode when it announces one, `/#whats-new` when several. | The next new-episodes envelope: `GET /internal/outbox/pending?channel=push` with the prod `INTERNAL_OUTBOX_TOKEN` (read-only; it only lists) shows `payload.open_url`. |
| `12212f83f` | Every email's episode summary is sent whole, no 240-char cut. | The next new-episodes / digest email shows the full summary, no trailing "…". |
| `fb70a9def` | A highlight spanning two transcript segments is no longer "anchor drifted" (re-anchoring joined segments with "" instead of " "). | `GET /api/app/highlights` for an account with a multi-segment highlight: `anchor_status` is `anchored`. Verified on the local stack 2026-10-09 for an existing highlight, re-anchored on read. |
| `6480cffbf` | Entity search sees re-enriched storylines and themes without an api restart. | After the next enrichment run, a new storyline/theme is findable in search without restarting. |
| `93100ec3d` (#2301) | ASR hardening on **fresh** transcriptions only (invented-line removal, untranscribed-speech recovery, punctuation repair), language refusal remembered, language badge. | Existing transcripts are untouched. The badge shows only when the corpus spans more than one language. The repair of the 121 already-broken English prod episodes is an **operator-scheduled** follow-up (`docs/wip/POST-DEPLOY-MULTILINGUAL-2026-10-05.md`) — not part of this deploy. |
| `6db4e1e31` | Test only (every served cache states its invalidation). | — |

Re-measure after the deploy as in the 2026-10-08 handover §2c (`/your-week`, `/related`).

## 3. Not prod's: the homelab template

`agentic-ai-homelab` `6483321`: `templates/podcast/push/new-episodes.v1.json.j2` reads
`payload.open_url`. Deployed by the homelab session on the mini. Until it is, a multi-episode push
still opens the first episode, as today.

## 4. App-side (ships in the 1.0.3 store build, NOT the deploy)

For the operator's device check once testers are on 1.0.3 — nothing for prod to do:

- A push about several episodes opens Home scrolled to **What's new**; one episode opens it. Native
  push taps did nothing before 1.0.3 (no handler existed).
- Dictation (Android): the mic turns off when the recogniser dies; the episode the mic paused
  resumes; tapping the mic to stop keeps the last word.
- The bell refreshes on every return to the foreground; boards/covers/"Your trends"/Your Week
  stale-surface fixes; mini-player without heart/board; no "Play all" on a board; Open on a
  highlight note shows Episode notes over the player.

## 5. Open, not part of this deploy

- Android black flashes / player showing through while dictating (operator's phone and a Pixel 8):
  **not reproduced** on the emulator. Next evidence: a screen recording plus Settings → Copy debug
  info, taken right after it happens.
- A separate Android dev app id (`app.closelistening.player.dev`, like iOS) needs that package
  registered in Firebase project `closelistening-39437` first; Android push is on.
- The local e2e build loads `https://analytics.closelistening.app/script.js`; whether those page
  views land in the real analytics has not been checked.
