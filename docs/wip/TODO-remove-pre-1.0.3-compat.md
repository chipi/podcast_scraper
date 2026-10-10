# TODO — remove what only pre-1.0.3 clients need

**Done 2026-10-10 (1.0.4 branch):** the client's fallbacks (batch / refs 404, local paging) and the
server's unpaged answers for entity-card episodes (default 20), `/playback` (50), `/favorites` (50),
`/resurfacing` (20 episodes) and `/collections/{id}` (50). **Kept**, because 1.0.3 reads them: the
perspectives routes unpaged (1.0.3's first load asks for every speaker — `fetchPage({ perSpeaker })`),
`/podcasts`, per-episode/per-target `/highlights` and `/notes`, and the favourites / delete write
answers (see the audit below).

**Part (b) done 2026-10-10 (1.0.4 branch):** 1.0.4 accepts the leaner write answers.
`favoriteRefsOf` reads a refs-only `{ items }` answer as well as the full list, and
`deleteHighlight` / `deleteNote` treat an answer with no body (204, or an empty 200) as success. The server still sends the whole-list
answers; switch them (refs-only favourites, 204 deletes) only once no 1.0.3 is in use.

Operator, 2026-10-08: every API change in the 1.0.3 arc is an **enhancement**. Old clients (1.0.2)
keep working against the new server and the new client works against both. Once no client older
than 1.0.3 is in use, delete the compatibility paths below. Do not delete them before that. Check
the released-version floor in the operator viewer and the app-version mix in the access logs first.

## Server: responses that exist only for clients that do not page

| Endpoint | Old behaviour kept | New behaviour | Delete when |
| --- | --- | --- | --- |
| `GET /persons/{id}/card`, `/topics/{id}/card`, `/storylines/{id}`, `/themes/{id}`, `/orgs/{id}/card` | no `episodes_offset` / `episodes_limit` → the WHOLE episode list | paged when the client sends them | 1.0.2 gone: make the page the only shape, with a default `limit` |
| `GET /topics/{id}/perspectives`, storyline and theme perspectives | no `speakers_offset` / `per_speaker` → every speaker, every take | paged when sent | same |
| `GET /favorites`, `/highlights`, `/notes`, `/collections/{id}`, `/resurfacing`, `/playback`, `/podcasts` | no `limit` → the whole list | paged + server-side search / filter / sort when `limit` is sent | same |

## Server: write responses that still answer with the WHOLE list

1.0.2 reads its new state from these answers.

- `PUT` / `DELETE` / `PATCH /favorites…` answer with every favourite, each episode hydrated.
  1.0.3 still reads them, but only for the identities (`favoriteRefsOf`). Once 1.0.2 is gone,
  answer with the refs (`/favorites/refs` shape) and drop the hydration.
- `DELETE /highlights/{id}` answers with every remaining highlight, each re-anchored; `DELETE
  /notes/{id}` with every remaining note. 1.0.3 ignores both answers (it removes the row it has).
  Once 1.0.2 is gone, answer 204.

## Client: fallbacks for a server that predates the batch route

- `getFavoritesPage`, `getFavoriteRefs`, `getHighlightsPage`, `getNotesPage` page (or derive)
  locally when the server ignores the paging parameters (no `total` in the answer, or 404).
- `getEpisodesBatch` (`web/learning-player/src/services/api.ts`) falls back to one
  `GET /episodes/{slug}` per slug when `/episodes/batch` answers 404. That is only for the window
  where 1.0.3 is in the store and the server is not yet deployed. Delete the fallback after the
  deploy is confirmed.

## Audit 2026-10-09 (1.0.4 branch): what 1.0.3 itself still needs

Prod runs `sha-2ca8dac` (checked inside `player-api-1`: `/episodes/batch`, `/favorites/refs`,
paged `/favorites` and `/playback` answer 401 unauthenticated, an unknown route 404 — the routes
exist). 1.0.3 is the oldest client in use from 1.0.4 on. Read against 1.0.3's own code
(`web/learning-player/src/services/api.ts` at `2ca8dac75`), the lists above are partly WRONG:

**NOT deletable while 1.0.3 is in use (it breaks 1.0.3):**

- Favourites write answers (`PUT`/`DELETE`/`PATCH /favorites…`): 1.0.3 does
  `favoriteRefsOf(await resp.json())`, which reads `resp.episodes` — a refs-only answer throws.
- `DELETE /highlights/{id}` and `DELETE /notes/{id}` answers: 1.0.3's `deleteHighlight` /
  `deleteNote` call `resp.json()` — a 204 with no body throws, and the delete reads as failed.
  ("1.0.3 ignores both answers" above is true of the store, not of the api call.)
- `GET /podcasts` unpaged: `useCorpusLanguages` (the #2301 language badge) reads the whole catalogue.
- `GET /highlights?episode=` and `GET /notes?target=&target_id=` unpaged: the capture store loads one
  episode's / target's rows with them (`stores/capture.ts`); a default page would cut them.

To drop the write answers later, 1.0.4 must first ACCEPT the new shapes (refs-only, 204), and the
server change waits for the release after everyone is on 1.0.4.

**Deletable now (no 1.0.3 caller), server side:** the unpaged branches of the entity cards
(persons/topics/orgs/themes/storylines — every 1.0.3 call sends `episodes_limit`), the perspectives
routes (`TopicPerspectives.vue` always passes a page — to confirm), `GET /playback` without `limit`
(all three callers page). `GET /favorites` without `limit` and `GET /resurfacing` without `limit`
have no product caller in 1.0.3 — but other clients (live smoke specs, the operator viewer, MCP) are
not checked yet.

**Client side (1.0.4), now that prod has the routes:** the `total === undefined` local-paging
branches and the `/episodes/batch` / `/favorites/refs` 404 fallbacks. Caveat: the `page…Locally`
functions are ALSO the unit tests' fake server (`src/test/apiViaSpies.ts`, 14 spec files), so
removing the fallbacks means moving those functions under `src/test/`, not deleting them.

## Not deletable

- `GET /episodes/{slug}` itself: the player and the episode page use it for ONE episode. The batch
  route is for lists only.
