# TODO — remove what only pre-1.0.3 clients need

Operator, 2026-10-08: every API change in the 1.0.3 arc is an **enhancement**. Old clients (1.0.2)
keep working against the new server and the new client works against both. Once no client older
than 1.0.3 is in use, delete the compatibility paths below. Do not delete them before that. Check
the released-version floor in the operator viewer and the app-version mix in the access logs first.

## Server: responses that exist only for clients that do not page

| Endpoint | Old behaviour kept | New behaviour | Delete when |
|---|---|---|---|
| `GET /persons/{id}/card`, `/topics/{id}/card`, `/storylines/{id}`, `/themes/{id}`, `/orgs/{id}/card` | no `episodes_offset` / `episodes_limit` → the WHOLE episode list | paged when the client sends them | 1.0.2 gone: make the page the only shape, with a default `limit` |
| `GET /topics/{id}/perspectives`, storyline and theme perspectives | no `speakers_offset` / `per_speaker` → every speaker, every take | paged when sent | same |
| `GET /favorites`, `/highlights`, `/notes`, `/collections`, `/collections/{id}`, `/resurfacing`, `/playback`, `/queue` | no paging params → the whole list | paged + server-side search / filter / sort when sent | same |

## Client: fallbacks for a server that predates the batch route

- `getEpisodesBatch` (`web/learning-player/src/services/api.ts`) falls back to one
  `GET /episodes/{slug}` per slug when `/episodes/batch` answers 404. That is only for the window
  where 1.0.3 is in the store and the server is not yet deployed. Delete the fallback after the
  deploy is confirmed.

## Not deletable

- `GET /episodes/{slug}` itself: the player and the episode page use it for ONE episode. The batch
  route is for lists only.
