# mock-feeds (stack-test sidecar)

Tiny Nginx container that serves the project's RSS / audio / transcript
fixtures to the pipeline container during `make stack-test-*` runs.
Wired in by [`compose/docker-compose.stack-test.yml`](../../compose/docker-compose.stack-test.yml)
as the `mock-feeds` service; the pipeline container reaches it as
`http://mock-feeds/<...>` on the compose network. Not exposed to the
host.

## Mount layout

| Container path | Mounted from | Served as |
| -- | -- | -- |
| `/etc/nginx/conf.d/default.conf` | `docker/mock-feeds/nginx.conf` | nginx config (this file) |
| `/usr/share/nginx/html/p01_fast_with_transcript.xml` | `tests/fixtures/rss/p01_fast_with_transcript.xml` | `GET /p01_fast_with_transcript.xml` (1 episode, transcript-only) |
| `/usr/share/nginx/html/p01_episode_selection.xml` | `tests/fixtures/rss/p01_episode_selection.xml` | `GET /p01_episode_selection.xml` (3 episodes, transcript-only) |
| `/usr/share/nginx/html/audio/` | `tests/fixtures/audio/` | `GET /audio/<name>.mp3` (used when an RSS fixture references an mp3 enclosure) |
| `/usr/share/nginx/html/transcripts/` | `tests/fixtures/transcripts/` | `GET /transcripts/<name>.txt` (transcript-download fast path so Whisper is skipped) |

The two RSS URLs are what `tests/stack-test/stack-jobs-flow.spec.ts`
drives — one is seeded into `<corpus>/feeds.spec.yaml` by
`make stack-test-seed`, the other is added via the Configuration dialog
during the test.

## Why this isn't done with `make serve-e2e-mock`

[`scripts/run_e2e_mock_server.py`](../../scripts/run_e2e_mock_server.py)
serves similar fixtures over HTTP for the **pytest** end-to-end suite
(host loopback). Stack-test runs the pipeline inside Docker, so the
mock host has to be on the **compose network** — not the developer's
loopback. A separate Nginx container with bind-mounted fixtures keeps
stack-test self-contained (no extra processes on the host) and matches
CI parity (the GitHub Actions runner sees the same compose graph).

## Healthcheck

`GET /healthz` returns `200 ok` so the pipeline can chain
`depends_on: condition: service_healthy` on the mock-feeds container
without racing on the first feed fetch.
