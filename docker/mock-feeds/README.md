# mock-feeds (stack-test sidecar)

Tiny Nginx container that serves the project's RSS + audio fixtures
to the pipeline container during `make stack-test-*`. Wired in by
[`compose/docker-compose.stack-test.yml`](../../compose/docker-compose.stack-test.yml)
as the `mock-feeds` service; the pipeline container reaches it as
`http://mock-feeds/feed.xml` on the compose network. Not exposed to
the host.

## Mount layout

| Container path | Mounted from | Served as |
| -- | -- | -- |
| `/etc/nginx/conf.d/default.conf` | `docker/mock-feeds/nginx.conf` | nginx config (this file) |
| `/usr/share/nginx/html/feed.xml` | `tests/fixtures/rss/p01_mtb.xml` | `GET /feed.xml` (RSS feed) |
| `/usr/share/nginx/html/audio/` | `tests/fixtures/audio/` | `GET /audio/<name>.mp3` |

The single served RSS file (`feed.xml`) is enough for the airgapped
and cloud-thin pipeline runs that share this overlay. The full UI
flow spec ([`tests/stack-test/stack-jobs-flow.spec.ts`](../../tests/stack-test/stack-jobs-flow.spec.ts))
references two distinct RSS URLs and a seeded `feeds.spec.yaml` —
that wiring is a follow-up; the spec is currently `testIgnore`'d in
[`tests/stack-test/playwright.config.ts`](../../tests/stack-test/playwright.config.ts)
until it lands.

## Why this isn't done with `make serve-e2e-mock`

[`scripts/run_e2e_mock_server.py`](../../scripts/run_e2e_mock_server.py)
serves the same fixtures over HTTP for the **pytest** end-to-end
suite. Stack-test runs the pipeline inside Docker, so we need the
mock host on the **compose network** — not the developer's loopback.
A separate Nginx container with bind-mounted fixtures keeps the
stack-test self-contained (no extra processes on the host) and
matches CI parity (the GitHub Actions runner sees the same compose
graph).

## Healthcheck

`GET /healthz` returns `200 ok` so the pipeline can chain
`depends_on: condition: service_healthy` on the mock-feeds container
without racing on the first feed fetch.
