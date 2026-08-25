# Retrospective — Stack-test publish: 6h QEMU-arm64 hang → amd64-only matrix split

Status: fix shipped to `main` in `def2dd503` (before this doc); this document is the
retrospective #1800 asked for, landed 2026-08-25 as part of the incident-hardening PR.


The `Stack test` workflow's `publish` job hung for the full **6-hour** GitHub Actions job cap and was killed, on multiple `main` pushes. Root cause: it built four images **sequentially in one job**, **multi-arch (amd64 + arm64 via QEMU emulation)**, and the `learning-app` arm64 build — a Vite/npm build under emulation — wedged. This retrospectively documents the incident, the analysis, and the fix (shipped in `def2dd503`).

## Incident

- Run [`32349843022`](https://github.com/chipi/podcast_scraper/actions/runs/32349843022) (main `c8b1d2ab7`): the `stack-test` job **passed** (18 min); the separate `publish` job ran **6h0m44s** and was cancelled by GitHub's 6h `max execution time`.
- It hung on step **"Build + assert + push (learning-app)"** (arm64 via QEMU). Because the four images built sequentially in one shell, `pipeline-llm` never built and the whole job burned to the ceiling.
- Recurring, not a one-off: the same `publish` cancellation appears on earlier commits (e.g. run `32323883428`, before the incident push), each with `stack-test` green + `publish` cancelled. **Not** a concurrency-supersede — the annotation was the 6h execution-time cap and the duration was exactly 6h.

## Root cause

1. **QEMU arm64 emulation of a JS build.** Emulating an `npm install` + `vite build` on arm64 on an amd64 runner is pathologically slow and wedged.
2. **One monolithic job.** Four builds in sequence in a single shell → any one hanging black-boxes the others; no per-image isolation.
3. **No per-image timeout.** Default 6h job cap meant a hang wasted 6h of runner time before failing.

## Why arm64 was pointless here (the key insight)

The published `stack-*` / `learning-app` images are consumed **only by the prod VPS, which is amd64** (Hetzner `cx43`, see `infra/terraform/variables.tf`). The operator's Apple-Silicon Mac builds these images **natively from source** for local work (`compose/docker-compose.stack.yml` has `build:`; `make stack-test-build`), so it **never pulls arm64 from GHCR**. Nothing consumes the arm64 variant. The `#712` arm64 was added speculatively to "unblock a future Hetzner CAX (ARM) target" that never happened.

Arch-to-consumer matrix:

| Context | Arch | How it gets images | Needs GHCR arm64? |
|---|---|---|---|
| Prod VPS (`cx43`) | amd64 | pulls `ghcr.io/…:main` | No |
| Operator Mac (Apple Silicon) | arm64 | builds from source locally | No — never pulls arm64 |
| GitHub CI publish | built both | pushes multi-arch | it was the only arm64 *producer*, consumed by nobody |

## The fix (shipped in `def2dd503`)

`.github/workflows/stack-test.yml` `publish` job:

1. **Matrix split** — one leg per image (`api`, `viewer`, `learning-app`, `pipeline-llm`), `fail-fast: false` (a wedged leg cannot cancel its siblings — the other three still publish), `timeout-minutes: 45` (a hang fails fast, not at 6h).
2. **amd64-only** — dropped `setup-qemu-action` and changed `--platform linux/amd64,linux/arm64` → `--platform linux/amd64`. No emulation → the hang cannot recur at the root, and builds run in parallel in minutes.
3. **`verify-manifests` job** — runs after all legs, asserts each pushed `:sha-<short>` manifest exists and lists `linux/amd64`.

## `obs-image.yml` deliberately left multi-arch

By contrast, `obs-image.yml` (the `podcast-obs` observability MCP control-plane image) is **kept multi-arch** — its consumer, the **homelab MCP endpoint, IS arm64**, and it's a tiny `python-slim` image so its QEMU arm64 build is seconds, not a Vite build. The principle: **match published arch to the actual consumer**, don't blanket-drop.

## Future note (preserved in the workflow header + summary)

If prod ever migrates to a Hetzner **CAX (ARM)** target, re-introduce arm64 by adding a **native `ubuntu-24.04-arm` runner** leg (GA since 2025) — **never** QEMU emulation, which is what caused this hang.

## Follow-ups (not done here)

- If branch protection lists a required check literally named `Stack test / publish`, the matrix renames it to `publish (api)` etc. — the required-check name may need updating (main is pushed with admin bypass today, so it isn't blocking).
- Pre-existing `private/var/folders/...` pytest-tmp leak in `test_pipeline_offload_evict_e2e` / `test_offload_e2e` (writes to a relative path) — unrelated to this change, worth a separate cleanup.

