# `tests/` — what lives where, and which command runs it

Start here before adding or debugging a test. This is a map, not a manual: each row points at the
guide that explains the tier properly.

Everything here is **offline by default**. No suite needs the public internet, cloud credentials, or
real podcast audio — feeds, transcripts and audio are all committed fixtures served by a local mock
host. If a test seems to need the network, that is a bug in the test.

## The tiers

| Directory | Files | What it proves | Run it | Guide |
| --- | --- | --- | --- | --- |
| [`unit/`](unit) | ~596 | One module, no I/O, fast | `make test-unit` | [UNIT_TESTING_GUIDE](../docs/guides/UNIT_TESTING_GUIDE.md) |
| [`integration/`](integration) | ~285 | Modules together — providers, server routes, workflow steps | `make test-integration` (`-fast` for the critical path) | — |
| [`e2e/`](e2e) | ~60 | The **real pipeline** end to end against a mock feed server, `--disable-socket` | `make test-e2e` | [E2E_TESTING_GUIDE](../docs/guides/E2E_TESTING_GUIDE.md) · [`e2e/README.md`](e2e/README.md) |
| [`stack-test/`](stack-test) | ~12 | The **shipped containers** together via compose — API, viewer, pipeline-in-docker | `make stack-test-ml` | [`stack-test/README.md`](stack-test/README.md) |
| [`analytical/`](analytical), [`e2e_observability/`](e2e_observability) | few | Metrics/o11y assertions over produced artifacts | part of the suites above | — |
| [`fixtures/`](fixtures) | — | **Not tests.** The data everything above runs on | — | [`fixtures/README.md`](fixtures/README.md) |

Browser tests live outside this tree, next to their apps — see [`../web/README.md`](../web/README.md).

**Slowness is designed, not accidental.** `make test-e2e` is ~12 min locally and ~25 min on CI
because it runs the real pipeline. [`e2e/README.md`](e2e/README.md) exists specifically so nobody
kills a healthy run believing it hung.

## Where the data comes from

Every fixture is committed and versioned. **This is the part people miss**, so it is spelled out:

| What | Path | Note |
| --- | --- | --- |
| RSS feeds | `fixtures/rss/*.xml` | Enclosure URLs are **relative** (`/audio/<id>.mp3`) — the mock host supplies the origin |
| Transcripts | `fixtures/transcripts/<version>/` | **Versioned** |
| Audio | `fixtures/audio/<version>/` | **Versioned** — [`fixtures/audio/README.md`](fixtures/audio/README.md) |
| App corpus (episodes, GI/KG, search index) | `fixtures/app-validation-corpus/v3/` | What the API boots against — [README](fixtures/app-validation-corpus/README.md) |
| Current version | [`fixtures/FIXTURES_VERSION`](fixtures/FIXTURES_VERSION) | **Read this before assuming any fixture path** |

> `transcripts/` and `audio/` are versioned (`v1`, `v2`, `v3`). A bare `fixtures/audio/*.mp3` path
> is wrong — and worse, `v1`/`v2` still exist, so a wrong path silently analyses a dead set.

Two mock hosts serve those fixtures as a real podcast host would (RSS + episodes + audio):

- **`make serve-e2e-mock`** — host loopback, port `18765`, serves `/audio/<episode_id>.mp3`, RSS,
  transcripts, plus LLM API stubs. Used by the pytest E2E suite; reachable from a browser.
- **[`../docker/mock-feeds/`](../docker/mock-feeds/README.md)** — the same fixtures over nginx on
  the compose network, for `make stack-test-*`.

## Before you conclude a fixture is missing

Search from the **repo root**, and read the `README.md` of the tree you are in *and its parent*.
Every asset these suites need already exists; the trees are versioned and split across
`fixtures/audio`, `fixtures/transcripts`, `fixtures/rss` and `fixtures/app-validation-corpus`, so a
scoped search finds nothing while the file sits one directory away.

On 2026-08-13 an agent searched only `fixtures/app-validation-corpus/v3`, found no `.mp3`, concluded
the repo had no fixture audio, and hand-wrote an MP3 encoder to synthesise some. `fixtures/audio/v3/`
had real audio for all 36 corpus episodes. This paragraph is cheaper than that afternoon.

## Adding a test

1. **Pick the lowest tier that can prove it.** A unit test that fails in 200 ms beats an E2E that
   fails in 12 minutes. The E2E tier is for the pipeline actually running end to end.
2. **Use the fixtures.** Do not generate audio, transcripts or feeds, and do not mock what the mock
   host already serves. If a fixture cannot express the state you need, extend the fixture — that is
   a better outcome than a per-test stub, which is invisible to everyone else.
3. **Check the fixture version** before writing a path.
4. Browser test? Read the surface map for that app first — the selectors are a contract, and a unit
   check fails the build if they drift.
