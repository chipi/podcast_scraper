# Fixture audio (`v1/`, `v2/`, **`v3/`**)

The synthetic MP3s every offline test plays, transcribes or diarizes. Generated from the
transcripts in [`../transcripts/`](../transcripts) by text-to-speech — **no real podcast audio is
committed anywhere in this repo**, and nothing here needs the network.

> **Read this first.** This folder is **versioned**, and only one version is current. The current
> set is named in [`../FIXTURES_VERSION`](../FIXTURES_VERSION) — today **`v3`**. A bare
> `tests/fixtures/audio/*.mp3` path is wrong: the files live one level down, under the version.
> Analysing `v1/` or `v2/` means analysing a dead set.

## Which version, and what is in it

| Version | Files | Size | Pairs with |
| --- | --- | --- | --- |
| `v1/` | 32 | 312 MB | superseded — legacy transcripts `../transcripts/v1/` |
| `v2/` | 32 | 210 MB | superseded — `../transcripts/v2/` |
| **`v3/`** | **46** | **83 MB** | **current** — `../transcripts/v3/`, and the app corpus [`../app-validation-corpus/v3`](../app-validation-corpus/README.md) |

`v3` is both larger in file count and smaller on disk than its predecessors: it covers more
episodes (4 per show across `p01`–`p09`, plus multi-episode and fast variants) at a lower bitrate,
because nothing downstream needs fidelity — only speech that a transcriber and diarizer can process
deterministically.

## Naming — the file name IS the mapping

```text
v3/p05_e03.mp3      show p05, episode 3   → transcript ../transcripts/v3/p05_e03.txt
v3/p01_e01_fast.mp3 "fast" variant        → ~1-minute episode for quick E2E arms
v3/p01_multi_e01.mp3 multi-episode set    → feed fixtures that need several episodes from one show
```

Episode ids are stable and shared across the whole fixture family: the same `p05_e03` names the
transcript, the audio, the RTTM diarization reference, the ground-truth sidecar, and the episode in
the app validation corpus. If you have an episode id from anywhere, you already have its audio path.

Show ids map to hosts (`p01` Maya, `p02` Ethan, `p03` Rina, `p04` Leo, `p05` Nora, `p07`–`p09`
Alex Morgan); `p06` is the edge-case show. The canonical list lives in
[`../scripts/transcripts_to_mp3.py`](../scripts/transcripts_to_mp3.py).

## How a client actually gets these bytes

The files are on disk; they are **not** served by the app or the API. Two mock hosts exist, both of
which simulate a real podcast host (RSS + episodes + audio enclosures):

| Host | Command / wiring | Serves | Reachable from |
| --- | --- | --- | --- |
| pytest E2E mock | `make serve-e2e-mock` (port `18765`, override `E2E_MOCK_PORT`) | `/audio/<episode_id>.mp3`, RSS, transcripts, and LLM API stubs | host loopback — so a browser on your machine can fetch it |
| stack-test sidecar | [`docker/mock-feeds/`](../../../docker/mock-feeds/README.md), wired by `compose/docker-compose.stack-test.yml` | same fixtures over nginx | the compose network (`http://mock-feeds/...`) |

Both resolve the version automatically: the E2E server versions its `audio` and `transcripts`
subdirectories from `FIXTURES_VERSION`, so `/audio/p05_e03.mp3` serves `v3/p05_e03.mp3` today and
follows the file when the version bumps.

**RSS fixtures reference audio relatively** — `url="/audio/p01_e01_fast.mp3"` — so the host is
supplied by whichever mock is serving the feed. Follow that convention rather than hard-coding a
port into a fixture.

## Regenerating

```bash
cd tests/fixtures/scripts
./generate_audio.sh                          # all transcripts of the current version
./generate_audio.sh ../transcripts/v3/p07_e01.txt   # one episode
```

`transcripts_to_mp3.py` renders each speaker with a **fixed voice — one voice per person across the
entire corpus**, so a voice identifies an identity. That rule is canonical (`FIXTURES_SPEC.md`) and
enforced by `tests/integration/eval/test_voice_assignment.py`; a deviation fails CI. Output version
mirrors the input version (`transcripts/v2/…` → `audio/v2/…`) unless `--output-version` says
otherwise.

After regenerating **anything**, refresh the ground-truth sidecars — they carry `audio_sha256`, so
audio cannot drift from its declared contents unnoticed:

```bash
python tests/fixtures/scripts/make_groundtruth.py           # rewrite sidecars
python tests/fixtures/scripts/make_groundtruth.py --check   # verify, fails on drift
```

## Who reads these files

Transcription and diarization evals (`scripts/eval/…`), the pytest E2E suite via
`tests/e2e/fixtures/e2e_http_server.py`, fixture-mapping integration tests, and the stack-test
pipeline runs. The **browser** E2E suites do not read them from disk — they need a served URL, which
is what the mock hosts above are for.

## Related

- [`../README.md`](../README.md) — the whole fixture family: RSS, transcripts, audio, conventions
- [`../app-validation-corpus/README.md`](../app-validation-corpus/README.md) — the committed app corpus these episodes appear in
- [`../../../docs/guides/E2E_TESTING_GUIDE.md`](../../../docs/guides/E2E_TESTING_GUIDE.md) — how the suites and mock hosts fit together
- [`../transcripts/v3/README.md`](../transcripts/v3/README.md) — what each v3 episode contains (speakers, failure modes)
