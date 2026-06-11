# `podcast-speaches:0.1.0` — derived speaches image

A thin downstream of `ghcr.io/speaches-ai/speaches:latest-cuda` that
applies the **#968 Thread B temperature-fallback patch** at build
time. Everything else (model loading, FastAPI routes, env-var
contract) is unchanged.

## Why this exists

Upstream speaches forwards the API `temperature` parameter as a
scalar float to `faster_whisper.WhisperModel.transcribe(...)`. A
scalar `0.0` disables the documented
`(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)` compression-ratio / logprob fallback
schedule — the rescue path that prevents autoregressive runaway loops
on Whisper large-v3. Without it the model emits empty transcripts or
hallucinated runs (we observed 4/5 empty + 1 hallucinated 11k-word
episode on the v2 fixtures).

This is the same bug class our sibling openai-whisper service fixes
in `infra/dgx/whisper-server/app.py` (#929, commit `6df41b31`). The
fix is identical in spirit: expand scalar 0.0 into the fallback
tuple at the call site.

## What the build does

`Dockerfile` runs three `sed` substitutions on
`/home/ubuntu/speaches/src/speaches/executors/whisper.py` at build
time, replacing each `temperature=request.temperature` with
`temperature=((0.0, 0.2, 0.4, 0.6, 0.8, 1.0) if request.temperature
== 0.0 else request.temperature)`.

A grep guard fails the build if exactly three substitutions don't
apply — protects against silent drift if speaches upstream renames
or relocates the pattern.

## Build / deploy

```bash
docker build -t podcast-speaches:0.1.0 infra/dgx/faster-whisper-server/
```

Wired into prod via `infra/dgx/converge/deploy.py` — the
`faster-whisper` service section uses `image: podcast-speaches:0.1.0`
instead of the upstream tag. Compose otherwise unchanged.

## Bump procedure (when speaches updates)

The base image is pinned only by tag (`:latest-cuda`). To bump:

1. `docker pull ghcr.io/speaches-ai/speaches:latest-cuda` on the
   DGX (or wherever the build runs).
2. Re-run the build. If the sed pattern still applies 3×, the
   patch carries forward and the new build ships clean.
3. If the build aborts with `FATAL: #968 Thread B patch expected
   3 hits`, speaches has refactored. Re-read
   `src/speaches/executors/whisper.py` upstream, update the sed
   pattern in `Dockerfile`, repeat.

## Future state

The right long-term fix is an upstream speaches PR mirroring our
patch — file `#968` referenced. Once that lands and reaches the
`:latest-cuda` tag, this Dockerfile becomes a no-op and can be
deleted (or kept as a paranoia safety net).

## References

- #968 — speaches deeper quality + speed investigation (this image
  is Thread B)
- `docs/wip/968-SPEACHES-RESEARCH.md` — research report with the
  citation chain
- `infra/dgx/whisper-server/app.py` — sibling openai-whisper fix
  that motivated this investigation
- Upstream: <https://github.com/speaches-ai/speaches>
