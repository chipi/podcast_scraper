# #1273 TODO 1 — large-v3 int8-vs-float16 serving test (BLOCKED on DGX SSH)

**Status:** BLOCKED — cannot deploy the parallel float16 service (SSH to the DGX is down).
**Decision:** v2.3 closes WITHOUT this. TODO 2 (the actionable fix) already shipped; TODO 1 is a
pure-curiosity quality follow-up and is not on the v2.3 critical path.

## What this was going to answer

The 2026-07-23 human-GT bake-off ranked **large-v3 last (16.3% WER)** — beaten by its own turbo
distillation (13.5%), which is backwards and points at an **int8 serving regression** on the DGX
speaches service (GB10). The int8 decision doc
(`infra/dgx/speaches/decisions/2026-06-15-compute-type-int8.md`) explicitly picked int8 on **speed**
and states it was *"Not a quality eval… does NOT mean WER-validated."* This is that missing
validation: re-transcribe the cached bake-off episodes with large-v3 at int8 vs float16 and compare
WER against the human ground truth.

## What's already DONE (TODO 2 — the fix that mattered)

The coverage-failover no longer routes to large-v3 (the least-accurate model). It now fails over to
**MOSS** (2nd-best accuracy 12.5%, still DGX-local) — commit `1fd5b7a0`. Registry-governed on the 3
DGX presets + the 2 reprocess YAMLs; new `transcription_coverage_failover_provider` config;
factory builds the MOSS provider; tests + drift-check green. So large-v3 is off the critical path;
the int8 question no longer gates anything.

## The harness is READY (committed `cdad8097`) — just needs the DGX

- `infra/dgx/experiments/docker-compose.1273-largev3-fp16.yml` — a TEMPORARY parallel speaches
  container serving large-v3 at **float16 on :8005** (prod int8 :8000 untouched). Plain
  `docker compose`, NOT the sudo pyinfra converge; cleanup is one `down`.
- `scripts/investigate/1273_largev3_compute_wer.py` — re-transcribes the cached human-GT episodes
  large-v3 @int8 (:8000) vs @float16 (:8005) + turbo anchor, WER via the bake-off's exact
  normalization (comparable to 16.3%), prints a verdict, writes
  `docs/wip/EVAL_1273_largev3_int8_vs_fp16.json`.

## The blockers (why it's parked)

1. **DGX SSH is down** — can't `docker compose up` the :8005 float16 container on the DGX.
2. **Reachability** — even once :8005 is up, this Mac is tailnet-only: it reaches `dgx-llm-1:8000`
   (200) but NOT the DGX LAN IP `192.168.1.111` (000), and `:8005` is not in the `dgx-llm-host`
   tailnet ACL (only `:8000` is). So `:8005` needs adding to the ACL, OR the scorer runs on a
   DGX-LAN host.
3. **GPU headroom** — float16 large-v3 is ~3 GB on top of the live int8 service on the GB10; confirm
   before loading.

## To run it once SSH is restored

```bash
# on the DGX:
docker compose -f infra/dgx/experiments/docker-compose.1273-largev3-fp16.yml up -d   # wait: /v1/models healthy
# add :8005 to the dgx-llm-host tailnet ACL (like :8000), OR run the scorer on a LAN host
# then from this repo:
.venv/bin/python scripts/investigate/1273_largev3_compute_wer.py --lan dgx-llm-1 --episodes 10
# cleanup:
docker compose -f infra/dgx/experiments/docker-compose.1273-largev3-fp16.yml down
```

**Verdict interpretation:** float16 materially better (Δ>2%) → int8 serving regression confirmed
(and large-v3-fp16 could be reconsidered as a failover); no gain → large-v3 genuinely worst, MOSS
was the right failover, close #1273.
