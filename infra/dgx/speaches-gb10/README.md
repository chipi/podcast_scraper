# DGX speaches-gb10 — Blackwell-capable Whisper server (#948)

Source-built variant of speaches with **real GPU support on the DGX Spark GB10**.

The upstream `ghcr.io/speaches-ai/speaches:latest-cuda` image bundles a
**CPU-only** ctranslate2 4.5.0 (the `-cuda` tag is misleading). This
directory holds the Dockerfile that rebuilds ctranslate2 from source
against CUDA 12.8 with `sm_120` PTX, then layers the result on top of the
upstream image. Result: real GPU transcription on GB10 at ~36× realtime.

## Why this exists (the 2026-06-09 incident)

| When | What |
| --- | --- |
| 2026-06-05 10:51 | apt unattended-upgrade bumps nvidia-container-toolkit 1.17.8 → 1.19.1 and the NVIDIA driver to 580.159.03. Operator reboots into kernel 6.17 + driver 580. |
| 2026-06-05 11:06 | First post-bump boot. GB10 now reports `compute_cap=12.1` (sm_121). nvidia-container-toolkit 1.19.1 prefers CDI mode but no `/etc/cdi/nvidia.yaml` exists → silently falls back to a degraded legacy injection. |
| 2026-06-05+ → 2026-06-09 | Whisper container *appears* to work — it returns transcriptions at 10-30× realtime on long episodes. This is the 20-core Grace ARM CPU running multi-threaded ctranslate2; the `-cuda` image's ctranslate2 is actually CPU-only. faster_whisper silently falls back to CPU when ctranslate2 has no CUDA support. Nobody notices. |
| 2026-06-09 23:39 | Operator edits `/opt/faster-whisper/docker-compose.yml` to try `WHISPER__COMPUTE_TYPE=int8` (chasing a perceived fp16 issue), runs `docker compose up -d`. The container recreate brings up a fresh ct2 process under the toolkit-1.19.1 fallback path; int8 on Grace ARM crawls (no vectorized SIMD kernel); transcription drops to ~1.4× realtime. |
| 2026-06-09 ~00:30 | Issue #948 filed describing the symptom. Diagnosis attributes it to a "host CUDA runtime wedge" — close, but incomplete. The actual story is the discovery that the image was never GPU-accelerated. |
| 2026-06-10 09:36-11:30 | Full investigation + the fix in this directory: generated `/etc/cdi/nvidia.yaml`, rebuilt ctranslate2 from source with `-DCMAKE_CUDA_ARCHITECTURES="90;100;120-real;120-virtual"`, layered onto the speaches image. Validation: `cuda_device_count == 1`, 5-min clip transcribes in 8.25 s (36.4× realtime). |

The deeper lesson: a `-cuda` Docker tag is a label, not a contract. The
build-time CUDA configuration of the bundled wheels is what actually
matters. **Trust `nm -D libctranslate2.so | grep -i cuda` over tag names.**

## What the Dockerfile builds

A two-stage build:

1. **Builder** (`nvidia/cuda:12.8.0-devel-ubuntu24.04`)
   - Apt-installs cmake + git + cuDNN dev + OpenBLAS + OpenMP dev.
   - Clones ctranslate2 v4.8.0 source.
   - `cmake -DCMAKE_CUDA_ARCHITECTURES="90;100;120-real;120-virtual"`.
   - `make -j$(nproc) && make install` → `/usr/local/lib/libctranslate2.so*`.
   - Builds the Python wheel against the freshly compiled .so.

2. **Runtime** (`ghcr.io/speaches-ai/speaches:latest-cuda`)
   - Apt-installs `libomp5` + `libopenblas0` (runtime deps the new .so needs).
   - Copies the built libctranslate2.so + wheel from the builder.
   - `ensurepip` + `pip install --force-reinstall` the custom wheel,
     replacing the CPU-only bundled version.
   - Writes `/etc/ld.so.conf.d/ctranslate2-cuda.conf` so the loader can find
     the .so at runtime.

The result is tagged `speaches:latest-cuda-gb10` on the DGX. The
`docker-compose.yml` written by `infra/dgx/converge/deploy.py` references
that tag.

## Why these exact build flags

| Flag | Reason |
| --- | --- |
| `-arch=sm_90` (Hopper) | Portability — image runs on Hopper GPUs if you ever move it. |
| `-arch=sm_100` (Blackwell datacenter) | Same; B100/B200 forward compat. |
| `-arch=sm_120-real` (Blackwell consumer) | Native cubin for sm_120 (RTX 50xx). |
| `-arch=sm_120-virtual` (PTX) | **The critical one.** PTX emitted at compute_120 forward-JITs to sm_121 (GB10) on first kernel launch. Without this, ctranslate2 can't enumerate the GB10 and `get_cuda_device_count()` silently returns 0. |
| `-DWITH_CUDNN=ON` | faster-whisper-large-v3's attention kernels expect cuDNN. Without it the build technically succeeds but inference crashes on first transcription. |
| `-DWITH_MKL=OFF` | aarch64 has no MKL. OpenBLAS is the CPU BLAS path. |

## Validating the image

Run on the DGX after `make dgx-deploy`:

```bash
# 1. ctranslate2 can enumerate the GPU (the truth signal — bypass nvidia-smi
#    which is unreliable on GB10's unified-memory architecture).
sudo docker exec faster-whisper \
    /home/ubuntu/speaches/.venv/bin/python -c \
    "import ctranslate2 as c; print('cuda_device_count:', c.get_cuda_device_count())"
# Expected: cuda_device_count: 1
# Failure shape: cuda_device_count: 0 → build didn't pick up CUDA or the
# host CDI spec is missing. Re-run with the install steps in deploy.py.

# 2. Supported compute types include fp16 and bf16 (not just float32).
sudo docker exec faster-whisper \
    /home/ubuntu/speaches/.venv/bin/python -c \
    "import ctranslate2 as c; print(c.get_supported_compute_types('cuda', 0))"
# Expected: {'int8', 'int8_bfloat16', 'int8_float16', 'float16',
#            'float32', 'bfloat16', 'int8_float32'}
# Failure shape: {'float32'} only → CUDA arch flags didn't include
# Blackwell PTX, only CPU + a degraded CUDA path got built.

# 3. End-to-end latency on a 5-minute clip should be ~8 s (36× realtime).
#    CPU baseline on the same hardware was ~100 s (3× realtime).
sudo docker exec faster-whisper bash -c '
  [ -f /tmp/probe5m.wav ] || ffmpeg -f lavfi -i "sine=frequency=440:duration=300" \
      -ac 1 -ar 16000 /tmp/probe5m.wav -y 2>/dev/null
  # Warm-up call to load the model.
  curl -s -X POST http://localhost:8000/v1/audio/transcriptions \
      -F file=@/tmp/probe5m.wav -F model=Systran/faster-whisper-large-v3 \
      -F response_format=json -o /dev/null
  time curl -s -X POST http://localhost:8000/v1/audio/transcriptions \
      -F file=@/tmp/probe5m.wav -F model=Systran/faster-whisper-large-v3 \
      -F response_format=json -o /dev/null
'
# Expected: real ~8 s
# Failure shape: real >60 s → silently fell back to CPU. Check #1 first.
```

For the CDI spec sanity check (host-side):

```bash
# CDI spec should exist + enumerate the GB10. Generated by deploy.py.
ls /etc/cdi/nvidia.yaml && sudo nvidia-ctk cdi list | head -5
# Expected:
#   /etc/cdi/nvidia.yaml
#   Found 3 CDI devices
#   nvidia.com/gpu=0
#   nvidia.com/gpu=GPU-<uuid>
#   nvidia.com/gpu=all
```

## Build is reproducible from scratch

```bash
# On the DGX (or any CUDA-12.8-capable aarch64 host):
cd /opt/faster-whisper/build
sudo docker build -t speaches:latest-cuda-gb10 .
```

The pyinfra recipe at `infra/dgx/converge/deploy.py` does this automatically
during `make dgx-deploy`. The build context (`/opt/faster-whisper/build/`)
is populated with the Dockerfile from this directory on every deploy run.

## Operational notes

- **Build time on DGX**: ~3 min for the C++ + CUDA compile, plus ~30 s
  for apt + git + Python wheel = total ~4 min including layer setup.
- **Image size**: ~5 GB (upstream speaches:latest-cuda is ~4.7 GB; we
  add ~300 MB for libctranslate2.so with CUDA kernels).
- **Re-pull policy**: `infra/dgx/converge/deploy.py` does NOT pull the
  upstream tag every run (rate limits + idempotency); it rebuilds the
  custom layer only when the Dockerfile content changes (Docker layer
  cache handles the no-op case in <1 s).
- **Upstream image refresh**: when speaches itself releases a meaningful
  new image, bump the `FROM` line in this Dockerfile and re-deploy.
- **GB10 nvidia-smi caveat**: `nvidia-smi --query-gpu=utilization.gpu`
  reports `0%` even while transcription is actively running because
  GB10 uses unified memory; util counters don't reflect compute. Use
  wall-time, not nvidia-smi, to judge whether GPU is in use.

## References

- Incident issue: [#948](https://github.com/chipi/podcast_scraper/issues/948)
- Original speaches deploy precedent: [#814](https://github.com/chipi/podcast_scraper/issues/814) (`infra/dgx/converge/deploy.py`)
- Pyannote-on-DGX sibling pattern: [`infra/dgx/pyannote-server/`](../pyannote-server/) (#926)
- Consumer-side resilience: [#946](https://github.com/chipi/podcast_scraper/issues/946) (duration-scaled timeouts, audio cache — survive any future GPU regressions)
- ctranslate2 v4.8.0: <https://github.com/OpenNMT/CTranslate2/releases/tag/v4.8.0>
