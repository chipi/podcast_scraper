# Build a Blackwell-capable speaches image for the DGX GB10 Spark (#948).
#
# Why this exists:
# - The upstream image ``ghcr.io/speaches-ai/speaches:latest-cuda`` bundles
#   a CPU-only ctranslate2 4.5.0 (the tag is misleading — the libctranslate2.so
#   has no CUDA symbols and ``faster_whisper.WhisperModel(..., device="cuda")``
#   raises ``"This CTranslate2 package was not compiled with CUDA support"``).
# - For ~5 days post the 2026-06-05 driver bump to 580.x, transcription
#   silently fell back to multi-threaded CPU on the 20-core Grace ARM, which
#   was fast enough (~10-30× realtime on long episodes) that nobody noticed.
# - On 2026-06-09 a compute-type change + container recreate exposed the
#   gap: same image, slower path → noticeable degradation → #948.
# - This image source-builds ctranslate2 4.8.0 with CUDA 12.8 and PTX for
#   sm_120, which the GB10 (compute_cap=12.1) driver JIT-compiles forward
#   to sm_121 at runtime. Result: real GPU transcription at 36× realtime
#   on a 5-minute clip (8.25 s wall, vs 103 s on CPU).
#
# Validation: see ../speaches-gb10/README.md "Validating the image" section.
# The runtime smoke probe is ``ctranslate2.get_cuda_device_count() == 1``
# inside the running container plus a 5-minute clip transcription
# completing in <15 s (>20× realtime → unambiguously GPU on this hardware).

# ---------------------------------------------------------------------------
# Stage 1: build ctranslate2 with CUDA 12.8
#
# We compile against the CUDA 12.8 developer image because (a) it's the
# first CUDA toolkit nvcc that ships ``-arch=sm_120`` natively, and (b) PTX
# emitted at compute_120 forward-JITs to sm_121 (GB10) via the host driver
# at first kernel launch — no source-level sm_121 entry needed. The build
# itself takes ~3 min on the DGX (20-core Grace) once the apt + git phases
# finish.
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake git ca-certificates \
        build-essential gcc g++ \
        python3.12-dev python3.12-venv python3-pip \
        libopenblas-dev libomp-dev \
        libcudnn9-dev-cuda-12 libcudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone --depth=1 --branch v4.8.0 --recurse-submodules \
        https://github.com/OpenNMT/CTranslate2

WORKDIR /build/CTranslate2
# Arch list:
#   90        — Hopper datacenter (H100 etc.); kept for portability.
#   100       — Blackwell datacenter (B100/B200).
#   120-real  — Blackwell consumer (RTX 50xx etc.). Native sm_120 cubin.
#   120-virtual — PTX. The driver JIT-compiles forward to sm_121 (GB10)
#                 on first launch. This is the bit that makes GB10 work.
RUN mkdir build && cd build && cmake \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DWITH_CUDA=ON \
        -DWITH_CUDNN=ON \
        -DWITH_MKL=OFF \
        -DWITH_OPENBLAS=ON \
        -DCMAKE_CUDA_ARCHITECTURES="90;100;120-real;120-virtual" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_CLI=OFF \
        .. && \
    make -j"$(nproc)" && \
    make install

# Build the Python wheel against the freshly compiled libctranslate2.so so
# the wheel's bundled .so is what we just produced (not a stale download).
WORKDIR /build/CTranslate2/python
RUN python3 -m venv /opt/build-venv && \
    /opt/build-venv/bin/pip install --upgrade pip wheel setuptools build pybind11 && \
    /opt/build-venv/bin/pip install -r install_requirements.txt && \
    CT2_ROOT=/usr/local /opt/build-venv/bin/python setup.py bdist_wheel && \
    ls -la dist/

# ---------------------------------------------------------------------------
# Stage 2: overlay onto the upstream speaches image. Replaces the bundled
# CPU-only ctranslate2 with the CUDA-enabled wheel we just built.
FROM ghcr.io/speaches-ai/speaches:latest-cuda

USER root

# Runtime deps the freshly-built ctranslate2.so needs at import time.
# - libomp5: the rebuilt .so links to OpenMP; the upstream speaches image
#   doesn't have it because their CPU-only build linked against a static
#   OMP that the wheel doesn't ship.
# - libopenblas0: CPU BLAS path inside ctranslate2 (used when device='cpu').
RUN apt-get update && apt-get install -y --no-install-recommends \
        libomp5 \
        libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the built libctranslate2.so. CUDA / cuDNN libs are host-injected via
# nvidia-container-toolkit at container start (not bundled here).
COPY --from=builder /usr/local/lib/libctranslate2.so* /usr/local/lib/
COPY --from=builder /build/CTranslate2/python/dist/*.whl /tmp/

# Replace the bundled CPU-only 4.5.0 with our CUDA 4.8.0 wheel.
#
# We install via pip because speaches' venv is uv-managed and lacks pip
# by default. ``ensurepip`` bootstraps just enough to do the swap.
#
# The final import-check inside the build runs without a GPU attached, so
# ``cuda_device_count`` is expected to be 0 here — that's a build-time
# environment artifact, not a real signal. The runtime probe lives in
# ../speaches-gb10/README.md.
RUN /home/ubuntu/speaches/.venv/bin/python -m ensurepip --upgrade && \
    /home/ubuntu/speaches/.venv/bin/python -m pip install --force-reinstall --no-deps /tmp/*.whl && \
    rm /tmp/*.whl && \
    echo "/usr/local/lib" > /etc/ld.so.conf.d/ctranslate2-cuda.conf && \
    ldconfig && \
    /home/ubuntu/speaches/.venv/bin/python -c \
        "import ctranslate2 as c; print('ctranslate2', c.__version__, 'cuda_device_count:', c.get_cuda_device_count())"

USER ubuntu
