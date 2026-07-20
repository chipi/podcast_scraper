# DGX-hosted MOSS-Transcribe-Diarize service (#1177).
#
# Serves OpenMOSS-Team/MOSS-Transcribe-Diarize on :8004, alongside faster-whisper
# (:8000), pyannote (:8001). The pipeline chunks long audio upstream (episode_processor
# + AudioChunker, #1174), so each request is a <=25 min window that fits the model's
# 128k context in one pass — the service never needs to chunk.
#
# BASE IMAGE: nvcr.io/nvidia/vllm:26.05-py3 — the one base verified to run MOSS on GB10
# (sm_121) during the #1174 bake-off: it ships torch 2.12 (CUDA 13, GB10-native) AND
# transformers 5.6 (MOSS requires >=5.0). Its cuFFT is a STUB, which breaks whisper's GPU
# mel-STFT ("cuFFT error 50") — app.py works around it by running feature extraction on
# CPU (cheap). An NGC pytorch base has real cuFFT but ships transformers <5; revisit if a
# base with both lands.
#
# Build:  docker build -t podcast-moss:0.1.0 infra/dgx/moss-server/
# Run:    docker run --rm --gpus all --ipc=host -p 8004:8004 \
#             -v /opt/llm-models:/opt/llm-models --env-file ~/.env podcast-moss:0.1.0
#
# In prod this ships via the compose file the converge deploy drops at
# /opt/moss-server/docker-compose.yml (mirrors the pyannote service).

FROM nvcr.io/nvidia/vllm:26.05-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# The moss_transcribe_diarize package supplies the model's remote-code inference helpers
# (build_transcription_messages / generate_transcription / parse_transcript). Install it
# with --no-deps so its unbounded `torch>=2.8` / `transformers>=5` requirements don't drag
# the GB10-pinned torch 2.12 or transformers 5.6 off the base image; its non-torch deps
# (numpy already present; librosa/soundfile/av for audio decode) are installed explicitly.
# Pin MOSS_MODEL_REVISION at runtime for a hard model-weights pin.
RUN pip install \
        "fastapi>=0.115,<1.0" \
        "uvicorn[standard]>=0.30,<1.0" \
        "python-multipart>=0.0.9" \
        "librosa>=0.11" \
        "soundfile>=0.12" \
        "av>=12" \
        "prometheus-fastapi-instrumentator>=7.0,<8.0" \
 && pip install --no-deps "git+https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git"

COPY app.py /app/app.py
WORKDIR /app

EXPOSE 8004

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8004"]
