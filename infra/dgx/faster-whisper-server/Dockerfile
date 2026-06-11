# Derived speaches/faster-whisper image with the #968 Thread B
# temperature-fallback patch baked in.
#
# Background:
#   speaches' `routers/stt.py` declares temperature as a scalar float
#   defaulting to 0.0, forwarded unchanged into
#   `faster_whisper.WhisperModel.transcribe(...)`. faster-whisper
#   wraps a scalar into a single-element list, which DISABLES the
#   documented (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) compression-ratio /
#   logprob fallback. Whisper large-v3 with deterministic
#   temperature=0.0 is known to enter repetitive non-speech-token
#   loops → either empty transcripts (post-processor filters non-
#   speech tokens) or 11k-word hallucinations. We observed both
#   failure modes on v2 episodes (4/5 empty + 1 hallucination).
#
#   This is the SAME bug class our sibling openai-whisper service
#   already fixes in `infra/dgx/whisper-server/app.py` for the
#   openai-whisper stack (#929, commit 6df41b31). Different repo,
#   same mechanism.
#
# What this image changes:
#   Three call sites in
#   `/home/ubuntu/speaches/src/speaches/executors/whisper.py`
#   are patched to expand a scalar 0.0 into the tuple
#   (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) before calling
#   `whisper_model.transcribe(...)`. Anything else (caller passes
#   a non-zero scalar, or already passes a tuple) passes through
#   unchanged.
#
#   The TranscriptionRequest pydantic model stays unchanged
#   (temperature: float = 0.0). All expansion is at the executor
#   call sites — this avoids touching the request schema and keeps
#   API compatibility with the upstream speaches client.
#
# Idempotency check:
#   The build fails fast if the sed pattern doesn't apply exactly
#   three times. If speaches upstream renames or restructures, the
#   build error tells us at deploy time — not at first-bad-output
#   time. Cheap protection against silent drift.
#
# Build:
#   docker build -t podcast-speaches:0.1.0 infra/dgx/faster-whisper-server/
#
# Run via deploy.py:
#   IMAGE switched from ghcr.io/speaches-ai/speaches:latest-cuda to
#   the locally-built podcast-speaches:0.1.0. Compose otherwise
#   identical.

FROM ghcr.io/speaches-ai/speaches:latest-cuda

# #968 Thread B — temperature fallback patch.
# Expand scalar temperature=0.0 into the documented faster-whisper
# fallback tuple (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) at the executor call
# sites. Engages the compression-ratio / logprob rescue path that
# prevents autoregressive runaway loops on Whisper large-v3.
USER root
RUN set -eux ; \
    target=/home/ubuntu/speaches/src/speaches/executors/whisper.py ; \
    test -f "$target" ; \
    sed -i 's|temperature=request.temperature|temperature=((0.0, 0.2, 0.4, 0.6, 0.8, 1.0) if request.temperature == 0.0 else request.temperature)|g' "$target" ; \
    n=$(grep -c "0.2, 0.4, 0.6, 0.8, 1.0" "$target") ; \
    if [ "$n" != "3" ]; then \
        echo "FATAL: #968 Thread B patch expected 3 hits, got $n. Speaches upstream may have changed shape; rebuild abort." >&2 ; \
        exit 1 ; \
    fi
