#!/bin/sh
set -e
# Ensure corpus mount exists before uvicorn (serve refuses missing output_dir).
OUT="${PODCAST_STACK_OUTPUT_DIR:-/app/output}"
mkdir -p "$OUT"
exec "$@"
