#!/usr/bin/env bash
# Dispatch ``gpu-mode-swap.sh`` on the DGX, either directly or over SSH.
#
# The autoresearch sweep's per-phase ``prep_cmd`` calls into this to swap
# which vLLM owns the GB10 GPU. It has to work from two very different
# callers:
#
#   * Developer laptop (default) — sweep runs on macOS; the DGX is
#     reachable via ``ssh dgx-llm-1``. We SSH out to the DGX and run
#     ``~/bin/gpu-mode-swap.sh`` there.
#
#   * DGX self-hosted GHA runner (``DGX_LOCAL_MODE=1``) — sweep runs
#     directly on the DGX; SSH'ing back to ourselves is wasteful and
#     may fail on gha-runner permissions. Just exec locally.
#
# Envs:
#   DGX_LOCAL_MODE           1 = local exec, 0 (or unset) = SSH out
#   GPU_MODE_START_TIMEOUT   Forwarded to gpu-mode-swap.sh (defaults 600)
#   GPU_MODE_SWAP_BIN        Explicit path to gpu-mode-swap.sh (for the
#                            gha-runner where $HOME is /opt/actions-runner
#                            not the operator's home). Defaults to
#                            ``$HOME/bin/gpu-mode-swap.sh``.
#
# Args are passed through verbatim to gpu-mode-swap.sh, e.g.:
#   scripts/ops/dgx_gpu_mode.sh judging a
#   scripts/ops/dgx_gpu_mode.sh idle
#   scripts/ops/dgx_gpu_mode.sh status

set -euo pipefail

TIMEOUT="${GPU_MODE_START_TIMEOUT:-600}"

if [ "${DGX_LOCAL_MODE:-0}" = "1" ]; then
    export GPU_MODE_START_TIMEOUT="$TIMEOUT"
    SWAP_BIN="${GPU_MODE_SWAP_BIN:-${HOME}/bin/gpu-mode-swap.sh}"
    if [ ! -x "$SWAP_BIN" ]; then
        echo "dgx_gpu_mode: $SWAP_BIN not executable" >&2
        echo "  set GPU_MODE_SWAP_BIN to the operator's gpu-mode-swap.sh path" >&2
        exit 1
    fi
    exec "$SWAP_BIN" "$@"
fi

# Off-host caller — pack the whole remote command into one SSH argument
# (single-quote the inline env-var assignment so /bin/sh on DGX parses it
# as a leading assignment, not as data). ``$*`` splits args on IFS which
# is fine here — every gpu-mode-swap.sh arg is a single unquoted token
# (mode + optional sub-arg like ``a``/``b``).
exec ssh dgx-llm-1 "GPU_MODE_START_TIMEOUT=$TIMEOUT ~/bin/gpu-mode-swap.sh $*"
