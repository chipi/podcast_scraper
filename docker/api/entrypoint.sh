#!/bin/sh
# API container entrypoint. Runs as root only long enough to prepare the
# corpus volume and adjust Docker socket group membership (#666 review #5),
# then drops to the unprivileged ``podcast`` user via ``su-exec`` / ``setpriv``.
#
# When a prebuilt image is run with its default ``USER podcast`` (non-root),
# every step below that requires root is skipped — ``mkdir -p`` and group
# adjustments succeed only on first-boot ownership fixups.
set -e

OUT="${PODCAST_STACK_OUTPUT_DIR:-/app/output}"
mkdir -p "$OUT"

# Only root can chown / add group memberships. When the image is launched as
# the ``podcast`` user directly (dev compose, CI smoke), skip these steps and
# assume the volume is already writable.
if [ "$(id -u)" = "0" ]; then
    # Take ownership of the corpus volume on first boot. Named volumes are
    # root-owned until an entrypoint fixes permissions — otherwise the API
    # refuses to write.
    chown -R podcast:podcast "$OUT" 2>/dev/null || true

    # Docker-backed jobs (PODCAST_PIPELINE_EXEC_MODE=docker) require the
    # ``podcast`` user to read/write the host Docker socket. The socket's
    # group GID is host-dependent; mirror it into the container so
    # ``docker compose`` does not need root.
    SOCK="/var/run/docker.sock"
    if [ -S "$SOCK" ]; then
        SOCK_GID="$(stat -c '%g' "$SOCK" 2>/dev/null || echo "")"
        if [ -n "$SOCK_GID" ] && [ "$SOCK_GID" != "0" ]; then
            if ! getent group "$SOCK_GID" >/dev/null 2>&1; then
                groupadd -g "$SOCK_GID" docker-host >/dev/null 2>&1 || true
            fi
            HOST_GROUP="$(getent group "$SOCK_GID" | cut -d: -f1)"
            if [ -n "$HOST_GROUP" ]; then
                usermod -aG "$HOST_GROUP" podcast >/dev/null 2>&1 || true
            fi
        fi
    fi

    # Drop privileges. ``setpriv`` is in util-linux; fall back to ``su``.
    if command -v setpriv >/dev/null 2>&1; then
        exec setpriv --reuid=podcast --regid=podcast --init-groups "$@"
    else
        exec su -s /bin/sh -c 'exec "$0" "$@"' podcast -- "$@"
    fi
fi

exec "$@"
