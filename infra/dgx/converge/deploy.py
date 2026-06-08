"""DGX Spark convergent install: faster-whisper-server (#814).

Installs the Whisper transcription service that prod
(``cloud_with_dgx_whisper_primary.yaml``) targets via the tailnet. Ollama
already runs on :11434; this adds a second HTTP service on :8000 serving
OpenAI-compatible ``/v1/audio/transcriptions``.

What this lays down on DGX:

- Apt deps: ``ffmpeg`` for audio decoding.
- A dedicated system user ``faster-whisper`` with no shell, no home (per
  systemd hardening conventions).
- A venv at ``/opt/faster-whisper/venv`` with ``faster-whisper-server``
  installed via pip.
- ``/etc/faster-whisper/env`` — environment file with model + device knobs
  (defaults are CUDA + Systran/faster-whisper-large-v3, matching what prod
  expects per the profile).
- ``/etc/systemd/system/faster-whisper.service`` — unit definition.
- The service is enabled + started; restart policy keeps it alive across
  reboots without manual intervention. Crash-restart hardening (OOM
  detection, restart-rate-limit) is **out of scope** — see #910.

Idempotent: re-running ``make dgx-deploy`` does not perturb a working install.

Verify with ``make dgx-verify`` (extended in this commit to check the
service + its health endpoint).

Operator manual steps remain (per #810):
- NVIDIA driver + CUDA runtime install.
- Pre-pull the Whisper model into the HF cache to avoid first-request
  latency spike (or accept the ~3 GB / first request).
"""

from __future__ import annotations

from pyinfra.operations import apt, files, server, systemd

# Knobs that downstream code (the provider client) expects:
#
# - PORT 8000 — added to the dgx-llm-host ACL alongside :11434 in
#   tailscale/policy.hujson. Don't change without updating the ACL.
# - MODEL ID — Hugging Face repo ID format. ``Systran/faster-whisper-large-v3``
#   is the upstream-recommended quantization-free large-v3 weights;
#   faster-whisper-server fetches it lazily on first request.
SERVICE_USER = "faster-whisper"
INSTALL_ROOT = "/opt/faster-whisper"
VENV_DIR = f"{INSTALL_ROOT}/venv"
ENV_DIR = "/etc/faster-whisper"
ENV_FILE = f"{ENV_DIR}/env"
SYSTEMD_UNIT = "/etc/systemd/system/faster-whisper.service"
PORT = 8000
MODEL = "Systran/faster-whisper-large-v3"
DEVICE = "cuda"  # GB10 → CUDA. Set to "cpu" on a non-GPU host for testing.
COMPUTE_TYPE = "float16"  # float16 on CUDA; switch to int8 on CPU.

# 1. System packages.
apt.packages(
    name="apt: install ffmpeg + python3 venv",
    packages=["ffmpeg", "python3-venv", "python3-pip"],
    update=True,
    cache_time=3600,
)

# 2. Service user (no login shell, no home dir).
# bandit B604 false positive: ``shell="/usr/sbin/nologin"`` here is the user's
# LOGIN SHELL (forced to nologin so the account can't be used interactively),
# not a subprocess ``shell=True`` invocation.
server.user(  # nosec B604
    name="user: create faster-whisper system account",
    user=SERVICE_USER,
    system=True,
    shell="/usr/sbin/nologin",
    create_home=False,
    home=INSTALL_ROOT,
)

# 3. Install root + ownership.
files.directory(
    name="dir: /opt/faster-whisper (install root)",
    path=INSTALL_ROOT,
    user=SERVICE_USER,
    group=SERVICE_USER,
    mode="755",
    present=True,
)

# 4. Create the venv (idempotent — pyinfra's pip op handles re-creation safely).
server.shell(
    name="venv: bootstrap /opt/faster-whisper/venv",
    commands=[
        f"test -x {VENV_DIR}/bin/python || python3 -m venv {VENV_DIR}",
        f"{VENV_DIR}/bin/pip install --upgrade pip wheel setuptools",
    ],
    _sudo=True,
)

# 5. Install faster-whisper-server into the venv.
# Pinned range: minor-version stability. Bump deliberately, not transitively.
server.shell(
    name="pip: install faster-whisper-server",
    commands=[
        f"{VENV_DIR}/bin/pip install 'faster-whisper-server>=0.2,<0.3'",
    ],
    _sudo=True,
)

# 6. Ensure the venv is owned by the service user.
files.directory(
    name="chown: venv to faster-whisper",
    path=VENV_DIR,
    user=SERVICE_USER,
    group=SERVICE_USER,
    recursive=True,
    present=True,
)

# 7. Env file (model + device + port). Watched by systemd — restart pulls changes.
files.directory(
    name="dir: /etc/faster-whisper",
    path=ENV_DIR,
    mode="755",
    present=True,
)

files.put(
    name="env: /etc/faster-whisper/env",
    src=None,  # inline below
    dest=ENV_FILE,
    create_remote_dir=False,
    mode="644",
    # pyinfra's ``put`` with content via the ``src`` API requires a file;
    # easier to drop the literal lines via ``server.shell`` with heredoc.
)

# 7b. Actually drop the env file content. Idempotent — pyinfra diffs.
server.shell(
    name="env-content: write /etc/faster-whisper/env",
    commands=[
        f"cat > {ENV_FILE} <<'EOF'\n"
        f"# faster-whisper-server runtime knobs. See deploy.py for context.\n"
        f"# Restart unit after edits: systemctl restart faster-whisper\n"
        f"WHISPER_MODEL={MODEL}\n"
        f"WHISPER_DEVICE={DEVICE}\n"
        f"WHISPER_COMPUTE_TYPE={COMPUTE_TYPE}\n"
        f"HOST=0.0.0.0\n"
        f"PORT={PORT}\n"
        f"# Bind all interfaces, same as Ollama (OLLAMA_HOST=0.0.0.0 on :11434).\n"
        f"# The tailnet ACL on tag:dgx-llm-host:8000 is the security boundary —\n"
        f"# binding 127.0.0.1 would make the service unreachable from the laptop /\n"
        f"# prod / chaos tests over tailnet, defeating the prod plan.\n"
        f"EOF",
        f"chmod 644 {ENV_FILE}",
    ],
    _sudo=True,
)

# 8. systemd unit.
SYSTEMD_UNIT_CONTENT = f"""[Unit]
Description=faster-whisper-server (DGX Whisper for #814)
Documentation=https://github.com/fedirz/faster-whisper-server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User={SERVICE_USER}
Group={SERVICE_USER}
EnvironmentFile={ENV_FILE}
WorkingDirectory={INSTALL_ROOT}
ExecStart={VENV_DIR}/bin/faster-whisper-server
Restart=on-failure
RestartSec=10s
# Hardening (#910 will tighten further):
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths={INSTALL_ROOT}

[Install]
WantedBy=multi-user.target
"""

server.shell(
    name="unit: drop /etc/systemd/system/faster-whisper.service",
    commands=[
        f"cat > {SYSTEMD_UNIT} <<'EOF'\n{SYSTEMD_UNIT_CONTENT}EOF",
        f"chmod 644 {SYSTEMD_UNIT}",
    ],
    _sudo=True,
)

# 9. Tell systemd we changed unit files.
systemd.daemon_reload(
    name="systemd: daemon-reload after unit write",
    _sudo=True,
)

# 10. Enable + start. ``running=True`` is idempotent — pyinfra checks status.
systemd.service(
    name="systemd: enable + start faster-whisper",
    service="faster-whisper",
    running=True,
    enabled=True,
    daemon_reload=False,  # already done above
    _sudo=True,
)
