# DGX verify — minimal pyinfra drift detection (RFC-089 / ADR-098)

Read-only assertions over the DGX Spark. Scope is deliberately small:
checks on the manual P0 steps (Tailscale, Ollama, NVIDIA driver) plus the
baseline Ollama model list — including `nomic-embed-text` for embeddings
(ADR-098 supersedes the deleted shim).

- **`make dgx-verify`** — runs assertions. Exits non-zero on drift.

Post-ADR-098 there is no convergent half — the FastAPI embedding shim was
dropped in favour of Ollama-served `nomic-embed-text`. If future convergent
work appears (additional services on DGX), re-introduce a `deploy.py` +
`make dgx-converge` target.

Runs from the operator laptop, over SSH on the tailnet. Nothing here
touches the public internet.

---

## One-time SSH setup

You already have a root account on DGX. Make a dedicated ed25519 key so
this key isn't reused for anything else, and add it to DGX:

```bash
# On the laptop:
ssh-keygen -t ed25519 -f ~/.ssh/dgx_ed25519 -C "dgx-converge $(date +%Y-%m-%d)"

# Copy to DGX (use Tailscale MagicDNS hostname):
ssh-copy-id -i ~/.ssh/dgx_ed25519.pub root@your-dgx.tailnet.ts.net
# If ssh-copy-id isn't installed, manual fallback:
#   cat ~/.ssh/dgx_ed25519.pub | ssh root@<host> 'cat >> ~/.ssh/authorized_keys'

# Smoke-test it:
ssh -i ~/.ssh/dgx_ed25519 root@your-dgx.tailnet.ts.net 'echo ok && uname -a'
```

If you already made a key but aren't sure it works, run the smoke-test
above. Output `ok` + a Linux kernel line means it's good.

**Recommended:** add a `Host` block to `~/.ssh/config` so you don't have
to retype the host / key path:

```sshconfig
Host dgx
    HostName your-dgx.tailnet.ts.net
    User root
    IdentityFile ~/.ssh/dgx_ed25519
    IdentitiesOnly yes
```

Then `ssh dgx` should just work. pyinfra picks this up automatically.

---

## Env vars

| Var                | Default              | Purpose                                                |
| ------------------ | -------------------- | ------------------------------------------------------ |
| `DGX_TAILNET_FQDN` | _required_           | MagicDNS hostname (e.g. `your-dgx.tailnet.ts.net`)     |
| `DGX_SSH_USER`     | `root`               | SSH user on DGX                                        |
| `DGX_SSH_KEY`      | `~/.ssh/dgx_ed25519` | Path to private key                                    |
| `DGX_SSH_PORT`     | `22`                 | SSH port                                               |

These are the same vars `scripts/ops/resolve_dgx_tailnet_host.sh` uses
elsewhere in the repo — set them once in `~/.zshrc` and forget.

---

## Install pyinfra (operator laptop, one-time)

Isolated venv keeps pyinfra off the main project deps:

```bash
python3 -m venv infra/dgx/converge/.venv
. infra/dgx/converge/.venv/bin/activate
pip install -r infra/dgx/converge/requirements.txt
```

The Makefile targets activate this venv automatically.

---

## What gets enforced

**Assertions (always — fail fast on drift):**

1. `nvidia-smi` reports GB10 + ~128GB unified memory.
2. `tailscale status` shows DGX `Online`, `DNSName == DGX_TAILNET_FQDN`, and tag `dgx-llm-host`.
3. `ollama` systemd unit is active + enabled.
4. Ollama API responds on `:11434/api/tags`.
5. Each baseline model from `DGX_MODEL_CATALOG.md` is present — including `nomic-embed-text` (ADR-098). Missing → `::warning::` line, not a failure (pulls are overnight work).

**Out of scope (operator-manual per #810):**

- Tailscale install + operator-account login.
- Ollama install.
- NVIDIA driver install.
- `ollama pull <model>` (slow, overnight — run by hand).
- ACL updates (lives in `tailscale/policy.hujson`, applied via `make infra-apply`).

---

## Cron / drift alerts

Out of scope for this iteration. Once `make dgx-verify` has been clean
for a week or two, wire it into a nightly GHA workflow on the tailnet
runner (P3 in RFC-089). The Makefile target is the right boundary for
that — no further changes needed to this directory.
