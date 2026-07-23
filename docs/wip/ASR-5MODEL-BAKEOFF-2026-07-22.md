# 5-model ASR bake-off (2026-07-22) — WIP pointer + incident note

**Canonical report** (numbers, tables, registry decisions):
[docs/guides/eval-reports/EVAL_ASR_5MODEL_BAKEOFF_2026_07.md](../guides/eval-reports/EVAL_ASR_5MODEL_BAKEOFF_2026_07.md).

This WIP file retains only the operational incident note from the session — it is not a second copy
of the results.

## DGX incident note (same session)

Mid-run, MOSS over the tailnet started timing out and SSH went unreachable. This was **not** a box
crash/OOM — the DGX and all services (MOSS, speaches, pyannote) were healthy the whole time; only
the **tailnet path** to `:22`/`:8004`/`:8003` was blocked (ACL / Tailscale-SSH), while `:8000`/`:8001`
worked. MOSS was completed over the **LAN IP** (`192.168.1.111:8004`).

Two diagnostic lessons:

- *Unreachable over one path ≠ service down.* Early "MOSS is down" / "OOM cascade" claims were wrong —
  verify a service over the LAN before declaring it dead.
- After a box restart, SSH stayed down because the break was **Tailscale SSH** (tailnet `:22`), not
  the box's `sshd` (which answered fine on the LAN). Fix is `tailscale up --ssh` + ACL, or add a key
  to the LAN `authorized_keys` — not another reboot.
