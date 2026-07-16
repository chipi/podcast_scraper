# Goal-1 — merge-to-live roadmap

**Living doc — evolve as decisions land.** Higher-level sequencing from "the
infra PR merges to main" through "orrery live on the shared edge." The *phase
detail* lives in [`GOAL1-GO-LIVE-PLAN.md`](GOAL1-GO-LIVE-PLAN.md); this doc is
the from-here roadmap + the open-decisions tracker. Don't duplicate phase steps
here — link to them.

- **Last updated:** 2026-07-16
- **Branch:** `production` = `origin/main` + 3 commits (infra programme ADR-114–118
  · go-live plan · drill fix #1088). Re-rebased onto main (incl. #1195). Not pushed.
- **Related:** [`GOAL1-GO-LIVE-PLAN.md`](GOAL1-GO-LIVE-PLAN.md) ·
  [`GOAL1-ORRERY-CRITICAL-PATH.md`](GOAL1-ORRERY-CRITICAL-PATH.md) ·
  [threat model + pre-public gate](../security/THREAT_MODEL.md) · ADR-114…118.

---

## Guiding invariant — merging to main applies NOTHING live

Merge = infra-as-code lands on the trunk. Nothing reaches the running VPS until
we deliberately apply it. Three independent reasons:

1. `deploy-prod.yml` is **manual-dispatch only** — no `workflow_run` auto-trigger,
   so a merge never deploys to the box.
2. `hcloud_server.prod` has `lifecycle { ignore_changes = [user_data] }` — even a
   `tofu apply` won't reprovision or re-run cloud-init.
3. The edge/hardening/o11y/CF config is all cloud-init + gated TF; none of it is
   wired to fire on push.

**Corollary:** go-live is a separate, gated, operator-driven operation — not a
side effect of the merge.

---

## The sequence (each step gates the next)

| # | Step | Owner | Gate before proceeding |
|---|------|-------|------------------------|
| A | **harden audit** → apply safe fixes, surface decisions | 🤝 | findings triaged |
| B | **push** `production` (`--force-with-lease` after currency check) | 🧑 approves | push authorized |
| C | **PR → main**, green CI | 🤝 | all required checks green |
| D | **merge to main** (fires main CI cascade: stack-test rebuild + push) | 🧑 | — |
| E | **DR drill GREEN end-to-end** — the trust anchor (see below) | 🧑 runs | deploy+restore+smoke all green |
| F | **Go-live Phase 0→7** — per `GOAL1-GO-LIVE-PLAN.md` | 🤝 | each phase's own gate |

### Why DR-drill-green is a hard gate (step E)
We fixed the drill's deploy image-SHA resolution (#1088), but `restore` / `smoke`
/ `stack-playwright` in that drill have **never actually run** — they were skipped
behind the deploy failure. Go-live **Phase 0** depends on a *verified* restore;
that guarantee only exists once the drill is green end-to-end. The drill
provisions a throwaway drill VPS → **operator-run, not agent-triggered.**

---

## Open decisions (tracker — resolve, then strike through)

| # | Decision | Options | Recommendation | Status |
|---|----------|---------|----------------|--------|
| D-1 | **Apply path** to converge the live box | A imperative-once · B rebuild · C blue-green | **A** — no rebuild, zero data-loss risk | open |
| D-2 | **Corpus location** — data-loss lever | volume (`volume_size_gb>0`, delete-protected) vs boot disk | confirm current value before any rebuild option | open |
| D-3 | **DR-drill-green as hard gate** before prod go-live | yes / no | **yes** | open |
| D-4 | **Branch hygiene** after merge | retire `production` (follow-ups off main) · keep as standing deploy branch | deploys read main+sha → not needed as persistent | open |
| D-5 | `production` → main **merge style** | squash · merge-commit | squash (single clean feature landing) | open |

---

## What the merge triggers (cost awareness)

- Main CI cascade: `stack-test` rebuilds + pushes `:main` + `:sha-<7>` images.
- No deploy, no infra apply (see invariant above).

## My immediate to-dos (once the apply path is locked)

1. Write `apply-edge.sh` (go-live Phase 1.1) — idempotent, mirrors cloud-init,
   dry-run mode.
2. Pre-stage the `tofu plan` review checklist for the firewall-open step
   (must be in-place `~`, never a server `-/+` replace).
3. One-paste Phase-0 backup + verify-restore command sequence.
