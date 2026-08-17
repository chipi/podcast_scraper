# Handover — provision the audio archive Storage Box (#1199 / H1)

**For:** whoever runs the infra apply from the laptop that holds the age key + Hetzner token.
**Written:** 2026-08-15. **Status of the code side:** done and merged into PR #1661.

Everything below was verified against the live repo and the live prod box, not recalled.
Anything I could **not** verify is called out in §6 — treat that section as unproven.

---

## 1. What this does, in one line

Turns on the remote audio archive: a Hetzner Storage Box that the pipeline writes raw episode
audio to, so future reprocessing reads from the archive instead of re-downloading from
publishers (which fails once an episode rolls off the feed).

## 2. Why it is not already on

The feature shipped complete under #1199 — runbook, terraform, rclone backend — and was never
enabled. Verified:

| Where | State (checked 2026-08-15) |
| --- | --- |
| `infra/terraform/storage_box.tf:12` | `count = var.audio_storage_box_type != "" ? 1 : 0` |
| `infra/terraform/variables.tf:96` | `audio_storage_box_type` default `""` → **provisions nothing** |
| prod `viewer_operator.yaml` (via `GET /api/operator-config`) | `audio_storage_backend` **not set** |
| same | `audio_cache_in_corpus: true` — the cheap fix **is** applied |

So since 2026-08-13 new episodes' audio survives on the **prod disk**
(`<corpus>/.podcast_scraper/audio-cache`). What is missing is the durable off-box copy.

## 3. Prerequisites (the reason this is a handover)

I could not do this myself. Two hard blockers, both on the machine running the apply:

1. **The sops age private key.** State is `infra/terraform/terraform.tfstate.enc`, encrypted to
   exactly one recipient: `age1yerjgvcm5trle7xl7jkmt53dp290y2fr9rt5zjpljzeyv0nkr5qqffhm00`.
   Without it `sops -d` fails, so no plan and no apply.
2. **`HCLOUD_TOKEN`** for the Hetzner API.

Also needed: `tofu` and `rclone` on PATH. Backend is **local** (`backend.tf`:
`path = "terraform.tfstate"`), so run from `infra/terraform/` with the decrypted state in place
and re-encrypt after — follow whatever wrapper the repo normally uses for that; do not
hand-roll it.

## 4. The apply

Pick the size. `bx11` = 1 TB, ~EUR 3.20/month, and is the documented choice; `bx21`/`bx31`/`bx41`
are 5/10/20 TB. The validation rejects anything else.

```hcl
# infra/terraform/<your>.tfvars
audio_storage_box_type     = "bx11"
audio_storage_box_location = "fsn1"   # or nbg1 / hel1
```

The password is a **separate secret variable**, never committed:

```bash
export TF_VAR_audio_storage_box_password='<32+ random chars>'
```

> **Make it high-entropy and mean it.** `storage_box.tf` sets
> `reachable_externally = true`, which puts this SFTP endpoint on the public internet with
> **password-only auth**. The VPS firewall does not protect it — a Storage Box is a separate
> Hetzner product with no server-side IP allowlist. The file's own security note (review
> 2026-07-17) says ≥32 random chars and suggests moving to key-only auth later.

Then plan, read the plan, apply. The resource also sets `delete_protection` (from
`enable_delete_protection`) and a weekly snapshot plan (Sunday 03:00 UTC, keep 4).

## 5. Wiring it up after the box exists

Two halves, per `docs/recipes/prod-audio-archive.md` §2.

**a. Profile / operator YAML** — this is the switch that turns the backend on:

```yaml
audio_storage_backend: remote
audio_remote_rclone_remote: hetznerbox
audio_remote_base_path: podcast-audio-archive
```

**b. rclone credentials in the prod host `.env`.** rclone reads `RCLONE_CONFIG_<NAME>_*` from
the environment natively — no volume mount, no key file, no compose change:

```dotenv
RCLONE_CONFIG_HETZNERBOX_TYPE=sftp
RCLONE_CONFIG_HETZNERBOX_HOST=<audio_storage_box_server output>
RCLONE_CONFIG_HETZNERBOX_USER=<audio_storage_box_username output>
RCLONE_CONFIG_HETZNERBOX_PASS=<rclone obscure '<the password>'>
```

Store the **obscured** value as GH secret `PROD_RCLONE_STORAGEBOX_PASS`. The host/user come from
terraform outputs `audio_storage_box_server` / `audio_storage_box_username` — both marked
`sensitive`, so read them deliberately and keep them out of CI logs.

The remote name must match in both halves: profile `audio_remote_rclone_remote: hetznerbox`
↔ env prefix `RCLONE_CONFIG_HETZNERBOX_*`.

## 6. Not verified — do not assume these

- **I never ran `tofu plan`.** No age key, no token. The terraform is read but unexercised;
  the plan may surface provider-version or state-drift issues I cannot predict.
- **`tofu`/`rclone` were absent from the box I work on.** I installed them into `~/bin` there
  (rclone v1.75.0, OpenTofu v1.12.5) — that says nothing about the laptop running the apply.
- **`infra-apply.yml` will NOT provision this as written.** It passes `TF_VAR_hcloud_token`,
  `TF_VAR_ssh_public_key`, etc. from secrets but has **no `TF_VAR_audio_storage_box_type`**, so
  dispatching it applies the `""` default and creates nothing. If you want the CI route, that
  line has to be added first (I could not: this PAT has no `workflow` scope, and
  `actions/variables` returns 403).
- **Whether `enable_delete_protection` is currently true for prod** — I did not read the
  effective tfvars.

## 7. After it is live — the part people get wrong

**Enabling the archive does not populate it retroactively.** It stores audio for episodes
ingested from that point on. And the corpus repair (#1655) will **not** fill it either — a
repair reuses existing transcripts and downloads nothing, which is why it costs $5.85 instead
of $113.

The only thing that recovers history is `archive backfill`, which is **new in PR #1661**:

```bash
# always preview first — per-feed report + size floor, fetches nothing
podcast-scraper archive backfill --corpus /app/output --rclone-remote hetznerbox --dry-run

# then run; idempotent and resumable
podcast-scraper archive backfill --corpus /app/output --rclone-remote hetznerbox
```

Two things about its results that are easy to misread:

- **`rolled_off` is a normal outcome, not a failure.** Publishers truncate feeds at very
  different depths; those episodes are unrecoverable and the command still exits `0`. A
  non-zero exit means `fetch_failed` (retryable) only.
- **Recovered audio is not the original bytes.** Dynamic-ad feeds re-encode per request, so a
  backfilled file is not the file that produced the existing transcript. Every recovery is
  stamped in `<corpus>/.podcast_scraper/audio-archive-provenance.jsonl` with
  `byte_identical_to_transcribed_audio: false`. Fine to re-transcribe from; **wrong** for any
  WER-vs-original comparison.

The ~473 episodes ingested before 2026-08-13 lost their audio to a cache written into an
ephemeral container layer (`/app/.cache/audio`, never mounted, destroyed by `--rm` every run).
No backfill recovers those originals — at best it re-scrapes what publishers still serve.

## 8. Ordering

None of this blocks PR #1661 or the deploy — it is infra plus config, no Python. The ordering
that matters is **before the next bulk ingest**, so new episodes land in the archive rather
than joining the unrecoverable pile.

## References

- `docs/recipes/prod-audio-archive.md` — the full runbook (§6 documents `archive backfill`)
- `infra/terraform/storage_box.tf`, `variables.tf:93-115`, `outputs.tf:39-51`
- [#1199](https://github.com/chipi/podcast_scraper/issues/1199) storage backend (closed),
  [#1631](https://github.com/chipi/podcast_scraper/issues/1631) backfill,
  [#947](https://github.com/chipi/podcast_scraper/issues/947) the audio archive
- `docs/wip/INCREMENTAL-ROLLOUT-FOLLOWUPS-2026-08-11.md` §H1 / §H1a — how the audio was lost
