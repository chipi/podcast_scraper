# Prod audio archive — remote storage + reprocessing (#1199)

## Why this exists

Prod stores **no audio** by default: episodes are transcribed, then the media is
discarded, so any reprocess (new diarization models, re-transcription) has to
re-download from the live feed — which fails once an episode rolls off or a
dynamic-ad feed mutates. The #1199 storage backend persists raw audio to a cheap
remote object store (a Hetzner Storage Box), so reprocessing reads from the
archive instead. This recipe wires it up and runs the two read-back flows:

- **A** — pull archived audio to your laptop (`archive pull`).
- **B** — one-off reprocess *in prod* that reads audio from the archive.

## Quick reference

| Action | Command |
| --- | --- |
| Enable remote archive | set `audio_storage_backend: remote` + `audio_remote_rclone_remote` in the prod profile/`.env` |
| Pull audio → laptop | `podcast-scraper archive pull --corpus <dir> --dest ~/audio --rclone-remote hetzner-box [--since 2026-06-01]` |
| One-off reprocess (workflow) | dispatch `reprocess-prod.yml` (`PROD_REPROCESS` confirm) |
| One-off reprocess (manual) | `docker compose … run --rm pipeline-llm python -m podcast_scraper.cli … --reprocess-existing-only --reprocess-source whisper_transcription` |

## 1. Provision the Storage Box with OpenTofu (operator, one-time)

The Storage Box is now infrastructure-as-code (`infra/terraform/storage_box.tf`,
`hcloud_storage_box`). Set the type + a password and apply:

```hcl
# infra/terraform/<your>.tfvars
audio_storage_box_type = "bx11"           # 1 TB, ~€3.20/mo (bx21=5TB, bx31=10TB, bx41=20TB)
# audio_storage_box_location = "fsn1"      # default; or nbg1 / hel1
```
```bash
cd infra/terraform
export TF_VAR_audio_storage_box_password='<a-strong-password>'   # meets Hetzner's policy; keep it
tofu apply        # provisions the box, enables SFTP, weekly snapshot, delete-protected
tofu output audio_storage_box_server     # -> the SFTP host (uXXXXXX.your-storagebox.de)
tofu output audio_storage_box_username   # -> the SFTP username (uXXXXXX)
```

No `rclone config` file is needed — the env-var injection in step 2 *is* the
remote definition. rclone creates the `podcast-audio-archive` base path on first
upload. The password you set in `TF_VAR_audio_storage_box_password` is the same
value you obscure for rclone in step 2.

## 2. Inject the rclone config on prod

Two halves — the **backend selection** goes in the prod **profile** (profiles are
the source of truth in this project, not env overrides), and the **rclone
credentials** go in the host `.env` (rclone reads `RCLONE_CONFIG_<NAME>_*` from
the environment natively, so no new volume mount is required).

**a. Enable the backend in the prod profile** (e.g. `config/profiles/cloud_balanced.yaml`
or the corpus's `viewer_operator.yaml`):

```yaml
audio_storage_backend: remote
audio_remote_rclone_remote: hetznerbox
audio_remote_base_path: podcast-audio-archive
```

**b. Stage the rclone credentials in the host `.env`.** rclone reads
`RCLONE_CONFIG_<NAME>_*` from the environment natively, so there is **no volume
mount and no key file** — SFTP **password** auth keeps the one secret in a single
env var. Obscure the Storage Box password once:

```bash
rclone obscure 'the-storagebox-password'   # -> an obscured string; this is the secret
```

Store that obscured string as GH secret `PROD_RCLONE_STORAGEBOX_PASS`, and have
`deploy-prod.yml` append these lines when it renders the prod `.env` (inert when
the archive is off — only `_PASS` is secret; the rest are plain config):

```dotenv
# #1199 remote audio archive (rclone reads RCLONE_CONFIG_<NAME>_* from env)
RCLONE_CONFIG_HETZNERBOX_TYPE=sftp
RCLONE_CONFIG_HETZNERBOX_HOST=uXXXXXX.your-storagebox.de
RCLONE_CONFIG_HETZNERBOX_USER=uXXXXXX
RCLONE_CONFIG_HETZNERBOX_PASS=<obscured>     # from `rclone obscure`; the only secret
```

Compose already passes `.env` to the container, so the pipeline's `rclone` calls
pick these up — **no compose change**. The remote name (`hetznerbox`) must match
between the profile's `audio_remote_rclone_remote` and the
`RCLONE_CONFIG_HETZNERBOX_*` prefix. The `rclone` binary ships in the pipeline
image (`docker/pipeline/Dockerfile`).

*(SSH-key auth is possible but needs the key mounted read-only into the container
plus a compose volume — avoided here in favour of the single obscured password.)*
**Never** commit the password, obscured or not.

Once set, normal pipeline runs **archive audio on download** automatically
(best-effort, ERROR-logged on failure, never silent). A misconfigured remote
(missing remote name) fails the run loud at config load — by design.

## 3. Flow A — pull audio to your laptop

The archive is keyed by `sha256(guid)`; `archive pull` resolves episode→guid→key
from corpus metadata and downloads with meaningful names.

```bash
# configure the same rclone remote on the laptop first (step 1)
podcast-scraper archive pull \
  --corpus /path/to/corpus \
  --dest ~/podcast-audio \
  --rclone-remote hetzner-box \
  --since 2026-06-01        # or --feed "Planet Money" | --episode <guid> | (default: all)

podcast-scraper archive pull … --dry-run   # preview + estimated size, no download
```

Files land as `~/podcast-audio/<Feed Title>/<NNNN> - <Episode Title>.mp3`,
deduped by guid, skipping ones already present (`--force` to re-pull). Use
`--local-root <dir>` instead of `--rclone-remote` to pull from a local archive.

## 4. Flow B — one-off reprocess in prod (reads from the archive)

Reprocess re-runs episode processing, which fetches each episode's audio from the
archive (no feed re-fetch) before re-diarizing / re-transcribing.

### Via the workflow (auditable)

Dispatch **`reprocess-prod.yml`** (Actions → Reprocess prod corpus), type
`PROD_REPROCESS`, pick `reprocess-source`. It joins the tailnet, SSHes to prod,
and runs the reprocess with `audio_storage_backend=remote`. **Take a corpus
backup first** (`backup-corpus-prod.yml`) — reprocess rewrites artifacts.

### Manual (ad-hoc, on the box)

```bash
ssh deploy@prod-podcast
cd /srv/podcast-scraper
docker compose \
  -f compose/docker-compose.stack.yml \
  -f compose/docker-compose.prod.yml \
  -f compose/docker-compose.vps-prod.yml \
  run --rm pipeline-llm \
  python -m podcast_scraper.cli \
    --config config/profiles/cloud_balanced.yaml \
    --feeds-spec /app/output/feeds.spec.yaml \
    --output-dir /app/output \
    --reprocess-existing-only \
    --reprocess-source whisper_transcription
```

`--reprocess-existing-only` processes only episodes already on disk (matched by
GUID); `--reprocess-source whisper_transcription` re-diarizes the Whisper-sourced
episodes. With the remote backend configured, each episode's audio is pulled from
the Storage Box on demand.

## Verification

- **Archive is populating:** after a pipeline run, look for
  `[#947] audio archive STORE (guid=…) -> … via rclone:hetznerbox:…` in the logs,
  or `rclone size hetzner-box:podcast-audio-archive`.
- **Pull works:** `archive pull … --dry-run` lists episodes + size; a real pull
  writes named files.
- **Reprocess reads remote:** logs show
  `[#947] audio archive HIT (guid=…) via rclone:… (no feed fetch)`.

## Troubleshooting

- **`… needs the 'rclone' binary on PATH`** — the image lacks rclone (rebuild the
  pipeline image) or you're running outside the container.
- **Run fails at config load with "requires audio_remote_rclone_remote"** —
  `audio_storage_backend=remote` but no remote name set. Set it or use `local`.
- **`MISS … not in archive` on pull** — that episode predates the archive (was
  processed before remote was enabled) — its audio was never stored.
- **Uploads ERROR-log but the run continues** — archiving is best-effort per
  episode; a persistent failure means a broken remote/creds. Check
  `rclone lsjson hetzner-box:podcast-audio-archive` from the box.
