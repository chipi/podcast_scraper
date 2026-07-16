# `infra/secrets/` — podcast_scraper runtime secrets (ADR-115)

sops/age-encrypted provider keys + Sentry DSNs, decrypted on the VPS into
`/run/secrets/podcast` (tmpfs) by `decrypt-secrets.sh` and **file-mounted** into the
api + pipeline containers via `compose/docker-compose.secrets.yml` — never env, never
plaintext at rest. Design: [ADR-115](../../docs/adr/ADR-115-multi-tenant-secret-delivery-sops-tmpfs-files.md).

## One-time: the VPS age recipient

The box holds one age **private** key at `/etc/vps-secrets/age.key`; tenants encrypt
to its **public** recipient. Provisioning (decision a):

1. `age-keygen -o vps-age.key` — keep the **private** key in your password manager.
2. Put its public key (`age1…`) into `infra/.sops.yaml` (replace the ADR-115 placeholder).
3. Stage the **private** key as GH secret `VPS_SECRETS_AGE_KEY` — `deploy-prod.yml`
   installs it to `/etc/vps-secrets/age.key` (0400) over the tailnet.

## Create / update `prod.enc.yaml`

```bash
cp infra/secrets/prod.yaml.template infra/secrets/prod.yaml   # NOT committed
$EDITOR infra/secrets/prod.yaml                                # fill real values
sops --encrypt --input-type yaml --output-type yaml \
  infra/secrets/prod.yaml > infra/secrets/prod.enc.yaml        # matches .sops.yaml rule
shred -u infra/secrets/prod.yaml                               # destroy plaintext
git add infra/secrets/prod.enc.yaml                            # commit ONLY the encrypted form
```

Each top-level key → `/run/secrets/podcast/<key>` (0444, in a 0700 dir) → via the shim,
the UPPERCASE env var the app reads (`openai_api_key` → `OPENAI_API_KEY`).

## Cutover (after the key + prod.enc.yaml exist)

1. `deploy-prod.yml` stages the age key + runs `decrypt-secrets.sh podcast …`.
2. Add `compose/docker-compose.secrets.yml` to the deploy `-f` chain.
3. Drop the 6 provider keys from the host `.env` (they now come from files).

## Rotate

Re-encrypt with new values + redeploy. Dynamic/audited rotation = OpenBao (#1162).
