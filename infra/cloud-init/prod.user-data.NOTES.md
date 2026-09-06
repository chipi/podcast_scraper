# `prod.user-data` — the long-form comments

Cloud-init `user_data` is capped by Hetzner at **32,768 bytes**, and the rendered payload
hit 34,488 — which broke every DR drill from 2026-08-05 (#2002). Half the template was
comments, so the prose moved here and the template keeps a one-line anchor pointing at each
note. Nothing about what the VM does changed.

Anchors read `[note N — see prod.user-data.NOTES.md]`.

## Note 1 — `default` MUST come first so cloud-init still creates Hetzner's default

```text
 `default` MUST come first so cloud-init still creates Hetzner's default
 user (preserves root's API-injected SSH key for emergency access). Without
 it, our explicit `deploy` user REPLACES the default user list and root
 loses its authorized_keys.
```

## Note 2 — ``.bootstrap-needs-env`` sentinel removed per #844 once .env became

```text
 ``.bootstrap-needs-env`` sentinel removed per #844 once .env became
 workflow-staged (#841): deploy-prod.yml / prod-restore-corpus.yml
 render /srv/podcast-scraper/.env from GH Secrets at deploy time.
 The systemd unit's ExecStartPre now positively checks for .env's
 presence (refuses to start if .env is missing — same protection,
 self-explanatory error).
```

## Note 3 — Allow deploy@ to refresh MagicDNS HTTPS after GHA runs deploy.sh (systemd

```text
 Allow deploy@ to refresh MagicDNS HTTPS after GHA runs deploy.sh (systemd
 ExecStartPost covers systemctl restarts only). ``install`` copies the canonical
 script from the repo (``/tmp/podcast-tailscale-serve.ci``) so CI can repair a
 broken first-boot ``write_files`` render without reprovisioning.
```

## Note 4 — Orrery co-tenant tailscale-serve wrapper (#838)

```text
 Orrery co-tenant tailscale-serve wrapper (#838).
 Publishes :8443 -> 127.0.0.1:8090 alongside podcast_scraper's :443.
 No systemd unit yet (chipi/orrery#261 phase 1) — invoked once from
 runcmd below at first boot, and re-armed by orrery's deploy workflow
 via the narrow sudoers allowlist immediately after.
```

## Note 5 — === Security hardening (#1160 Phase 2 — host) ===

```text
 === Security hardening (#1160 Phase 2 — host) ===
 cloud-init is first-boot-only and hcloud_server ignores user_data drift, so
 editing this block does NOT change the live VPS — it is for future rebuilds.
 Apply the same steps imperatively once on the live box (tailscale-serve
 precedent) to harden the running host.
```

## Note 6 — T-11 (ADR-117 / #1160): ban scanners probing the public edge. Caddy logs one

```text
 T-11 (ADR-117 / #1160): ban scanners probing the public edge. Caddy logs one
 JSON access line per request; this filter bans an IP that racks up a burst of
 4xx (path scanning, auth probing, or our own 429 rate-limit rejections).
 Host-level Caddy => the ban lands in the INPUT chain and drops the peer.
 Keys on `client_ip` (ADR-118), NOT `remote_ip`: pre-Cloudflare the two are
 equal (direct peer → real drop); once CF fronts the box, remote_ip is a CF
 edge and banning it would take the whole site down — client_ip stays the real
 visitor (behind CF the ban is a harmless no-op; CF-side rules take over).
```

## Note 7 — T-07 / T-10 (metadata slice): block containers from reaching the cloud

```text
 T-07 / T-10 (metadata slice): block containers from reaching the cloud
 metadata endpoint (169.254.169.254) — kills SSRF-to-cloud-creds from any app
 container. A oneshot unit re-inserts the DOCKER-USER rule after docker starts
 (idempotent) so it survives reboots — plain iptables rules do not persist.
```

## Note 8 — === INCIDENT-2026-08-05 prevention (prod root-disk exhaustion + tailnet clobber) ===

```text
 === INCIDENT-2026-08-05 prevention (prod root-disk exhaustion + tailnet clobber) ===
 Root cause was 128 GB of unpruned Docker images from deploying nonstop. Daily
 prune of unused images + build cache. Applied imperatively on the live box too
 (user_data is first-boot-only — see the hardening note above).
```

## Note 9 — Inbound tailnet TCP accept. During the incident, dockerd churn dropped

```text
 Inbound tailnet TCP accept. During the incident, dockerd churn dropped
 tailscale's INPUT accept rule, refusing ALL inbound tailnet TCP (SSH, :443)
 while pings still worked — blocking recovery. Oneshot re-inserts it after
 docker (mirrors the block-metadata-egress guard above). Idempotent; survives
 reboots. Plain iptables rules do not persist — this unit is why it does.
```

## Note 10 — === Secret infra (ADR-115) ===

```text
 === Secret infra (ADR-115) ===
 Shared multi-tenant secret-decrypt helper (sops/age -> /run/secrets/<tenant>
 tmpfs files). Injected via file() (see main.tf ``decrypt_secrets_body``) so
 its shell ``$${..}`` syntax survives templatefile — tailscale-serve precedent.
```

## Note 11 — === Public edge: shared Caddy engine (ADR-114 / #1158) ===

```text
 === Public edge: shared Caddy engine (ADR-114 / #1158) ===
 Base Caddyfile staged here, then copied to /etc/caddy/Caddyfile in runcmd
 AFTER ``apt-get install caddy`` (avoids dpkg clobbering our conffile) — same
 staging pattern as the Alloy config below. file()-injected (see main.tf
 ``caddy_base_body``) so Caddy ``{..}`` syntax survives templatefile.
```

## Note 12 — Shared reload grant — any tenant's deploy (running as ``deploy``) reloads the

```text
 Shared reload grant — any tenant's deploy (running as ``deploy``) reloads the
 engine after dropping its /etc/caddy/sites/<app>.caddy vhost. Engine-level, so
 one line covers all tenants (mirrors the caddy-reload grant in ADR-114 §5).
 reload for plain config changes; restart because the base Caddyfile sets
 `admin off` (T-02) so admin-API-based `caddy reload` fails — a config/vhost
 change then needs a restart (task #27: keep admin off + grant restart rather
 than reopen the admin surface). Both narrow + engine-level (ADR-114 §5).
```

## Note 13 — Edge convergence (ADR-114 Phase 1). Let `deploy` (the agent, via its CI key)

```text
 Edge convergence (ADR-114 Phase 1). Let `deploy` (the agent, via its CI key)
 run the root-owned apply-edge / verify-edge — idempotent edge/hardening/o11y
 reconcile that explicitly NEVER opens the firewall (Phase 4, Terraform). This
 is what lets the agent converge the edge WITHOUT a standing root credential.
 RUN-ONLY on root-owned copies (installed in runcmd from the repo): deploy can
 invoke them but not modify them, so a leaked deploy key can reconcile the edge
 but not inject arbitrary root code (tighter than the tailscale-serve install
 grants). Trade-off: updating the scripts needs a root re-install — automatic on
 a fresh box (runcmd below), or one operator touch to refresh a live box.
```

## Note 14 — === Belt-and-suspenders SSH key install ===

```text
 === Belt-and-suspenders SSH key install ===
 The `users:` block above SHOULD populate ~deploy/.ssh/authorized_keys, but
 we've seen it silently fail on Hetzner's Ubuntu 24.04 image (cloud-init
 parses the YAML fine but the keys never land). Brute-force the install
 via runcmd — this runs unconditionally after the user is (or should have
 been) created.
```

## Note 15 — Multi-key write via heredoc — primary operator key plus each

```text
 Multi-key write via heredoc — primary operator key plus each
 ``additional_authorized_keys`` entry on its own line. ``join`` renders
 at templatefile time; the ``\n    `` separator preserves the 4-space
 YAML literal-block indent for every subsequent key, otherwise the
 render breaks YAML parsing.
```

## Note 16 — Pre-create access.log OWNED BY caddy so the first `import hardened` vhost can

```text
 Pre-create access.log OWNED BY caddy so the first `import hardened` vhost can
 start. The caddy deb ships a root:root 0600 access.log the caddy service user
 can't write; the base Caddyfile never uses it, so it only breaks on the first
 hardened vhost drop (task #28, orrery go-live 2026-07-22). install -o caddy
 overwrites ownership idempotently.
```

## Note 17 — ADR-114: Caddy binds the PUBLIC IP only — ``tailscale serve`` owns :443 on the

```text
 ADR-114: Caddy binds the PUBLIC IP only — ``tailscale serve`` owns :443 on the
 tailnet IP, so an all-interfaces bind collides ("address already in use").
 Discover the public IP(s) from the default route (the metadata API is blocked
 by block-metadata-egress) and set the caddy service env for
 ``default_bind {$CADDY_BIND_ADDRS}``, then validate + start with it. Mirrors
 scripts/ops/apply-edge.sh so rebuild (cloud-init) and imperative apply agree.
```

## Note 18 — === Repo checkout to /srv/podcast-scraper ===

```text
 === Repo checkout to /srv/podcast-scraper ===
 Clone-to-tmp + ``cp -a`` (not a direct ``git clone`` into ``/srv/podcast-scraper``)
 so a non-empty destination can't make the clone fail and leave the tree absent —
 deploy workflows would then see ``deploy.sh: No such file or directory`` while SSH
 auth still works. Kept defensively against any future ``write_files`` entry under
 ``/srv/podcast-scraper`` (the old ``.bootstrap-needs-env`` sentinel that originally
 motivated this was removed in #844).
```

## Note 19 — Install root-owned copies of the edge-convergence scripts so `deploy` can run

```text
 Install root-owned copies of the edge-convergence scripts so `deploy` can run
 them via the 99-podcast-apply-edge grant (root-owned = deploy can't tamper).
 Same idiom as the decrypt-secrets root-owned script. On a live box that predates
 this, one operator re-install of these two lines does the same thing.
```

## Note 20 — Install the prod container-metrics collector from the checkout (feeds the homelab

```text
 Install the prod container-metrics collector from the checkout (feeds the homelab
 landing page's per-box Containers table + apps colouring for the prod box). Root-run
 at boot so it survives a rebuild; deploy@ can't do this itself (root-owned /opt +
 enabling a non-caddy unit are outside its narrow sudo). Standalone systemd service:
 reads local docker, pushes to homelab:8428 over the tailnet — independent of the
 homelab-managed /opt/vps-observability Alloy container (shares only the parent dir).
 On a live box that predates this, one operator re-run of these lines installs it
 identically. See infra/observability/container-metrics/README.md.
```

## Note 21 — === Repo checkout to /srv/orrery (mirror of /srv/podcast-scraper above) ===

```text
 === Repo checkout to /srv/orrery (mirror of /srv/podcast-scraper above) ===
 Co-tenant deploy on the same VPS (chipi/orrery#260 / #261). Phase 1 has no
 secret env or systemd unit gating, so the clone-to-tmp + cp -a dance from
 the podcast_scraper block isn't strictly needed — but mirror it anyway so
 the two app checkouts are structurally identical and survive any future
 write_files entry under /srv/orrery without surprises.
```

## Note 22 — === Orrery tailscale-serve first-boot invocation (#838) ===

```text
 === Orrery tailscale-serve first-boot invocation (#838) ===
 podcast_scraper has a systemd unit with ``ExecStartPost`` that fires this
 for its serve script; orrery has no unit yet (chipi/orrery#261 phase 1),
 so invoke once here. Subsequent re-arms come from orrery's deploy workflow
 via the narrow sudoers allowlist added above. Tailscale is already up at
 this point (``tailscale up …`` above).
 YAML-level single-quote wrap: the unquoted ``: `` inside the echo string
 would otherwise trigger plain-scalar-as-mapping parsing, crashing
 cloud-init's runcmd module entirely (TypeError in ``shellify``).
```
