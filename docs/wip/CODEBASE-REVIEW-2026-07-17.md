# Whole-codebase review — 2026-07-17

Adversarial multi-agent review (143 agents, 20 subsystems). **67 confirmed** of 122 raw findings (each confirmed by a second agent reading the actual code).

**Severity:** critical 0 · high 9 · medium 18 · low 40

## Confirmed findings (by severity)

| sev | subsystem | file | title |
|-----|-----------|------|-------|
| high | server-api | `src/podcast_scraper/server/routes/enrichment.py:194-338` | Enrichment status/health/metrics/events routes bypass OperatorWriteGuard |
| high | providers-ml | `src/podcast_scraper/providers/ml/diarization/cache.py:48-56` | Cache fingerprint excludes clustering_threshold, min_cluster_size, min_segment_ms, diarization_device — stale cache served on config change |
| high | enrichment | `src/podcast_scraper/enrichment/executor.py:605` | circuit.record_success() is never called — circuit failure counter never resets within a run |
| high | kg | `src/podcast_scraper/kg/pipeline.py:94-103` | Entity descriptions silently dropped when kind repair fires |
| high | workflow-pipeline | `src/podcast_scraper/workflow/episode_processor.py:1958-1977` | skip_existing + generate_summaries: direct-download path silently drops summarization |
| high | provisioning-workflows | `.github/workflows/reprocess-prod.yml:73-88` | User-controlled PROFILE input injected into remote SSH command string — command injection on prod VPS |
| high | provisioning-workflows | `.github/workflows/reprocess-prod.yml:29-35` | reprocess-prod has no GitHub Environment gate — any repo writer can rewrite prod corpus artifacts without approval |
| high | ops-scripts | `scripts/ops/restore_corpus_from_tarball_host.sh:24` | No rollback when tar extraction fails after corpus backup was already moved |
| high | security-crosscut | `infra/cloud-init/prod.user-data:167-196` | T-07/T-10 metadata-egress SSRF guard only applies on rebuild — live box is unprotected |
| medium | server-api | `src/podcast_scraper/server/routes/corpus_binary.py:57-68` | Corpus artwork endpoint lacks `os.path.realpath` symlink-escape guard |
| medium | server-api | `src/podcast_scraper/server/routes/corpus_text_file.py:78-135` | Corpus text-file endpoint lacks `os.path.realpath` symlink-escape guard |
| medium | server-api | `src/podcast_scraper/server/app.py:252-263` | CORS allows credentials from localhost dev origins — must not reach production |
| medium | providers-llm | `src/podcast_scraper/providers/mistral/mistral_provider.py:228-232` | Mistral SDK client has no HTTP timeout — mega-bundle / Voxtral calls can hang indefinitely |
| medium | enrichment | `src/podcast_scraper/enrichment/executor.py:669-684` | Per-enricher cost cap is permanently inert — record_cost() is never called |
| medium | enrichment | `src/podcast_scraper/enrichment/protocol.py:289-306` | sync_enricher runs in asyncio.to_thread — cancel_event is invisible inside the thread |
| medium | kg | `src/podcast_scraper/kg/pipeline.py:371-384` | kg_merge_pipeline_entities=False is silently ignored when LLM extraction fails |
| medium | workflow-pipeline | `src/podcast_scraper/workflow/episode_processor.py:560-562` | Segment file write silently swallows OSError — GI timestamps silently lost |
| medium | text-stages | `src/podcast_scraper/cleaning/commercial/detector.py:241-246` | _scan_for_pattern(reverse=True) dead code — block_start refinement never fires |
| medium | upgrade-migrations | `scripts/ops/restore_corpus_from_tarball_host.sh:50-53` | Upgrade rollback deletes corpus with no recovery on first deploy |
| medium | upgrade-migrations | `src/podcast_scraper/upgrade/migrations/m0003_gi_v3_typed_mentions.py:129` | Non-atomic per-file write leaves .gi.json permanently corrupt on mid-migration kill |
| medium | upgrade-migrations | `src/podcast_scraper/upgrade/migrations/m0003_gi_v3_typed_mentions.py:88-157` | m0003 has no verify() override — silent partial migration is undetectable |
| medium | preprocessing-audio | `src/podcast_scraper/preprocessing/audio/cache.py:57` | Non-atomic cache write: partial file returned as cache hit on concurrent access or crash |
| medium | rss-net-cache | `src/podcast_scraper/rss/parser.py:157-194` | Feed-embedded transcript/enclosure URLs fetched without scheme or host validation (SSRF lite) |
| medium | rss-net-cache | `src/podcast_scraper/rss/http_policy.py:255-287` | HostThrottle interval guarantee is violated under concurrent workers |
| medium | entrypoints-config | `src/podcast_scraper/cli.py:4749-4776` | ffmpeg skip-list missing four subcommands — they abort with sys.exit(1) if ffmpeg absent |
| medium | provisioning-workflows | `.github/workflows/deploy-prod.yml:60-78` | Missing-secret preflight emits skip=true (exit 0, green) instead of failing — silent non-deploy looks like a successful deploy |
| medium | security-crosscut | `compose/docker-compose.prod.yml:116` | LLM keys still injected as env vars — ADR-115 cutover not wired into deploy chain |
| low | providers-llm | `src/podcast_scraper/prompts/store.py:123-125` | Prompt store has no path containment check — config-driven path traversal possible |
| low | providers-ml | `src/podcast_scraper/providers/ml/diarization/pyannote_provider.py:27-37` | Pipeline.from_pretrained fallback on TypeError masks genuine load errors when 'token' appears in unrelated messages |
| low | providers-ml | `src/podcast_scraper/providers/ml/diarization/gemini_provider.py:126` | Gemini diarization uploads audio file to Google Files API with no size or type guard |
| low | enrichment | `src/podcast_scraper/enrichment/executor.py:674-684` | Envelope written to disk before per-enricher cost-cap check — output and accounting diverge |
| low | enrichment | `src/podcast_scraper/enrichment/enrichers/topic_theme_clusters.py:57-91` | O(n^4) average-linkage inside a sync thread — no timeout escape on large corpora |
| low | enrichment | `src/podcast_scraper/enrichment/executor.py:630` | t_start scope guard via locals() is fragile — silent zero-duration on first-attempt cancel |
| low | search | `src/podcast_scraper/search/backends/lancedb_backend.py:248-253` | Unparameterized SQL filter builder (_to_sql) — acknowledged latent injection |
| low | search | `src/podcast_scraper/search/chunker.py:189-191` | chunk.text uses space-join of sentences, diverging from original text slice at char_start:char_end |
| low | search | `src/podcast_scraper/search/backends/lancedb_backend.py:255-283` | Cross-tier BM25/vector score merge without normalization in Python-side fallback path |
| low | gi | `src/podcast_scraper/gi/invariants.py:94-99` | Quote fabrication check: transcript.find(text) returns first occurrence; misplaced-offset count may miss quotes with multiple occurrences |
| low | kg | `src/podcast_scraper/kg/schema.py:21-37` | Module-level schema cache has a non-atomic double-write race |
| low | kg | `src/podcast_scraper/kg/corpus.py:44-49` | O(n^2) duplicate check in scan_kg_artifact_paths with list membership test |
| low | workflow-pipeline | `src/podcast_scraper/workflow/run_manifest.py:284` | OS username written into run_manifest.json via $USER/$USERNAME env var |
| low | workflow-pipeline | `src/podcast_scraper/workflow/orchestration.py:52-59` | Production code imports and checks unittest.mock at runtime |
| low | text-stages | `src/podcast_scraper/cleaning/llm_based.py:69, 89` | LLM cleaner returns unstripped output despite computing stripped version |
| low | text-stages | `src/podcast_scraper/transcript_formats/cues.py:14-18, 22-30` | VTT/SRT timestamp regex caps hours at 99 — silently drops cues for long recordings |
| low | text-stages | `src/podcast_scraper/cleaning/commercial/diarization_signals.py:112-115` | TOPIC_DISCONTINUITY_BOOST fires when non-host speaks BEFORE candidate, not around it |
| low | upgrade-migrations | `src/podcast_scraper/upgrade/state.py:58-60` | Non-atomic ledger write corrupts state on mid-write kill |
| low | upgrade-migrations | `src/podcast_scraper/upgrade/state.py:86-96` | No file locking — concurrent upgrade invocations produce a corrupt ledger |
| low | preprocessing-audio | `src/podcast_scraper/preprocessing/audio/ffmpeg_processor.py:232-236` | silence_threshold string injected unsanitised into ffmpeg -af filter chain |
| low | preprocessing-audio | `src/podcast_scraper/preprocessing/audio/chunker.py:167` | Single-chunk ffmpeg failure aborts all remaining chunks with no cleanup of tmpdir |
| low | rss-net-cache | `src/podcast_scraper/rss/http_retry.py:142-144, 168-170` | Retry log message emits impossible 'attempt N/M' when N > M on last retry |
| low | utils | `src/podcast_scraper/utils/storage_backend.py:107-112` | Atomic write in LocalStorageBackend.upload() leaks .tmp sibling on copy failure (same issue in audio_cache.store() line 127-130) |
| low | utils | `src/podcast_scraper/utils/retryable_errors.py:203-208` | is_non_retryable_http_error() classifies any error containing 'invalid' as non-retryable, silently dropping retries for legitimate transient failures |
| low | utils | `src/podcast_scraper/utils/retryable_errors.py:111-125` | String-contains checks for HTTP status codes produce false positives on non-HTTP error messages |
| low | mcp | `src/podcast_scraper/mcp/server.py:26` | _safe wrapper leaks internal exception messages to MCP callers |
| low | mcp | `src/podcast_scraper/mcp/tools/connectivity.py:140` | `bridge` materialises the entire co-speaker set (k=10_000) into a Python set for membership test |
| low | entrypoints-config | `src/podcast_scraper/utils/redaction.py:42` | API_KEY_PATTERN value-guard misses hyphenated key formats (Anthropic sk-ant-, HF tokens) |
| low | entrypoints-config | `src/podcast_scraper/config.py:554-915 (Config model fields for API keys)` | API key fields stored as plain str in Config — no SecretStr / repr=False protection |
| low | terraform-iac | `infra/terraform/storage_box.tf:20-26` | Storage Box exposed publicly with password-only SFTP and no IP allowlist |
| low | terraform-iac | `infra/terraform/outputs.tf:39-47` | Storage Box SFTP host and username outputs not marked sensitive — leak in CI logs |
| low | terraform-iac | `infra/terraform/main.tf:93-107` | Corpus Volume size change will plan destroy+create with no lifecycle guard |
| low | terraform-iac | `infra/terraform/terraform.drill.ci.tfvars:20` | Committed CI tfvars opens SSH port 22 world-wide on drill servers |
| low | provisioning-workflows | `.github/workflows/backup-corpus-prod.yml:31-61` | backup-corpus-prod has no GitHub Environment gate and uses the same silent-skip pattern |
| low | provisioning-workflows | `.github/workflows/verify-backup-restore.yml:91` | Weekly backup-restore smoke test uses floating :main image tag — does not validate the image actually running in prod |
| low | ops-scripts | `scripts/ops/corpus_snapshot/restore_corpus_release.sh:63` | Codespace layout extraction does not guard against overwriting an existing .codespace_corpus tree |
| low | ops-scripts | `scripts/ops/corpus_snapshot/emit_manifest.sh:140` | --argjson for WORKFLOW_ATTEMPT silently breaks on any non-integer env value |
| low | security-crosscut | `infra/.env.drill.local:11` | Live Hetzner API token committed to disk without infra/.gitignore coverage |
| low | security-crosscut | `compose/docker-compose.player-public.yml:37` | FORWARDED_ALLOW_IPS=* trusts XFF from any source — safe only while api port is unexposed |
| low | security-crosscut | `infra/.gitignore:1-12` | infra/.gitignore does not cover .env.*.local files — relies solely on root .gitignore |

---

## Synthesized report

I'll synthesize the confirmed findings into a prioritized report. Let me work through the data — 67 confirmed findings, grouped by adversarially-verified severity (the `verdict.severity`, which corrected several raw ratings).

Let me organize by the verdict severity, dedupe overlapping issues, and produce the report.

# Prioritized Security & Correctness Review — Pre-Go-Live

Solo-dev prod push. 67 adversarially-confirmed findings across 20 subsystems. Severities below are the **post-verification** ratings (several raw "high"/"medium" were downgraded on evidence; a few premises were partly refuted — noted inline). Overlapping symlink/atomic-write/env-gate items are deduped into single entries.

---

## CRITICAL
None. No finding survived verification at critical. The T-01 crown-jewel risk (api container RW docker.sock + LLM keys) is not directly exposed by any confirmed finding.

---

## HIGH

**H1 — Metadata-egress SSRF guard only applies on rebuild; live box unprotected**
`infra/cloud-init/prod.user-data:167-196` — The iptables DROP on `169.254.169.254` is a cloud-init once-at-first-boot entry; the running VPS has no guard, so a compromised/SSRF-able container can reach Hetzner metadata + the Tailscale auth key rendered into user-data.
Fix: Apply imperatively on the live box (`install script + service; systemctl enable --now`) before opening the public firewall; add to the Goal-1 gate checklist.

**H2 — reprocess-prod: user-controlled `PROFILE` → command injection on prod VPS**
`.github/workflows/reprocess-prod.yml:73-88` — Free-form `profile` input is interpolated as `--config '${PROFILE}'` inside the SSH command string; a repo writer can break out of the quotes and run arbitrary commands as `deploy` on prod.
Fix: Change `profile` to `type: choice` pinned to `config/profiles/cloud_balanced.yaml|cloud_thin.yaml` (allowlist), or `printf '%q'`-quote before the remote call.

**H3 — reprocess-prod has no `environment: prod` gate**
`.github/workflows/reprocess-prod.yml:29-35` — Rewrites all prod corpus artifacts with only a typed confirm string any repo writer can supply; unlike deploy-prod/prod-restore it lacks the GH Environment required-reviewer gate.
Fix: Add `environment: prod` to the `reprocess` job; add a missing-secret preflight that `exit 1`s (mirror deploy-prod).

**H4 — restore script: no rollback when tar extraction fails**
`scripts/ops/restore_corpus_from_tarball_host.sh:24` — Under `set -e`, live `corpus/` is moved to `corpus.bak.$STAMP` then unguarded `tar -xzf` runs; on failure the host is left with no `corpus/` and api won't boot. Existing rollback (L47-54) only covers the migration path. (Note: a second unrestored `exit 1` at the layout-check guard L25-28 has the same gap.)
Fix: `if ! tar -xzf … ; then mv "corpus.bak.$STAMP" corpus; exit 1; fi` — mirror the migration rollback.

**H5 — Enrichment status/health/metrics/events/re-enable routes bypass OperatorWriteGuard**
`src/podcast_scraper/server/routes/enrichment.py:194-338` — Five GETs and one mutating POST (`re-enable`) aren't in `_OPERATOR_BASES` (only `/api/enrichment/config` is); gated only on the internal `jobs_api_enabled` flag. On a public api surface, unauthenticated read of operator data + one state mutation. (Mitigated today: api container is tailnet-private per T-01a.)
Fix: Replace `/api/enrichment/config` with `/api/enrichment` in `_OPERATOR_BASES` (subsumes all sub-routes); add `Depends(get_admin_user)` on `re-enable` as belt-and-suspenders.

**H6 — KG entity descriptions silently dropped when kind-repair fires**
`src/podcast_scraper/kg/pipeline.py:94-103` — `entities_for_repair` is stripped to `{name, kind}`, and on `ents_repaired > 0` the whole entity list is replaced with `{name, entity_kind}` — dropping `description` for the entire batch. Fires on common orgs (openai/anthropic/google), so hits most corpora.
Fix: Merge repaired kind back into originals: `{**orig_ent, "entity_kind": r["kind"]}` (mirrors the topic-path pattern already above it).

**H7 — skip_existing + generate_summaries: direct-download path silently drops summarization**
`src/podcast_scraper/workflow/episode_processor.py:1958-1977` — Returns `success=False` when reusing an existing transcript; both result handlers (`stages/processing.py:920`, `:1091`) only enqueue a ProcessingJob on `success=True`, so summarization/metadata is silently skipped for every direct-download episode with a pre-existing transcript. Whisper path handles this correctly; direct-download doesn't.
Fix: Enqueue on the `(False, non-None path, non-None source)` pattern (or return `success=True` for reuse + fix the counter). Add a regression test (repro-before-fix).

**H8 — Diarization cache fingerprint excludes tuning params → stale result served silently**
`src/podcast_scraper/providers/ml/diarization/cache.py:48-56` — Fingerprint hashes only model+speaker counts; `clustering_threshold`/`min_cluster_size`/`min_segment_ms` (all output-affecting, passed by `factory.py:54-58`) aren't in the key. Re-tuning against a warm cache serves the old wrong result with no detection. (`device` correctly excluded — doesn't affect output.)
Fix: Add the three tuning params to the fingerprint tuple (`getattr` defaults); write a `config_snapshot` into the cached JSON for defensive load validation.

**H9 — Enrichment circuit breaker never resets: `record_success()` never called**
`src/podcast_scraper/enrichment/executor.py:605` — `record_failure()` fires on exceptions but the happy path (L624-626) never calls `record_success()`, so `consecutive_failures` accumulates monotonically across bundles. Five spread-out transient failures on an EMBEDDING-tier enricher (threshold=5) falsely quarantine a healthy enricher. Every other circuit breaker in the repo pairs the two calls; this is the lone omission.
Fix: Call `circuit.record_success(policy)` on the success path after L625.

---

## MEDIUM

**M1 — LLM keys still injected as env vars; ADR-115 secrets cutover not wired into deploy chain**
`compose/docker-compose.prod.yml:116` — All 6 provider keys are plaintext container env (readable via `/proc/self/environ`, `docker inspect`); the sops/tmpfs `docker-compose.secrets.yml` overlay exists but is not in the deploy `-f` chain (CUTOVER GATE). Pre-public hardening blocker (tracked #1162/T-08).
Fix: Complete cutover — provision VPS age key + `prod.enc.yaml`, verify `decrypt-secrets.sh`, add `-f docker-compose.secrets.yml`, drop the 6 keys from `.env`.

**M2 — CLI ffmpeg skip-list missing 4 subcommands → `sys.exit(1)` on ffmpeg-less host**
`src/podcast_scraper/cli.py:4749-4776` — `mcp`, `index-two-tier`, `enrich-edges`, `cluster-corpus-topics` aren't in the skip tuple, so they abort before parsing on the api container. Same bug class already fixed for `upgrade` (7c85e610).
Fix: Add the four to the skip tuple at L4753.

**M3 — deploy-prod missing-secret preflight emits `skip=true` (green) instead of failing**
`.github/workflows/deploy-prod.yml:60-78` — A missing secret no-ops every step and exits 0; operator sees a green "successful deploy" that deployed nothing (GH Environment records success). Same soft-skip in `backup-corpus-prod.yml`. (Partly by-design for the pre-first-`tofu-apply` window; reconcile.)
Fix: `exit 1` on missing prereqs (fail only when explicitly triggered, to preserve the pre-provisioning rationale).

**M4 — SSRF-lite: feed-embedded transcript/enclosure URLs fetched with no scheme/host allowlist**
`src/podcast_scraper/rss/parser.py:157-194` — `urljoin` output flows to the downloader with no filter for loopback/link-local (`169.254.169.254`)/RFC-1918; feed-URL validation (feeds_spec) only covers operator-declared URLs, not XML-body URLs. Blind SSRF (response streamed to file); attacker must first get a malicious feed added.
Fix: Central scheme+private-range reject in `_open_http_request`.

**M5 — HostThrottle interval violated under concurrent workers**
`src/podcast_scraper/rss/http_policy.py:255-287` — `last_finish` is written in `release()` (post-request), not in `acquire()`, so with `host_max_concurrent=0` two workers both compute `wait≈0` and fire simultaneously, defeating the per-host throttle.
Fix: Update `last_finish` at end of `acquire()` before releasing the lock, or validate/coerce `interval_ms>0 ⇒ max_concurrent>=1`.

**M6 — Non-atomic audio cache write → truncated MP3 served as cache hit**
`src/podcast_scraper/preprocessing/audio/cache.py:57` — `shutil.copy2` writes in place; a mid-copy crash or same-key race (runs under a ThreadPoolExecutor) leaves a partial file that `os.path.exists` treats as a valid hit, feeding a truncated MP3 downstream (silent-corrupt transcript).
Fix: Write to same-dir tempfile then `os.replace` (atomic POSIX same-fs).

**M7 — Segment-file write swallows OSError at debug → GI timestamps silently lost**
`src/podcast_scraper/workflow/episode_processor.py:560-562` — `.segments.json` write failure (ENOSPC/perms) logs at debug; downstream GI quote timestamps stay at zero with no operator signal. `append_corpus_incident` is already imported and used for sibling failures — inconsistent.
Fix: Raise to WARNING and emit a corpus incident for both `_save_transcript_segments_file` and `_save_speaker_diagnostics_file`.

**M8 — m0003 GI-v3 migration: non-atomic per-file write leaves `.gi.json` permanently corrupt on mid-write kill**
`src/podcast_scraper/upgrade/migrations/m0003_gi_v3_typed_mentions.py:129` — `f.write_text(...)` truncates on kill; on replay the corrupt file is skipped as `unparsable` yet the migration still returns `applied=True`, silently abandoning it.
Fix: Atomic tmp+rename per file; override `verify()` (see M9).

**M9 — m0003 has no `verify()` override → partial migration undetectable**
Same file, L88-157 — Base `verify()` returns `(True, "no verification defined")`; m0001/m0002 both override it, m0003 doesn't, so `upgrade verify` is always green on a half-migrated corpus.
Fix: Override `verify()` to confirm all `.gi.json` parse + `schema_version==3.0` + no residual Insight-sourced MENTIONS edges; report unparsable count as failure.

**M10 — restore script upgrade rollback deletes corpus with no recovery on first deploy**
`scripts/ops/restore_corpus_from_tarball_host.sh:50-53` — On a first deploy (no prior corpus → no `.bak`), an upgrade failure runs `rm -rf corpus` while the guarded rollback finds nothing to restore. (Recoverable via re-running the workflow — source snapshot survives in backup repo — hence medium not high.)
Fix: Guard the `rm -rf corpus` on backup existence; error out preserving the extracted corpus if no backup exists.

**M11 — Mistral SDK client has no HTTP timeout**
`src/podcast_scraper/providers/mistral/mistral_provider.py:228-232` — Timeout is commented out with a false justification; the only un-timed client of all providers. A hung Voxtral/summarization call has no client deadline.
Fix: `client_kwargs["timeout_ms"] = int(float(getattr(cfg,"mistral_timeout",600))*1000)` — **note the finding's proposed `timeout=get_http_timeout(...)` is wrong for this SDK; it only accepts `timeout_ms` (int ms).**

**M12 — CORS allows credentials from localhost dev origins with no prod override**
`src/podcast_scraper/server/app.py:252-263` — Hardcoded `localhost:5173/5174` origins + `allow_credentials=True`, added unconditionally even to the intended-public app-only backend; no env knob to set the real public hostname. Auth is cookie-based (real credentialed-CSRF primitive, but only from a localhost page in the victim's own browser).
Fix: Env-driven `PODCAST_SERVE_CORS_ORIGINS` (default the localhost set); set to real hostname(s) in prod.

**M13 — `kg_merge_pipeline_entities=False` silently ignored when LLM extraction fails**
`src/podcast_scraper/kg/pipeline.py:371-384` — Guard `if llm_partial and not merge_pipe` means on LLM failure (`llm_partial` falsy) pipeline hosts/guests are injected regardless of the flag, contradicting the config docstring.
Fix: Gate injection on `merge_pipe` unconditionally, or document the failure-path exception and update the docstring.

**M14 — sync_enricher cancel_event invisible inside the worker thread**
`src/podcast_scraper/enrichment/protocol.py:289-306` — `@sync_enricher` runs via `asyncio.to_thread`; a running sync body can't be interrupted by `wait_for`/cancel_event, so an O(n²) corpus enricher burns a CPU core after cancel/timeout until it self-completes. (Bounded — hard `wait_for` still returns control; wasted-compute, not a hang. Also a doc-vs-code gap: the docstring instructs bodies to check cancel_event, which sync bodies structurally can't honor.)
Fix: Add periodic `ctx.cancel_event.is_set()` checks in O(n²) inner loops (`ctx` already passed).

**M15 — Per-enricher cost cap permanently inert — `record_cost()` never called**
`src/podcast_scraper/enrichment/executor.py:669-684` — Cap checks read an always-empty dict; an operator declaring `manifest.max_cost_usd_per_run` gets silent non-enforcement. (Documented deferral to chunk-5 LLM enrichers; current enrichers cost $0, so no live leak — the risk is false-sense-of-safety.)
Fix: Gate the check on `run_total_cost > 0` or add a visible executor-body FIXME so the inert cap isn't mistaken for active.

**M16 — Commercial detector `_scan_for_pattern(reverse=True)` is dead code**
`src/podcast_scraper/cleaning/commercial/detector.py:241-246` — Forward regexes are run against a char-reversed string, so BLOCK_START refinement never matches; `block_start` always falls back to the paragraph boundary, over-removing up to ~500 chars of legitimate content when a sponsor mention sits mid-paragraph. (Catastrophic-delete case separately guarded.)
Fix: Use rightmost forward match (`finditer`) across patterns; keep paragraph fallback as safety net.

---

## LOW (worth doing soon)

Grouped; each is a real but bounded defense-in-depth / hygiene / minor-correctness item.

**Symlink-escape & atomic-write hardening (dedup cluster)**
- `server/routes/corpus_binary.py:57-68` & `corpus_text_file.py:78-135` — string-only path checks, no `os.path.realpath`; sibling `corpus_media.py` already has `_within_root_realpath`. Requires in-corpus symlink plant. → Extract `_within_root_realpath` to `utils/path_validation.py`, apply before `FileResponse` in both.
- `utils/storage_backend.py:107-112` & `preprocessing/audio/audio_cache.py:127-130` — `.tmp` sibling leaks on copy failure (self-heals on same-key retry; SIGKILL uncatchable). → try/finally `tmp.unlink(missing_ok=True)`.
- `upgrade/state.py:58-60` (ledger), `:86-96` (concurrent record_applied no lock) — non-atomic/unlocked; self-healing via idempotent migrations. → tmp+`replace`; `fcntl.flock`.
- `scripts/ops/corpus_snapshot/restore_corpus_release.sh:63` — no overwrite guard vs sibling `import_local_snapshot.sh` (both branches). → add existence check.

**Secrets hygiene / gitignore (dedup cluster — no live leak)**
- `config.py:554-915` API keys as plain `str`, no `SecretStr`/`repr=False`. → `SecretStr` or `Field(repr=False, exclude=True)`.
- `utils/redaction.py:42` value-pattern misses hyphenated keys (name-denylist covers all live fields; note finding mis-described `hf_` handling). → extend prefixes.
- `infra/.gitignore` / `infra/.env.drill.local` — file is ignored via root `.gitignore` (not committed, not in history); missing infra-local backstop only. → add `.env.*.local` to `infra/.gitignore`. **Not a live token leak** — the "committed to disk" framing is refuted.

**Terraform / IaC (all off-by-default or already-guarded)**
- `storage_box.tf:20-26` password-only public SFTP — off by default, low-sensitivity re-fetchable public audio; **note finding's `ssh_keys` remediation isn't schema-valid**. → high-entropy password.
- `outputs.tf:39-47` box host/user not `sensitive` (leaks to CI logs; password stays redacted). → `sensitive = true`.
- `main.tf:93-107` volume resize plans destroy+create — but `delete_protection=true` already blocks the real destroy. → `prevent_destroy` belt-and-suspenders.
- `terraform.drill.ci.tfvars:20` SSH 0.0.0.0/0 — deliberate documented break-glass on ephemeral drill VMs, key-only+fail2ban. → log effective rules.

**Injection/validation defense-in-depth (all currently unreachable from untrusted input)**
- `search/backends/lancedb_backend.py:248-253` `_to_sql` f-string (OQ-3, tracked) — no caller passes user filters today. → parameterize/allowlist before wiring user filters.
- `prompts/store.py:123-125` no path-containment — inputs are operator config/internal literals, **not API-reachable** (premise refuted). → add `resolve().is_relative_to`.
- `preprocessing/audio/ffmpeg_processor.py:232-236` unsanitized `silence_threshold` into `-af` — **not shell/RCE**, operator-only argv, ffmpeg-filter-DSL only. → regex validator in config.
- `player-public.yml:37` `FORWARDED_ALLOW_IPS=*` — safe while api port unmapped (it is); **CIDR remediation not uvicorn-valid**. → add a contract test asserting api has no host `ports` mapping.
- `mcp/server.py:26` exception messages to MCP callers — stdio-only, operator is the caller. → return class name + opaque id.

**Minor correctness / robustness**
- `player-public` cost cap ordering `executor.py:674-684` — envelope written before cost check (latent; cap inert today). → reorder.
- `gi/invariants.py:94-99` `find()` first-occurrence → false misplaced-quote count on repeated phrases. → `re.finditer` any-within-slack.
- `kg/pipeline.py` deps: `schema.py:21-37` cache double-write race (benign); `corpus.py:44-49` O(n²) redundant dedup guard. → lock/`lru_cache`; drop guard.
- `search/chunker.py:189-191` space-join diverges from `char_start:char_end` slice (cosmetic; **finding's "verified by test" claim is false**). → slice original.
- `search/backends/lancedb_backend.py:255-283` cross-tier score merge without normalization — non-default signal paths only. → RRF-style per-tier norm before enabling KG-proximity.
- `cleaning/llm_based.py:69,89` returns unstripped output (cosmetic). → return `stripped`.
- `transcript_formats/cues.py` hours capped at `\d{1,2}` — needs >99h recording (unreachable for podcasts). → `\d+`.
- `cleaning/commercial/diarization_signals.py:112-115` topic-discontinuity boost doesn't check `after` vs host — vs RFC-060 spec; +0.15 precision nudge. → require speaker change on both sides.
- `providers/ml/diarization/pyannote_provider.py:27-37` TypeError `"token"` substring fallback (self-limiting — re-raises genuine failures). → `inspect.signature`.
- `providers/ml/diarization/gemini_provider.py:126` upload no size/exists/mime guard (path is internal). → mirror deepgram's `os.path.exists` guard.
- `enrichment/executor.py:630` `locals()` t_start guard (refactor-fragile). → hoist/sentinel.
- `enrichment/enrichers/topic_theme_clusters.py:57-91` O(n⁴) linkage in sync thread — hard 30s `wait_for` already caps; leaks one CPU thread. → merge-step/time guard.
- `utils/retryable_errors.py:203-208` bare `"invalid"` → non-retryable (redundant w/ `_TERMINAL_SIGNALS`; **5xx worst-case refuted** — status checks run first). → remove/narrow.
- `utils/retryable_errors.py:111-125` `"500" in str(error)` false positives (default is already retryable; narrow ordering hazard). → `elif http_code is None:` guard.
- `rss/http_retry.py:142-144,168-170` log "attempt 6/5" off-by-one (log-only). → denominator `+1`.
- `mcp/tools/connectivity.py:140` materializes k=10000 co-speaker set for one membership test (**k not the traversal driver — root-cause partly wrong**). → short-circuit helper.
- `workflow/run_manifest.py:284` OS username in manifest — **no public-exposure path** (Caddy serves no static files; premise refuted). → drop/replace with run_id.
- `workflow/orchestration.py:52-59` production imports `unittest.mock` in dispatch (test scaffolding in prod; no live harm). → explicit DI.
- `preprocessing/audio/chunker.py:167` single-chunk ffmpeg failure leaks tmpdir (rare path; **finding's "log+continue" fix is unsafe** — would silently drop audio). → only the try/finally cleanup half is in-scope.
- `verify-backup-restore.yml:91` smoke test uses floating `:main` not prod SHA (complementary DR drill covers real image; `last_deployed_prod_version.json` sha is `unknown` so that remediation source won't work). → resolve deployed SHA via GHCR API.
- `backup-corpus-prod.yml:31-61` no `environment:` gate + silent-skip — **but no `schedule:` trigger exists** (workflow_dispatch-only), so the "silent scheduled backup rot" impact is refuted. → add `environment: prod`, `exit 1` preflight for consistency.
- `emit_manifest.sh:140` `--argjson` on non-int aborts loudly (not silent; fail-fast). → regex guard.
- `kg/schema.py`, `kg/corpus.py` — see minor cluster above.

---

## "MUST fix before go-live" shortlist
Ordered by blast radius on the public cutover:
1. **H1** — apply metadata-egress guard on the live box (it's an explicit pre-public gate item; blocks tailnet-key exfil).
2. **H2** — pin `reprocess-prod` `profile` to `type: choice` (prod RCE via workflow).
3. **H3** — add `environment: prod` to `reprocess-prod`.
4. **M1** — complete ADR-115 secrets cutover (drop 6 LLM keys from container env) — the T-08 pre-public hardening blocker.
5. **H5** — add `/api/enrichment` to `_OPERATOR_BASES` (do before any public api exposure; harmless to land now).
6. **H4 + M10** — restore-script rollback guards (both the tar-fail and first-deploy paths) — recovery-critical, no <5-min rollback today.

## "Worth doing soon" (post-go-live, high value)
- **H6, H7, H8, H9** — the four silent data-quality/correctness bugs (KG description loss, dropped summarization, stale diarization cache, false circuit quarantine). Each degrades output invisibly; each needs a repro test.
- **M6, M8, M9** — atomic-write + verify() for audio cache and m0003 (silent-corruption class).
- **M2, M3** — CLI ffmpeg skip-list + deploy-prod fail-loud preflight (operational footguns).
- **M11** — Mistral timeout (use `timeout_ms`, not the finding's proposed API).
- **M4, M5, M7, M13, M16** — SSRF allowlist, throttle correctness, GI-timestamp incident surfacing, KG merge-flag contract, dead sponsor-detector code.

**Verification caveats to trust:** the report flags six findings whose *premise or proposed fix was partly refuted* — M11/storage_box/player-public remediations aren't valid as written; the run_manifest public-leak, prompts/store API-reachability, chunker "tested" claim, connectivity k-scaling, and backup-corpus "scheduled rot" impacts don't hold. Apply the corrected fixes noted inline, not the raw recommendations.