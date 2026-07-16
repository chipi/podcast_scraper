# Outbound HTTP factory — #1129 (proxy) + #1130 (TLS trust)

Shared design note for the two admin-config issues. Both extend the same central
outbound-httpx factory + operator-config surface.

## Scope split

| Issue | Config block | Adds |
| --- | --- | --- |
| #1129 | `outbound.proxy` | `enabled`, `url` (http/https + optional basic auth), `no_proxy` list |
| #1130 | `outbound.tls`   | `verify` toggle, `ca_bundle`, `client_cert`, `client_key` (mTLS) |

Land #1129 first (introduces the factory + registry); #1130 extends both.

Explicitly **out of scope** for these tickets: inbound HTTPS termination, SOCKS5,
per-service proxy selection, cert pinning, per-service trust overrides.

## Module layout

```
src/podcast_scraper/net/                          # new package
├── __init__.py
├── outbound_config.py         # OutboundConfig dataclass + parse/validate/redact
├── outbound_registry.py       # process-wide singleton, atomic hot-swap
└── outbound_http.py           # client factory (sync + async httpx.Client builders)
```

Existing `providers/resilience/sockets.py::hardened_http_client()` refactors to
compose on top of the new factory (keeps its TCP-keepalive + `Connection: close`
tweak; delegates transport/proxy/TLS to the factory).

## Config shape (operator YAML)

```yaml
outbound:
  proxy:
    enabled: false
    url: null            # e.g. http://user:pass@proxy.internal:3128
    no_proxy: []         # e.g. ["localhost", "127.0.0.1", "*.internal"]
  tls:
    verify: true
    ca_bundle: null      # path to PEM
    client_cert: null    # path to PEM (mTLS)
    client_key: null     # path to PEM (mTLS; must appear iff client_cert set)
```

Absent block = defaults (proxy off, verify=true, system trust). Adding to
`operator_config_security._FORBIDDEN_NORMALIZED` isn't needed — the block itself
is not a secret; the proxy password is inline inside `url`. Redaction handles it.

## Redaction

- `outbound_config.redact()` masks the userinfo segment of `proxy.url` before it
  is echoed to the operator-config GET response or logged.
- Reuses `podcast_scraper.utils.log_redaction` patterns; adds a small
  `_redact_proxy_userinfo` helper (userinfo is not a Bearer/Basic form so we can't
  rely on the existing regexes).
- `verify: false` sets a `insecure_mode_flags: {"tls_verify": false}` echo field
  on GET, plus a `logger.warning` at every config swap.

## Hot-apply

`OutboundConfigRegistry` (module-level singleton):

- `current() -> OutboundConfig` — cheap read on every factory call.
- `swap(new: OutboundConfig)` — atomic replace under a lock; emits a
  `logger.info` with redacted diff.

Factory contract:

- `create_client(*, subsystem: str, timeout=..., transport_wrapper=..., **httpx_kwargs) -> httpx.Client`
- `create_async_client(*, subsystem: str, timeout=..., transport_wrapper=..., **httpx_kwargs) -> httpx.AsyncClient`
- `sdk_http_client(*, subsystem: str, **kwargs) -> httpx.Client | None` —
  same shape but returns `None` on any construction error, safe to pass
  as `http_client=` to a vendor SDK constructor without breaking init.
- All three read the *current* registry snapshot when constructing the
  transport.

Rebuild-on-swap coverage (as shipped 2026-07-07):

- **RSS thread-local sessions** — rebuilt via
  `podcast_scraper.rss.downloader.reset_http_sessions()`. Callers that
  swap the registry mid-pipeline should call this to invalidate the
  thread-local `httpx.Client` cache.
- **Provider SDK clients (OpenAI / Anthropic / Mistral / Grok / DeepSeek
  / Ollama)** — built once at provider `__init__` and NOT rebuilt on
  swap. The vendor SDKs cache their own httpx transport internally, so
  reinjecting `http_client=` requires reconstructing the whole provider.
  A registry swap mid-pipeline leaves already-constructed provider
  clients on the old proxy/TLS config until the pipeline finishes and
  the providers are re-instantiated on the next run. This is a documented
  limitation, not a bug: in practice the operator swaps config between
  pipeline runs, not during one. Env-mirror (`HTTPS_PROXY` etc.) does
  reach the SDK's underlying httpx defaults but the factory-injected
  `http_client=` overrides it, so an in-flight SDK client keeps the old
  transport regardless.
- **All other call sites** (`audio_bridge`, `dgx_health`, `oauth`,
  `hf_system_prompt`, `podcast_obs`, `webhook`, judges, embed) build
  the client per-call via `create_client(...)` context managers — they
  automatically pick up the current snapshot on every call.

Env-var mirror (for third-party libs whose transport we cannot inject):

- On every registry swap the factory also updates the process-wide env vars
  `HTTPS_PROXY` / `HTTP_PROXY` / `NO_PROXY` (proxy block) and
  `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` (TLS block).
- This is the honest coverage story for `huggingface_hub` / pyannote and any
  vendor SDK we can't reach. Documented as such in both issues.

**Env-mirror scope — precise limits.** The mirror covers only two knobs:
proxy routing (`HTTPS_PROXY` family) and CA bundle (`SSL_CERT_FILE` family).
It does NOT cover:

- `tls.verify=False`. No widely-honored env var disables verification;
  SDK defaults are `verify=True`. A `verify=False` intent that falls back
  to env-only routing silently keeps verifying.
- `tls.client_cert` / `tls.client_key` (mTLS). Each SDK has its own client-
  cert API; no cross-vendor env convention. A mTLS intent that falls back
  to env-only routing silently omits the client cert.

Consequences of that scope:

- `huggingface_hub` and `pyannote` (transport-internal SDKs) work only for
  proxy + CA bundle. `verify=False` or mTLS toward those services is a
  design change, not a config change.
- `sdk_http_client(...)` returning `None` on error logs at ERROR so
  operators can page on the divergence when `verify=False`/mTLS is in play.
- The DGX Whisper + diarize multipart POSTs go through
  `hardened_http_client`, which delegates to `create_client(...)` — those
  requests get the full registry treatment (proxy + verify + mTLS) and
  never depend on env-mirror. This was the biggest gap plugged in the
  hardening pass (2026-07-09).

## Migration matrix (per subsystem)

Full inventory from the recon pass. Each row lists the touch-up scope for the
factory rollout.

**Status column reflects the AS-SHIPPED state on `main` after the 2026-07-16
option-3 rebase.** Rows marked ✅ shipped route production traffic through
the outbound HTTP factory today. Rows marked 🔵 planned describe a designed
migration that has NOT landed on `main` — the callsite still uses bare
`httpx.Client` / `requests.Session` / SDK-default transport. Chunk names
(3b / 3c / 3e) reference the original design plan, not merged commits;
they're preserved so a future PR that lands one of these rows can cite the
same chunk it belongs to.

Legend: ✅ shipped · 🔵 planned (not yet merged) · ⛔ documented exception

| # | Subsystem | Files | Migration | Status |
| - | --- | --- | --- | --- |
| 1 | RSS feed downloads | `rss/downloader.py` (currently `requests.Session` + urllib3 `Retry`) | Full swap to `httpx.Client` via `create_client(subsystem="rss_feed", transport_wrapper=RetryTransport(...))`; urllib3 `Retry` → a new `rss/http_retry.py::RetryTransport` (currently unwritten). | 🔵 planned (chunk 3c) |
| 2 | Media/transcript downloads | same file | Same client, `subsystem="rss_generic"`. | 🔵 planned (chunk 3c) |
| 3 | Audio bridge resolution | `server/app_audio_bridge.py::_head_request` | `create_client(subsystem="audio_bridge", ...)`. | 🔵 planned (chunk 3b) |
| 4a | OpenAI SDK | `providers/openai/openai_provider.py` | `client_kwargs["http_client"] = sdk_http_client(subsystem="llm_openai", timeout=...)`. | 🔵 planned (chunk 3e) |
| 4b | Anthropic SDK | `providers/anthropic/anthropic_provider.py` | Same, `subsystem="llm_anthropic"`. | 🔵 planned (chunk 3e) |
| 4c | Mistral SDK | `providers/mistral/mistral_provider.py` | `Mistral(client=..., ...)` (2.x kwarg name) with `try/except TypeError` fallback for 1.x. `subsystem="llm_mistral"`. | 🔵 planned (chunk 3e) |
| 4d | Grok SDK | `providers/grok/grok_provider.py` | OpenAI SDK, `subsystem="llm_grok"`. | 🔵 planned (chunk 3e) |
| 4e | DeepSeek SDK | `providers/deepseek/deepseek_provider.py` | OpenAI SDK, `subsystem="llm_deepseek"`. | 🔵 planned (chunk 3e) |
| 5 | Ollama LLM | `providers/ollama/ollama_provider.py` | OpenAI SDK via factory (`subsystem="llm_ollama"`). Direct `httpx.get/post` health calls will stay on env-mirror because test rig patches `httpx` at the module namespace. | 🔵 planned (chunk 3e) |
| 6a | DGX embed shim | `providers/ml/embedding_remote.py` | `create_client(subsystem="embedding_remote", ...)`. | 🔵 planned (chunk 3b) |
| 6b | Ollama embed | `providers/ml/embedding_ollama.py` | `create_client(subsystem="embedding_ollama", ...)`. | 🔵 planned (chunk 3b) |
| 7a | Ollama chat judge | `evaluation/judges/ollama_chat.py` | `create_client(subsystem="judge_ollama", ...)`. | 🔵 planned (chunk 3b) |
| 7b | vLLM chat judge | `evaluation/judges/vllm_chat.py` | `create_client(subsystem="judge_vllm", ...)`. | 🔵 planned (chunk 3b) |
| 8 | Tailnet DGX health | `providers/tailnet_dgx/health.py` | Three call sites: `dgx_health_probe` / `dgx_ollama_health` / `dgx_service_health`. Currently uses bare `httpx.Client`. | 🔵 planned (chunk 3b) |
| 9 | Job webhooks | `server/job_webhook.py` | Extracted `_post_webhook(url, payload, timeout)` seam using `create_async_client(subsystem="webhook", ...)`; test rig patches the seam. | 🔵 planned (chunk 3e) |
| 10 | OAuth identity | `server/app_oauth.py::GoogleProvider.exchange_code` | `create_client(subsystem="oauth_google", ...)`. | 🔵 planned (chunk 3b) |
| 11 | Reference client | `server/app_reference_client.py` | Test harness — bare httpx; not factory-relevant. | ⛔ documented no-op |
| 12 | HF system prompt fetch | `providers/common/hf_system_prompt.py` | Extract `_http_get(url, headers, timeout)` seam using `create_client(subsystem="hf_system_prompt", ...)`. Tests patch the seam. | 🔵 planned (chunk 3e) |
| 13 | podcast_obs helpers | `podcast_obs/_http.py` | `create_client(subsystem="podcast_obs", ...)`. | 🔵 planned (chunk 3e) |
| 14 | pyannote diarization | `providers/ml/diarization/pyannote_provider.py` | **Exception**: `huggingface_hub` owns the transport internally. Env-mirror (`HTTPS_PROXY` / `SSL_CERT_FILE`) covers proxy + CA. | ⛔ documented exception |
| 15 | DGX inference (Whisper) | `providers/tailnet_dgx/whisper_provider.py` | Uses `hardened_http_client(subsystem="dgx_whisper", ...)` which delegates to `create_client` (TCP keepalive + `Connection: close` + factory routing). | ✅ shipped |
| 16 | DGX inference (Diarize) | `providers/tailnet_dgx/diarization_provider.py` | Same delegate, `subsystem="dgx_diarize"`. | ✅ shipped |
| 17 | MOSS provider | `providers/moss/moss_provider.py` | Uses `hardened_http_client(...)` (default `subsystem="dgx_inference"`). | ✅ shipped |

**Migration coverage today:** 3 of 17 callsites (rows 15, 16, 17) route through
the factory via `hardened_http_client`. The other 14 rows are staged as
follow-up work; the factory itself is landed and stable, so each of the
planned migrations is a small independent PR.

### SDK-injection quick check (before writing code)

Confirm each vendor SDK exposes an `http_client=` kwarg on its constructor
(current openai>=1, anthropic>=0.34, etc. all do). Where it doesn't, fall back
to env-mirror only for that provider and log an INFO once at construction.

## Testing shape

- **Mock proxy fixture**: tiny threaded HTTP CONNECT proxy in a pytest fixture
  (~80 LOC; no new dep). Records observed hits so tests can assert traversal.
- **Mock TLS server**: `pytest-httpserver` (already? check) or a stdlib
  `http.server.HTTPServer` wrapped in `ssl.wrap_socket` with a self-signed CA
  fixture in `tests/fixtures/tls/`.
- **Env-mirror tests**: assert `HTTPS_PROXY` / `SSL_CERT_FILE` are set/unset
  precisely — no cross-test leakage (fixture snapshots + restores `os.environ`).
- **Redaction test**: GET config response never contains `user:pass@`.
- **Warning test**: `verify=false` logs `WARNING` at swap AND the config-GET echo
  carries `insecure_mode_flags.tls_verify == false`.
- **Hot-apply test**: swap registry mid-test, next `create_client(...)` uses
  new settings without process restart.

CI must not reach the internet — all tests offline.

## Mock servers — two scopes, both already covered

This branch's testing surface uses TWO distinct localhost mock servers.
Neither needs to evolve for #1129 / #1130 / #1142 as landed; the
sections below record what each covers and what would trigger an
extension.

### Scope 1: outbound-network layer (`tests/integration/net/conftest.py`)

New in this branch. Scoped to the outbound factory / proxy / TLS trust
plumbing:

- `mock_target_server` — plain HTTP echo, records seen requests.
- `mock_http_proxy` — threaded HTTP forward proxy, records observed
  hops. Optional Basic-auth challenge (`mock_http_proxy_with_auth`
  fixture).
- `mock_tls_server` — target wrapped in an `ssl.SSLContext` presenting
  a self-signed cert.
- `mock_mtls_server` — same, with `CERT_REQUIRED` client-cert exchange.
- `self_signed_ca` — session-scoped fixture minting CA + server + client
  certs via openssl.

**Purpose:** prove that the outbound factory
(`podcast_scraper.net.create_client`) actually threads admin-configured
proxy / TLS trust into every migrated call site AND — chunk 7b —
threads through the OpenAI SDK's `http_client` kwarg. This is a
network-layer seam, not a vendor API contract.

**Evolution triggers (none active):**

- CONNECT-through-proxy tunneling — currently plain-HTTP only. Extend
  `mock_http_proxy` with a CONNECT handler (~50 LOC) if HTTPS-through-
  proxy semantics need direct testing.
- Certificate pinning or per-service TLS trust (both explicitly out of
  scope for #1130).
- Proxy auth beyond Basic (Digest / NTLM / Kerberos) — YAGNI.

### Scope 2: vendor LLM API surface (`tests/e2e/fixtures/e2e_http_server.py`)

**Pre-existing.** The canonical mock server for E2E provider tests. It
serves the full chat / messages / generate_content / embed / whisper
endpoints for every vendor plus fault-injection primitives that
guardrail and resilience E2E tests use:

- OpenAI-shape `chat/completions` — used by OpenAI, DeepSeek, Grok,
  Ollama (they share the SDK).
- Anthropic-shape `/v1/messages` — used by Anthropic.
- Gemini-shape `/v1beta/generateContent` — used by Gemini.
- Mistral-shape `chat/complete`.
- Whisper transcription + diarization endpoints.
- `_emit_chat_violation`, `_emit_anthropic_violation`,
  `_emit_gemini_violation` — inject empty / thinking-prose /
  finish=length response shapes for ADR-099/100 guardrail E2E.
- `set_error_behavior`, `set_transient_error` — inject permanent /
  transient 5xx for per-stage resilience E2E.

Every `test_*_provider_e2e.py` uses this server. Chunk 3e + chunk 5–6
of #1142 didn't change any of that: the `mega_bundled` E2E tests for
all 6 billable providers (OpenAI/Anthropic/Gemini/Mistral/DeepSeek/Grok)
still hit the mock server via their vendor SDK and still round-trip
cleanly through the `BundledSummarizationMixin` flow. Ollama has no
E2E provider test (free / local target, out of scope for cost-tracked
E2E).

**Evolution triggers for #1142:**

- A new provider whose SDK isn't already served by the endpoints
  above → add a `_handle_<vendor>_...` method to the mock server, wire
  it into `do_POST` / `do_GET`, and add matching fault-injection
  primitives per the guardrail vocabulary at the top of
  `_injected_violations`. See the `ADD_LLM_PROVIDER_GUIDE.md` § E2E
  section for the concrete pattern.
- Provider ships a new response shape (e.g. streaming tool calls) that
  the current handlers can't emit → extend the handler.

**Why the mixin migration didn't require new E2E cases.** All three
mixin methods (`summarize_bundled`, `summarize_extraction_bundled`,
`summarize_mega_bundled`) route through the SAME `_call_bundled_llm`
per provider. Pre-#1142 the three method bodies were literal copies of
each other — the `mega_bundled` E2E case covers the shared flow. Post-
#1142 the flow IS shared, so the E2E coverage story hasn't weakened;
one E2E per provider still guards all three methods.

### Why we have both

The two mocks have non-overlapping scopes and aren't substitutes:

- The network-layer mocks can't validate vendor JSON shapes (they
  return plain-text `"target-ok"`); they exist to prove the factory
  threads traffic correctly.
- The E2E mock LLM server can't validate outbound proxy / TLS routing
  (it doesn't proxy, and there's no `CONNECT`); it exists to prove
  full-pipeline provider behavior.

Chunk 7b (`test_openai_sdk_http_client_routes_through_proxy`) is the
tiny bridge that connects them: real OpenAI SDK → outbound factory
`http_client` → network-layer mock proxy → mock target. It's ~40 LOC
and covers the ONE seam neither mock server proves on its own.

## Validation gates

- `make test-unit` — new `tests/unit/net/` module.
- `make ci-fast` — full sweep at end.
- `make docs` — this WIP note + any doc cross-refs.

## Scope expansions confirmed 2026-07-07

- **UI included in this branch** (both issues). #1128 shipped 2026-06 with the
  admin gate + `Admin` tab already wiring `UsersAdminView` /
  `RankingConfigAdminView` / `GraphAnalyticsAdminView`. New sibling:
  `web/gi-kg-viewer/src/components/admin/NetworkConfigAdminView.vue` — one
  admin form covering `outbound.proxy` (#1129) + `outbound.tls` (#1130) with
  redacted GET / validated PUT and a loud `verify=false` warning surface.
- **`requests.Session` migrated to httpx** in `rss/downloader.py` + `providers/common/hf_system_prompt.py`.
  Retry logic in `rss/downloader.py` is currently urllib3-tied (`HTTPAdapter`
  + `LoggingRetry`); rewrite it as an httpx `Transport` wrapper that observes
  the same retry codes (408/429/500/502/503/504) and methods (GET/HEAD/OPTIONS)
  and preserves the `http_urllib3_retry_events` metric (rename → `http_retry_events`,
  keep the old name as an alias for one release per docs/wip/wip-concurrent-pipeline-http-retry-metrics.md).
- **Env-mirror still applies** to the residual `huggingface_hub`/pyannote path
  (SDK internals we can't inject).

## API surface (new)

Read/write the outbound block without round-tripping the whole operator YAML:

- `GET /api/network-config` — returns the current `outbound.*` block, redacted
  (proxy password masked, `insecure_mode_flags` echoed), admin-gated.
- `PUT /api/network-config` — validates the block, persists into
  `viewer_operator.yaml` under the `outbound:` key (atomic write), calls
  `OutboundConfigRegistry.swap(...)`, returns the same redacted echo.

Both routes reuse `get_admin_user` (from #1128) and the existing
`operator_config_api_enabled` gate. Persistence goes through the same
`atomic_write_text` helper.
