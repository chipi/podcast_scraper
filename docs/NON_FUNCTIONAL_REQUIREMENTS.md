# Non-Functional Requirements (NFRs)

This document defines **how well** the podcast scraper system must behave: performance, security, reliability, observability, maintainability, and scalability. It complements the [Architecture](ARCHITECTURE.md) (what we build and how it is structured), the [Testing Strategy](TESTING_STRATEGY.md) (how we assure quality), and the [Roadmap](ROADMAP.md) (what is planned). Where the *how* is implemented is left to ADRs, RFCs, PRDs, and CI; this document states the **requirements** and points to those sources.

## Purpose and Scope

- **Audience**: Contributors, operators, and reviewers who need a single place for non-functional expectations.
- **Relationship to other docs**:
  - **[Architecture](ARCHITECTURE.md)**: Describes structure and goals; NFRs define measurable quality attributes.
  - **[Testing Strategy](TESTING_STRATEGY.md)**: Testing validates that NFRs are met (e.g. performance tests, security scans).
  - **[Roadmap](ROADMAP.md)**: Planned work (e.g. RFC-044, PRD-016) often advances or refines NFRs.
  - **ADRs / RFCs / PRDs**: Implement or refine decisions that satisfy NFRs; referenced below where relevant.

### Standards and Practice Alignment

This document aligns with established quality and security practice so that requirements are consistent with industry norms and easier to evaluate.

| Our section | Aligns with | Notes |
| ----------- | ----------- | ----- |
| **1. Performance** | **ISO/IEC 25010** — *Performance efficiency* (time behaviour, resource utilization, capacity) | [ISO/IEC 25010:2023](https://www.iso.org/standard/78176.html) (SQuaRE product quality model). Our timeouts, latency, and memory limits map to time behaviour and resource utilization. |
| **2. Security** | **ISO/IEC 25010** — *Security*; **NIST** secure software development and supply chain guidance | 25010 covers confidentiality, integrity. Dependency scanning, secrets management, and safe parsing align with [NIST Secure Software Development Framework (SSDF)](https://www.nist.gov/itl/executive-order-improving-nations-cybersecurity/software-supply-chain-security-guidance) and supply-chain good practice. |
| **3. Reliability** | **ISO/IEC 25010** — *Reliability* (fault tolerance, recoverability) | Retries, timeouts, deterministic output, and reproducibility map to fault tolerance and recoverability. |
| **4. Observability** | Operational best practice (monitoring, metrics) | Not a distinct 25010 characteristic; often grouped under operability and runbook practice. Our requirements follow common observability practice (instrumentation, metrics, structured logs). |
| **5. Maintainability** | **ISO/IEC 25010** — *Maintainability* (modularity, reusability, analysability) | Code style, test pyramid, documentation, and module boundaries support analysability and modifiability. |
| **6. Scalability** | **ISO/IEC 25010** — *Performance efficiency* (capacity) | Bounded concurrency and resource limits define capacity within single-process scope. |
| **7. Deployment and Operations** | **ISO/IEC 25010** — *Portability* (adaptability, installability) | Local-first, config/env, and diagnostics support installability and adaptability to environments. |
| **8. Privacy** | **ISO/IEC 25010** — *Security* (confidentiality); privacy-by-design practice | Local-first default and opt-in providers support data minimization and user control over where data is sent. |

**References (informative):**

- **ISO/IEC 25010** — Systems and software engineering — Systems and software Quality Requirements and Evaluation (SQuaRE) — Product quality model. Provides a standard set of product quality characteristics and sub-characteristics.
- **NIST SSDF / Software Supply Chain Security** — Guidance on secure development and dependency/vulnerability management. See [NIST Software Supply Chain Security](https://www.nist.gov/itl/executive-order-14028-improving-nations-cybersecurity/software-supply-chain-security-guidance).

### Document Status

- **Status column**: In each requirements table, **Met** = requirement is satisfied today; **Plan** = planned (roadmap); **Part** = partially satisfied.
- **Using this doc**: When writing PRDs or RFCs, call out any new or changed NFRs. When reviewing code or docs, check that tests and CI validate the relevant NFRs.

### Table of Contents

- [Standards and practice alignment](#standards-and-practice-alignment) (industry alignment)
- [1. Performance](#1-performance)
- [2. Security](#2-security)
- [3. Reliability](#3-reliability)
- [4. Observability](#4-observability)
- [5. Maintainability](#5-maintainability)
- [6. Scalability](#6-scalability)
- [7. Deployment and Operations](#7-deployment-and-operations)
- [8. Privacy](#8-privacy)
- [Gaps and limitations](#gaps-and-limitations) (what we are not yet covering)
- [Traceability](#traceability)

## 1. Performance

### 1.1 Policy

The pipeline and its components must complete within bounded time and resource limits so that single-feed runs are predictable and CI remains fast.

### 1.2 Requirements

| Requirement | Expectation | Status | References |
| ----------- | ----------- | ------ | ---------- |
| **Pipeline stage timing** | Transcription and summarization are bounded by configurable timeouts; no unbounded hangs. | Met | [ARCHITECTURE](ARCHITECTURE.md) (Timeout Enforcement), [TROUBLESHOOTING](guides/TROUBLESHOOTING.md) |
| **Per-episode summarization** | Full summarization pipeline &lt; 10 min per episode on target hardware (e.g. Mac laptop). | Met | [RFC-042](rfc/RFC-042-hybrid-summarization-pipeline.md) (Performance Requirements) |
| **Resource / memory** | Full model suite and pipeline stay within documented limits (e.g. &lt; 16 GB on target Mac laptop); sequential ML and MPS exclusive mode avoid unbounded memory growth. | Met | [RFC-042](rfc/RFC-042-hybrid-summarization-pipeline.md), [ADR-001](adr/ADR-001-hybrid-concurrency-strategy.md), [ADR-048](adr/ADR-048-mps-exclusive-mode-apple-silicon.md) |
| **Model registry** | Registry lookups O(1); no noticeable overhead. | Plan | [RFC-044](rfc/RFC-044-model-registry.md) |
| **Local inference** | Acceptable latency for local LLMs (e.g. &lt; 5 min per episode where specified). | Met | [RFC-052](rfc/RFC-052-locally-hosted-llm-models-with-prompts.md), [RFC-042](rfc/RFC-042-hybrid-summarization-pipeline.md) |
| **CI speed** | Fast feedback on push/PR; full CI and nightly runs documented and monitored. | Met | [ADR-017](adr/ADR-017-stratified-ci-execution.md), [CI](ci/index.md) |
| **Disk and storage** | Output and cache writes use operator-controlled paths; write failures (e.g. disk full) are reported clearly and do not corrupt existing data. Cache and output growth are operator-managed (no unbounded automatic growth). | Part | [ARCHITECTURE](ARCHITECTURE.md); cache cleanup and size guidance in operations docs to be strengthened (see [Gaps and limitations](#gaps-and-limitations)) |

### 1.3 Out of Scope (Performance)

- Exact SLAs for third-party APIs (OpenAI, Anthropic, etc.); those are provider-specific and documented in provider RFCs/PRDs.

---

## 2. Security

### 2.1 Policy

Inputs, dependencies, and model loading must be handled in a security-first way to prevent injection, traversal, and supply-chain risks.

### 2.2 Requirements

| Requirement | Expectation | Status | References |
| ----------- | ----------- | ------ | ---------- |
| **RSS/XML parsing** | All RSS/XML parsing uses defusedxml; no unsafe parsing. | Met | [ADR-002](adr/ADR-002-security-first-xml-processing.md) |
| **Path validation** | Path validation prevents directory traversal; user-controlled paths are validated before use. | Met | [ARCHITECTURE](ARCHITECTURE.md) (Reproducibility & Operational Hardening) |
| **Model loading** | Model allowlist validation for HuggingFace sources; safetensors preferred; `trust_remote_code=False` enforced. | Met | [ARCHITECTURE](ARCHITECTURE.md), Issue #379 |
| **Secrets** | No secrets in repo; credentials via environment or config files (not committed). | Met | [ADR-015](adr/ADR-015-secure-credential-injection.md) |
| **Dependency and image scanning** | CI runs security scanning (e.g. CodeQL, Snyk, bandit, pip-audit) on code and dependencies; findings addressed or accepted with justification. | Met | [CI WORKFLOWS](ci/WORKFLOWS.md), `make security` |
| **Vulnerability disclosure** | A documented process for external reporters to report security vulnerabilities and for the project to respond (e.g. acknowledge, triage, fix or decline, disclose). Typically a `SECURITY.md` or equivalent. | Plan | [Gaps and limitations](#gaps-and-limitations) (serious gap until added) |

### 2.3 Out of Scope (Security)

- Formal threat model or penetration testing; those would be separate initiatives.

---

## 3. Reliability

### 3.1 Policy

The pipeline must degrade gracefully under transient failures, respect operator controls (fail-fast, max-failures), and support reproducibility.

### 3.2 Requirements

| Requirement | Expectation | Status | References |
| ----------- | ----------- | ------ | ---------- |
| **Retries** | Transient errors (network, model loading) use exponential backoff retry with configurable counts/delays; HTTP uses retry adapters. | Met | [ADR-044](adr/ADR-044-unified-retry-policy-with-metrics.md), [ARCHITECTURE](ARCHITECTURE.md) |
| **Rate limiting** | External API calls respect provider rate limits; on throttling (e.g. HTTP 429) the system backs off and retries (exponential backoff). No unbounded retry storms. | Met | [ADR-044](adr/ADR-044-unified-retry-policy-with-metrics.md), provider PRDs (e.g. [PRD-010](prd/PRD-010-mistral-provider-integration.md), [PRD-011](prd/PRD-011-deepseek-provider-integration.md)) |
| **Timeouts** | Configurable timeouts for transcription and summarization; no indefinite hangs. | Met | [ARCHITECTURE](ARCHITECTURE.md), [TROUBLESHOOTING](guides/TROUBLESHOOTING.md) |
| **Failure handling** | Operators can set `--fail-fast` and `--max-failures`; episode-level failures are tracked in metrics without masking exit codes. | Met | [ARCHITECTURE](ARCHITECTURE.md) |
| **Exit codes** | Success, config error, and partial/full failure have well-defined exit codes for scripting and orchestration (CLI and service API). | Met | [ARCHITECTURE](ARCHITECTURE.md), [TESTING_STRATEGY](TESTING_STRATEGY.md) (Failure Handling) |
| **Deterministic output** | Output layout is deterministic (hash-based paths per feed/episode); same inputs produce same directory structure. | Met | [ADR-003](adr/ADR-003-deterministic-feed-storage.md), [ADR-004](adr/ADR-004-flat-filesystem-archive-layout.md) |
| **Reproducibility** | Seed-based reproducibility for `torch`, `numpy`, `transformers`; run manifests capture system state (Python, OS, GPU, models, git SHA, config hash). | Met | [ARCHITECTURE](ARCHITECTURE.md), Issue #379 |
| **MPS / GPU** | On Apple Silicon, MPS exclusive mode (default) serializes GPU work to prevent memory contention; configurable. | Met | [ADR-048](adr/ADR-048-mps-exclusive-mode-apple-silicon.md) |
| **Flakiness** | Flaky tests are tracked and reduced; automated retries and health reporting in CI. | Part | [ADR-022](adr/ADR-022-flaky-test-defense.md), [PRD-016](prd/PRD-016-operational-observability-pipeline-intelligence.md) |

---

## 4. Observability

### 4.1 Policy

Operators and developers must be able to understand pipeline health, stage timing, costs, and failures through logs, metrics, and (where implemented) dashboards.

### 4.2 Requirements

| Requirement | Expectation | Status | References |
| ----------- | ----------- | ------ | ---------- |
| **Stage instrumentation** | Major stages (RSS, download, transcription, summarization, etc.) are instrumented; per-episode and aggregate timing available. | Met | [ARCHITECTURE](ARCHITECTURE.md), [PRD-016](prd/PRD-016-operational-observability-pipeline-intelligence.md) |
| **Provider metrics** | Provider calls (e.g. LLM) emit a unified metrics contract (latency, token usage where applicable) for cost and performance analysis. | Met | [ADR-043](adr/ADR-043-unified-provider-metrics-contract.md), [ADR-044](adr/ADR-044-unified-retry-policy-with-metrics.md) |
| **Run manifests and metrics** | Run manifests and pipeline metrics (e.g. `metrics.json`) capture system state and stage outcomes for reproducibility and analysis. | Met | [ARCHITECTURE](ARCHITECTURE.md), [RFC-026](rfc/RFC-026-metrics-consumption-and-dashboards.md) |
| **Structured logging** | Optional structured (JSON) logging for log aggregation systems. | Met | [ARCHITECTURE](ARCHITECTURE.md) (`--json-logs`) |
| **Operational dashboards** | PRD-016 goals: 100% of pipeline stages instrumented and visible; root-cause analysis of CI slowdown in &lt; 5 minutes. | Plan | [PRD-016](prd/PRD-016-operational-observability-pipeline-intelligence.md) |

---

## 5. Maintainability

### 5.1 Policy

Code and documentation must stay understandable, consistent, and measurable so that changes are safe and onboarding is straightforward.

### 5.2 Requirements

| Requirement | Expectation | Status | References |
| ----------- | ----------- | ------ | ---------- |
| **Code style and quality** | Code conforms to project style (black, isort, flake8, mypy); `make format` and `make lint` pass before merge. | Met | [Development Guide](guides/DEVELOPMENT_GUIDE.md), [CI](ci/index.md) |
| **Complexity and maintainability** | Complexity and maintainability (e.g. radon, wily) are tracked; significant degradation is reviewed. | Met | [RFC-031](rfc/RFC-031-code-complexity-analysis-tooling.md), [CI CODE_QUALITY_TRENDS](ci/CODE_QUALITY_TRENDS.md) |
| **Test pyramid and coverage** | Unit, integration, and E2E layers are used consistently; coverage targets (e.g. ≥70%) are enforced in CI. | Met | [ADR-021](adr/ADR-021-standardized-test-pyramid.md), [TESTING_STRATEGY](TESTING_STRATEGY.md) |
| **Documentation** | Architecture, ADRs, PRDs, RFCs, and guides are kept in sync with implementation; markdown and docs build pass in CI. | Met | [ARCHITECTURE](ARCHITECTURE.md), `make docs`, `make fix-md` |
| **Module boundaries** | Public surface and module boundaries are respected (e.g. CLI vs service vs workflow vs config). | Met | `.cursorrules`, [ARCHITECTURE](ARCHITECTURE.md) |
| **Compatibility** | Python 3.10+; public API follows [semantic versioning](api/VERSIONING.md); breaking changes only in MAJOR; deprecations communicated. | Met | [api/VERSIONING](api/VERSIONING.md), [TROUBLESHOOTING](guides/TROUBLESHOOTING.md) |

---

## 6. Scalability

### 6.1 Policy

The system must scale to typical single-feed and multi-episode use cases without architectural rewrites; scaling beyond that (e.g. many feeds, very large episode counts) may require planned work (e.g. GIL database projection).

### 6.2 Requirements

| Requirement | Expectation | Status | References |
| ----------- | ----------- | ------ | ---------- |
| **Single-feed, many episodes** | Pipeline handles many episodes per feed within resource limits (timeouts, memory); sequential ML per ADR-001. | Met | [ADR-001](adr/ADR-001-hybrid-concurrency-strategy.md), [ARCHITECTURE](ARCHITECTURE.md) |
| **Concurrency** | IO-bound work (downloads) uses threading; ML work is sequential to avoid GPU OOM and contention. | Met | [ADR-001](adr/ADR-001-hybrid-concurrency-strategy.md), [ADR-048](adr/ADR-048-mps-exclusive-mode-apple-silicon.md) |
| **GIL / query scale** | File-based GIL scales to moderate episode counts; database projection (PRD-018, RFC-051) is the path for fast cross-episode queries at scale. | Plan | [RFC-051](rfc/RFC-051-grounded-insight-layer-database-projection.md), [RFC-050](rfc/RFC-050-grounded-insight-layer-use-cases.md) |

### 6.3 Out of Scope (Scalability)

- Distributed or multi-node execution; current design is single-process, single-machine.

---

## 7. Deployment and Operations

### 7.1 Policy

The system runs locally or in operator-controlled environments with minimal required external services; optional features (e.g. cloud providers) are opt-in via config.

### 7.2 Requirements

| Requirement | Expectation | Status | References |
| ----------- | ----------- | ------ | ---------- |
| **Local-first** | Core pipeline (RSS, download, local Whisper, local summarization) runs without mandatory external APIs. | Met | [ADR-009](adr/ADR-009-privacy-first-local-summarization.md), [ARCHITECTURE](ARCHITECTURE.md) |
| **Config and env** | Configuration via file and/or environment variables; no hardcoded secrets or environment-specific paths in repo. | Met | [ADR-015](adr/ADR-015-secure-credential-injection.md), [Configuration](api/CONFIGURATION.md) |
| **Diagnostics** | `podcast_scraper doctor` (or equivalent) validates environment (Python, ffmpeg, permissions, model cache, network) for operator troubleshooting. | Met | [ARCHITECTURE](ARCHITECTURE.md) |
| **Releases** | Pre-release validation and versioning follow project standards; release notes reference PRDs/RFCs. | Met | [ADR-041](adr/ADR-041-mandatory-pre-release-validation.md), [Releases](releases/index.md) |

---

## 8. Privacy

### 8.1 Policy

The default pipeline does not require sending data to third parties; cloud and API providers are opt-in. Users can run the full core workflow (RSS, download, local Whisper, local summarization) without exposing content to external services.

### 8.2 Requirements

| Requirement | Expectation | Status | References |
| ----------- | ----------- | ------ | ---------- |
| **Local-first default** | Summarization and transcription can run entirely locally; no mandatory external API calls for core pipeline. | Met | [ADR-009](adr/ADR-009-privacy-first-local-summarization.md), [ARCHITECTURE](ARCHITECTURE.md) |
| **Opt-in providers** | Use of OpenAI, Anthropic, Mistral, and other API providers is explicit via config; no data is sent to third parties unless the user configures a provider. | Met | [ADR-049](adr/ADR-049-per-capability-provider-selection.md), [Configuration](api/CONFIGURATION.md) |

---

## Gaps and Limitations

This section states what the NFR document and the project currently **do not** fully cover, so that reviewers and contributors can see the limits and decide what to add over time.

### Serious gaps (should be addressed)

| Gap | Description | Recommendation |
| --- | ----------- | ---------------- |
| **Vulnerability disclosure** | There is no stated process for external reporters to report security vulnerabilities (e.g. a `SECURITY.md` or policy). CI does dependency and code scanning; we do not yet document how to report issues or how the project will respond. | Add a `SECURITY.md` (or equivalent) describing how to report vulnerabilities and expected response (e.g. acknowledge, triage, fix or decline, disclose). Aligns with NIST SSDF and common open-source practice. |
| **Rate limiting / backpressure** | Provider PRDs require handling API rate limits (retry with backoff); behaviour is implemented per provider. The NFR doc does not state once that the system must respect external rate limits and back off on throttling (e.g. 429). | Add an explicit requirement under **Reliability** (or **Performance**): e.g. "External API calls respect provider rate limits; on throttling (e.g. HTTP 429) the system backs off and retries with exponential backoff." Reference ADR-044 and provider RFCs. |
| **Disk and storage** | No requirement that the pipeline fails gracefully when disk is full or that cache/output growth is bounded or operator-managed. Long runs or large caches could fill disk. | Add under **Performance** or **Reliability**: e.g. "Writes to output and cache directories are bounded by operator-controlled paths and capacity; pipeline reports write failures (e.g. disk full) clearly rather than corrupting data." Optionally document cache cleanup or size guidance in operations docs. |

### Gaps that are acceptable for now (documented as out of scope or deferred)

| Gap | Description | Why it is acceptable (for now) |
| --- | ----------- | ------------------------------ |
| **NFR–verification matrix** | There is no single table mapping each NFR requirement to the specific test(s) or CI job(s) that verify it. Traceability points to Testing Strategy and CI in general. | Acceptable for current scale; an explicit matrix would help audits and onboarding. Can be added when needed (e.g. in this doc or in TESTING_STRATEGY). |
| **Data retention and deletion** | No stated policy for retention of run manifests, cache, or metrics, or for "user data deletion." | Local-first tool: data lives under operator-controlled paths; operator can delete. If the product gains hosted/CI artifacts or cloud features, define retention and deletion then. |
| **License and dependency compliance** | No NFR that dependencies and distribution comply with the project license (e.g. avoiding incompatible licenses). | Often handled by policy and tooling (e.g. license checkers in CI) rather than NFR; can be added if the project needs to attest to license compliance. |
| **Formal threat model** | Already listed as out of scope under Security. | Appropriate for current scope; revisit if the product is used in regulated or high-assurance environments. |

### Not gaps (already covered elsewhere)

- **Resume / partial failure**: Idempotent runs and `--skip-existing` are in Architecture; no separate NFR needed.
- **Usability (ISO 25010)**: Clear exit codes, diagnostics, and docs cover basic operability; no separate Usability section for the CLI/library.

---

## Traceability

- **New features**: PRDs and RFCs should call out any new or changed NFRs (e.g. "RFC-044: registry lookups O(1)").
- **Tests**: [Testing Strategy](TESTING_STRATEGY.md) and test guides define how NFRs are validated (unit, integration, E2E, security scans, performance checks). When adding or changing behavior, ensure the relevant NFR rows are still satisfied and update **Status** (Met/Plan/Part) when implementation ships.
- **Roadmap**: [Roadmap](ROADMAP.md) items (e.g. RFC-054, RFC-044, PRD-016) often advance observability, reliability, or performance; priorities are set there.
- **Review**: When closing an RFC or shipping a feature that affects an NFR, update this document (new requirement, status change, or reference).

---

**Last Updated**: 2026-02-10  
**Next Review**: When major NFR-related work ships (e.g. model registry, observability platform) or at least annually.
