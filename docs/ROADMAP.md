# Roadmap

This document provides an overview of all open Product Requirements Documents (PRDs) and Requests for Comment (RFCs), prioritized by end-user value and impact. The roadmap helps stakeholders understand what's coming, why it matters, and when to expect it.

## Overview

The roadmap is organized around **end-user value** as the central principle. Features that directly improve user experience, reduce costs, or enhance quality are prioritized over infrastructure improvements. However, infrastructure work (testing, observability, developer experience) is essential for maintaining quality and velocity.

### Current Status

- **Open PRDs**: 5 (knowledge graph, experimentation platform, observability, engineering governance)
- **Open RFCs**: 19 (quality improvements, GIL, model registry, observability, infrastructure)
- **Completed**: 13 PRDs, 28 RFCs (core pipeline, 8-provider ecosystem v2.5, OpenAI + Anthropic + Mistral + DeepSeek + Gemini + Grok + Ollama, modularization, metrics)

## Categories

### 1. Direct User Features (High Impact)

These features directly improve what users can do with the tool:

- **Provider Ecosystem**: Complete (v2.5) — 8 providers (local ML, OpenAI, Anthropic, Mistral, DeepSeek, Gemini, Grok, Ollama) for choice, cost, and quality
- **Summarization Improvements**: Better summaries through adaptive routing (RFC-053), hybrid pipeline (RFC-042), and optimization guides (RFC-045)
- **Audio Preprocessing**: Cost savings and quality improvements through VAD, normalization, and format optimization (RFC-040)
- **Knowledge Graph (GIL)**: Structured knowledge extraction (PRD-017, RFC-049, 050, 051) enables advanced querying and analysis

### 2. Quality & Reliability (Medium-High Impact)

These features improve output quality and system reliability:

- **Adaptive Summarization Routing**: Episode-specific strategies for better summaries
- **Hybrid Summarization Pipeline**: Better quality through instruction-tuned LLMs
- **Model Registry**: Eliminates hardcoded limits, enables easier model updates
- **Provider Hardening**: Unified retry policies, timeouts, capability contracts (RFC-054, #399)

### 3. Developer Experience & Infrastructure (Medium Impact)

These features improve maintainability, testing, and operational visibility:

- **Testing Infrastructure**: Better E2E mocks (RFC-054), comprehensive test coverage
- **Observability**: Structured logs, metrics, alerts (RFC-027, #401)
- **Engineering Tools**: Continuous review, code quality, benchmarking frameworks

### 4. Advanced Features (Lower Priority)

These features enable advanced use cases but are not essential for most users:

- **Knowledge Graph**: Structured knowledge extraction (PRD-017, RFC-049, 050, 051)
- **Experimentation Platform**: AI quality experimentation and benchmarking (PRD-007, RFC-015, 016, 041)
- **Database Export**: Fast queryable exports for knowledge graph (PRD-018, RFC-051)

## Prioritized Roadmap

The following table lists all open PRDs and RFCs, ordered by priority with most pressing items at the top. Priority is determined by:

1. **Direct user value**: Features users directly interact with or benefit from
2. **Cost impact**: Features that reduce API costs or improve efficiency
3. **Quality impact**: Features that improve output quality or reliability
4. **Infrastructure needs**: Features that enable other work or prevent issues
5. **Dependency analysis**: Items that block other work are prioritized higher
6. **Effort vs Impact**: High-impact, low-effort items are prioritized over high-effort, low-impact items

| Priority | Item | Type | User Value | Impact | Effort | Dependencies | Reasoning |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **1** | [RFC-054](rfc/RFC-054-e2e-mock-response-strategy.md) | RFC | Testing Infrastructure | High | High | Blocks #399 | **Critical blocker**: Enables comprehensive error testing (rate limits, retries, timeouts). Blocks provider hardening work (#399). Without this, we can't confidently test provider reliability. |
| **2** | [RFC-040](rfc/RFC-040-audio-preprocessing-pipeline.md) | RFC | Direct Feature | High | Low | None | **High impact, low effort**: VAD removes silence (reduces API costs 10-30%), normalization improves quality, format optimization prevents upload failures. Well-defined scope, clear implementation path. |
| **3** | [RFC-044](rfc/RFC-044-model-registry.md) | RFC | Infrastructure | Medium-High | Low | Enables provider additions | **Low effort, high value**: Eliminates technical debt, enables easier model updates and provider additions. Centralized change, prevents bugs. |
| **4** | [RFC-053](rfc/RFC-053-adaptive-summarization-routing.md) | RFC | Quality Improvement | High | Medium | None | **Quality improvement**: Episode-specific routing improves summary quality across diverse podcast types. Medium effort, high impact on user experience. |
| **5** | [RFC-042](rfc/RFC-042-hybrid-summarization-pipeline.md) | RFC | Quality Improvement | High | High | None | **Quality improvement**: Instruction-tuned LLMs in REDUCE phase produce better structured summaries. Addresses persistent quality issues. Higher effort but high impact. |
| **6** | [RFC-045](rfc/RFC-045-ml-model-optimization-guide.md) | RFC | Quality Guide | Medium | Low | None | **User education**: Comprehensive guide helps users maximize ML quality. Low effort (documentation), medium value. |
| **7** | [RFC-052](rfc/RFC-052-locally-hosted-llm-models-with-prompts.md) | RFC | Provider Enhancement | Medium | Medium | Complements Ollama | **Cost & latency**: Locally hosted LLMs with optimized prompts. Complements existing Ollama provider. Medium effort, medium value. |
| **8** | [RFC-027](rfc/RFC-027-pipeline-metrics-improvements.md) | RFC | Observability | Medium | Medium | None | **Operational visibility**: Better metrics collection and reporting. Helps users understand costs, performance, quality. Medium effort, medium value. |
| **9** | [PRD-016](prd/PRD-016-operational-observability-pipeline-intelligence.md) | PRD | Observability | Medium | High | None | **System health**: Operational observability and pipeline intelligence. High effort, medium value. |
| **10** | [RFC-043](rfc/RFC-043-automated-metrics-alerts.md) | RFC | Observability | Medium | Medium | Depends on RFC-027 | **Regression detection**: Automated alerts for quality regressions. Depends on metrics improvements. Medium effort, medium value. |
| **11** | [RFC-047](rfc/RFC-047-run-comparison-visual-tool.md) | RFC | Developer Tool | Medium | Medium | None | **Quality analysis**: Visual tool for comparing runs and diagnosing regressions. Medium effort, medium value. |
| **12** | [PRD-017](prd/PRD-017-grounded-insight-layer.md) / [RFC-049](rfc/RFC-049-grounded-insight-layer-core.md), [RFC-050](rfc/RFC-050-grounded-insight-layer-use-cases.md) | PRD/RFC | Advanced Feature | Medium | High | None | **Grounded Insight Layer**: Evidence-backed insights and quotes with grounding relationships. High effort, high value (trust + navigation). |
| **13** | [PRD-018](prd/PRD-018-database-projection-grounded-insight-layer.md) / [RFC-051](rfc/RFC-051-grounded-insight-layer-database-projection.md) | PRD/RFC | Advanced Feature | Medium | Medium | Depends on PRD-017 | **Fast queries**: Database projection for Grounded Insight Layer. Depends on PRD-017. Medium effort, medium value. |
| **14** | [PRD-007](prd/PRD-007-ai-quality-experiment-platform.md) / [RFC-015](rfc/RFC-015-ai-experiment-pipeline.md), [RFC-041](rfc/RFC-041-podcast-ml-benchmarking-framework.md) | PRD/RFC | Advanced Platform | Low-Medium | High | None | **Experimentation**: Platform for AI quality experimentation (RFC-016 modularization complete). High effort, low-medium value (advanced use case). |
| **15** | [RFC-046](rfc/RFC-046-materialization-architecture.md) | RFC | Infrastructure | Low-Medium | Medium | Depends on PRD-007 | **Honest comparisons**: Materialization architecture. Depends on experimentation platform. Medium effort, low-medium value. |
| **16** | [RFC-048](rfc/RFC-048-evaluation-application-alignment.md) | RFC | Infrastructure | Low-Medium | Medium | Depends on PRD-007 | **Alignment**: Ensures evaluation results represent application behavior. Depends on experimentation platform. Medium effort, low-medium value. |
| **17** | [RFC-023](rfc/RFC-023-readme-acceptance-tests.md) | RFC | Documentation | Low | Low | None | **Documentation quality**: Acceptance tests for README accuracy. Low effort, low value. |
| **18** | [PRD-015](prd/PRD-015-engineering-governance-productivity.md) | PRD | Developer Experience | Low | High | None | **Developer velocity**: Engineering governance and productivity platform. High effort, low value (indirect user impact). |
| **19** | [RFC-038](rfc/RFC-038-continuous-review-tooling.md) | RFC | Developer Tool | Low | Low | None | **Code quality**: Continuous review tooling (Dependabot, pydeps). Low effort, low value (indirect user impact). |

## Priority Rationale

### Top 5 Priorities

1. **RFC-054 (E2E Mock Response Strategy)**: **Critical blocker** - Enables comprehensive error testing. Blocks provider hardening work (#399). High effort but essential infrastructure.

2. **RFC-040 (Audio Preprocessing)**: **High impact, low effort** - Well-defined scope, clear implementation path. Immediate cost savings (10-30% reduction) and quality improvements. No dependencies.

3. **RFC-044 (Model Registry)**: **Low effort, high value** - Centralized change that eliminates technical debt and enables easier provider additions. Low effort because it's a refactoring task with clear scope.

4. **RFC-053 (Adaptive Summarization Routing)**: **Quality improvement** - Episode-specific routing improves summary quality. Medium effort, high impact on user experience. No dependencies.

5. **RFC-042 (Hybrid Summarization Pipeline)**: **Quality improvement** - Addresses persistent summary quality issues. Higher effort but high impact. No dependencies.

### Provider Ecosystem (Complete)

The 8-provider ecosystem is complete as of v2.5.0: local ML, OpenAI, Anthropic, Mistral, DeepSeek, Gemini, Grok, and Ollama. Remaining provider-related work is enhancement (RFC-052: locally hosted LLMs with prompts) and infrastructure (RFC-044: model registry). New provider additions benefit from RFC-044 but are not on the near-term roadmap.

## Implementation Status

### Recently Completed (v2.4–v2.5)

- **Provider ecosystem**: Anthropic (PRD-009, RFC-032), Mistral (PRD-010, RFC-033), DeepSeek (PRD-011, RFC-034), Gemini (PRD-012, RFC-035), Grok (PRD-013, RFC-036), Ollama (PRD-014, RFC-037)
- **Modularization**: RFC-016 (modularization for AI experiments) — protocol-based providers, typed params, preprocessing profiles
- **Production hardening**: MPS exclusive mode (Apple Silicon), run manifests, entity reconciliation, unified provider metrics (ADR-043, ADR-044, ADR-048, ADR-049)

### In Progress

- Provider hardening (retry policies, timeouts — #399); blocked by RFC-054 for full E2E error testing

### Planned (Ahead)

- **High priority**: RFC-054 (E2E mocks), RFC-040 (audio preprocessing), RFC-044 (model registry), RFC-053 (adaptive routing), RFC-042 (hybrid pipeline)
- **Medium**: RFC-045 (ML optimization guide), RFC-052 (locally hosted LLMs), observability (RFC-027, 043, PRD-016), run comparison (RFC-047)
- **Advanced**: Grounded Insight Layer (PRD-017, RFC-049/050/051), database projection (PRD-018), experimentation platform (PRD-007, RFC-015/041)

### Future

- Full experimentation platform (RFC-046, RFC-048)
- Engineering governance (PRD-015)
- Documentation and tooling (RFC-023, RFC-038)

## Related Documents

- **[PRDs](prd/index.md)** - Product requirements documents
- **[RFCs](rfc/index.md)** - Technical design documents
- **[Architecture](ARCHITECTURE.md)** - System design and module responsibilities
- **[Non-Functional Requirements](NON_FUNCTIONAL_REQUIREMENTS.md)** - Performance, security, reliability, observability
- **[Releases](releases/index.md)** - Release notes and version history

---

## Prioritization Methodology

This section documents how priorities are determined for items in the roadmap.

### Core Principles

1. **End-User Value First**: Features that directly improve user experience, reduce costs, or enhance quality are prioritized over infrastructure improvements.
2. **Dependency-Aware**: Items that block other work are prioritized higher, even if they have lower direct user value.
3. **Effort vs Impact**: High-impact, low-effort items are prioritized over high-effort, low-impact items.

### Evaluation Criteria

Each item is evaluated on four factors:

1. **Direct user value**: Features users directly interact with or benefit from
2. **Cost impact**: Features that reduce API costs or improve efficiency
3. **Quality impact**: Features that improve output quality or reliability
4. **Infrastructure needs**: Features that enable other work or prevent issues

### Dependency Graph Analysis

Dependencies are identified and used to adjust priorities:

**Critical Blockers** (must be done first):

- RFC-054 blocks #399 (provider hardening) - **Priority 1**

**Enablers** (make other work easier, but not blocking):

- RFC-044 enables easier provider additions - **Priority 3** (done early to benefit providers)
- RFC-027 enables RFC-043 (metrics alerts) - RFC-043 depends on RFC-027
- PRD-017 enables PRD-018 (database export) - PRD-018 depends on PRD-017
- PRD-007 enables RFC-046 and RFC-048 (experimentation platform dependencies)

**Independent Items** (no dependencies):

- RFC-040, RFC-053, RFC-042, RFC-045, most provider integrations

### Effort vs Impact Matrix

Items are categorized by effort (Low/Medium/High) and impact (Low/Medium/High):

**High Impact, Low Effort** (Prioritize First):

- RFC-040 (Audio Preprocessing) - Well-defined scope, clear implementation
- RFC-044 (Model Registry) - Centralized refactoring task
- RFC-045 (ML Optimization Guide) - Documentation work

**High Impact, Medium Effort** (Prioritize Second):

- RFC-053 (Adaptive Summarization Routing) - Medium complexity, high value
- RFC-042 (Hybrid Summarization Pipeline) - Higher complexity but addresses quality issues
- Provider integrations (Gemini, Ollama, DeepSeek, Mistral, Anthropic, Grok) - **Completed** in v2.5

**High Impact, High Effort** (Prioritize Third):

- RFC-054 (E2E Mock Response Strategy) - Complex infrastructure but critical blocker
- PRD-016 (Operational Observability) - Comprehensive system, high effort
- PRD-017 (Knowledge Graph) - Advanced feature, high effort

**Medium Impact, Low Effort** (Prioritize Fourth):

- RFC-023 (README Acceptance Tests) - Low effort, maintains quality

**Medium Impact, Medium Effort** (Prioritize Fifth):

- RFC-027 (Pipeline Metrics) - Medium effort, operational value
- RFC-043 (Automated Metrics Alerts) - Depends on RFC-027
- RFC-047 (Run Comparison Tool) - Medium effort, quality analysis

**Low Impact, High Effort** (Prioritize Last):

- PRD-015 (Engineering Governance) - High effort, indirect user impact
- PRD-007 (Experimentation Platform) - High effort, advanced use case

**Low Impact, Low Effort** (Prioritize Last):

- RFC-038 (Continuous Review Tooling) - Low effort, indirect user impact

### Priority Calculation

Final priority is determined by:

1. **Dependency analysis**: Blockers get highest priority regardless of other factors
2. **Effort vs Impact**: High-impact, low-effort items prioritized over high-effort, low-impact
3. **User value**: Direct user features prioritized over infrastructure (unless blocking)
4. **Cost/Quality impact**: Items with immediate cost savings or quality improvements prioritized

### Example: RFC-040 (Audio Preprocessing)

- **User Value**: High (direct cost savings, quality improvement)
- **Cost Impact**: High (10-30% API cost reduction)
- **Quality Impact**: Medium (normalization improves quality)
- **Infrastructure Needs**: Low (independent feature)
- **Dependencies**: None
- **Effort**: Low (well-defined scope, clear implementation path)
- **Impact**: High (affects all API transcription users)
- **Result**: Priority 2 (after critical blocker RFC-054)

### Example: RFC-054 (E2E Mock Response Strategy)

- **User Value**: Low (testing infrastructure)
- **Cost Impact**: None
- **Quality Impact**: Medium (enables better testing)
- **Infrastructure Needs**: High (blocks #399)
- **Dependencies**: Blocks #399 (provider hardening)
- **Effort**: High (complex testing infrastructure)
- **Impact**: High (critical for reliability)
- **Result**: Priority 1 (critical blocker, must be done first)

### Review Process

Priorities are reviewed quarterly or when:

- New dependencies are identified
- Effort estimates change significantly
- User feedback indicates different priorities
- Implementation reveals new blockers or enablers

---

**Last Updated**: 2026-02-10
**Next Review**: Quarterly (or as priorities shift)
