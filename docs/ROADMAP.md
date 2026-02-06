# Roadmap

This document provides an overview of all open Product Requirements Documents (PRDs) and Requests for Comment (RFCs), prioritized by end-user value and impact. The roadmap helps stakeholders understand what's coming, why it matters, and when to expect it.

## Overview

The roadmap is organized around **end-user value** as the central principle. Features that directly improve user experience, reduce costs, or enhance quality are prioritized over infrastructure improvements. However, infrastructure work (testing, observability, developer experience) is essential for maintaining quality and velocity.

### Current Status

- **Open PRDs**: 9 (provider integrations, knowledge graph, experimentation platform)
- **Open RFCs**: 26 (provider implementations, quality improvements, infrastructure)
- **Completed**: 7 PRDs, 22 RFCs (core pipeline, basic features, OpenAI integration)

## Categories

### 1. Direct User Features (High Impact)

These features directly improve what users can do with the tool:

- **Provider Integrations**: More provider options (Anthropic, Mistral, Gemini, Grok, DeepSeek, Ollama) give users choice, cost optimization, and quality options
- **Summarization Improvements**: Better summaries through adaptive routing, hybrid pipelines, and optimization guides
- **Audio Preprocessing**: Cost savings and quality improvements through VAD, normalization, and format optimization
- **Knowledge Graph**: Structured knowledge extraction enables advanced querying and analysis

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
| **6** | [PRD-012](prd/PRD-012-gemini-provider-integration.md) / [RFC-035](rfc/RFC-035-gemini-provider-implementation.md) | PRD/RFC | Provider Integration | High | Medium | Benefits from RFC-044 | **Unique capabilities**: 2M context window, native audio transcription. Enables processing very long episodes. Medium effort, high value. |
| **7** | [PRD-014](prd/PRD-014-ollama-provider-integration.md) / [RFC-037](rfc/RFC-037-ollama-provider-implementation.md) | PRD/RFC | Provider Integration | High | Medium | Benefits from RFC-044 | **Zero cost option**: Fully local/offline provider. Appeals to privacy-conscious users. Medium effort, high value for cost-sensitive users. |
| **8** | [PRD-011](prd/PRD-011-deepseek-provider-integration.md) / [RFC-034](rfc/RFC-034-deepseek-provider-implementation.md) | PRD/RFC | Provider Integration | Medium-High | Medium | Benefits from RFC-044 | **Ultra low-cost**: DeepSeek offers very low pricing. Cost optimization for users processing many episodes. Medium effort, medium-high value. |
| **9** | [PRD-010](prd/PRD-010-mistral-provider-integration.md) / [RFC-033](rfc/RFC-033-mistral-provider-implementation.md) | PRD/RFC | Provider Integration | Medium | Medium | Benefits from RFC-044 | **OpenAI alternative**: Complete provider with all capabilities. Vendor diversity. Medium effort, medium value. |
| **10** | [PRD-009](prd/PRD-009-anthropic-provider-integration.md) / [RFC-032](rfc/RFC-032-anthropic-provider-implementation.md) | PRD/RFC | Provider Integration | Medium | Medium | Benefits from RFC-044 | **High-quality option**: Anthropic Claude offers strong quality. Vendor diversity. Medium effort, medium value. |
| **11** | [PRD-013](prd/PRD-013-grok-provider-integration.md) / [RFC-036](rfc/RFC-036-grok-provider-implementation.md) | PRD/RFC | Provider Integration | Medium | Medium | Benefits from RFC-044 | **Real-time info**: Grok has X/Twitter integration. Niche but valuable for current events. Medium effort, medium value. |
| **12** | [RFC-045](rfc/RFC-045-ml-model-optimization-guide.md) | RFC | Quality Guide | Medium | Low | None | **User education**: Comprehensive guide helps users maximize ML quality. Low effort (documentation), medium value. |
| **13** | [RFC-052](rfc/RFC-052-locally-hosted-llm-models-with-prompts.md) | RFC | Provider Integration | Medium | Medium | Complements PRD-014 | **Cost & latency**: Locally hosted LLMs solve cost and latency issues. Complements Ollama. Medium effort, medium value. |
| **14** | [RFC-027](rfc/RFC-027-pipeline-metrics-improvements.md) | RFC | Observability | Medium | Medium | None | **Operational visibility**: Better metrics collection and reporting. Helps users understand costs, performance, quality. Medium effort, medium value. |
| **15** | [PRD-016](prd/PRD-016-operational-observability-pipeline-intelligence.md) | PRD | Observability | Medium | High | None | **System health**: Operational observability and pipeline intelligence. High effort, medium value. |
| **16** | [RFC-043](rfc/RFC-043-automated-metrics-alerts.md) | RFC | Observability | Medium | Medium | Depends on RFC-027 | **Regression detection**: Automated alerts for quality regressions. Depends on metrics improvements. Medium effort, medium value. |
| **17** | [RFC-047](rfc/RFC-047-run-comparison-visual-tool.md) | RFC | Developer Tool | Medium | Medium | None | **Quality analysis**: Visual tool for comparing runs and diagnosing regressions. Medium effort, medium value. |
| **18** | [PRD-017](prd/PRD-017-podcast-knowledge-graph.md) / [RFC-049](rfc/RFC-049-podcast-knowledge-graph-core.md), [RFC-050](rfc/RFC-050-podcast-knowledge-graph-use-cases.md) | PRD/RFC | Advanced Feature | Medium | High | None | **Structured knowledge**: Enables advanced querying and analysis. High effort, medium value (advanced use case). |
| **19** | [PRD-018](prd/PRD-018-database-export-knowledge-graph.md) / [RFC-051](rfc/RFC-051-database-export-knowledge-graph.md) | PRD/RFC | Advanced Feature | Medium | Medium | Depends on PRD-017 | **Fast queries**: Database export for knowledge graph. Depends on PRD-017. Medium effort, medium value. |
| **20** | [PRD-007](prd/PRD-007-ai-quality-experiment-platform.md) / [RFC-015](rfc/RFC-015-ai-experiment-pipeline.md), [RFC-016](rfc/RFC-016-modularization-for-ai-experiments.md), [RFC-041](rfc/RFC-041-podcast-ml-benchmarking-framework.md) | PRD/RFC | Advanced Platform | Low-Medium | High | None | **Experimentation**: Platform for AI quality experimentation. High effort, low-medium value (advanced use case). |
| **21** | [RFC-046](rfc/RFC-046-materialization-architecture.md) | RFC | Infrastructure | Low-Medium | Medium | Depends on PRD-007 | **Honest comparisons**: Materialization architecture. Depends on experimentation platform. Medium effort, low-medium value. |
| **22** | [RFC-048](rfc/RFC-048-evaluation-application-alignment.md) | RFC | Infrastructure | Low-Medium | Medium | Depends on PRD-007 | **Alignment**: Ensures evaluation results represent application behavior. Depends on experimentation platform. Medium effort, low-medium value. |
| **23** | [RFC-023](rfc/RFC-023-readme-acceptance-tests.md) | RFC | Documentation | Low | Low | None | **Documentation quality**: Acceptance tests for README accuracy. Low effort, low value. |
| **24** | [PRD-015](prd/PRD-015-engineering-governance-productivity.md) | PRD | Developer Experience | Low | High | None | **Developer velocity**: Engineering governance and productivity platform. High effort, low value (indirect user impact). |
| **25** | [RFC-038](rfc/RFC-038-continuous-review-tooling.md) | RFC | Developer Tool | Low | Low | None | **Code quality**: Continuous review tooling (Dependabot, pydeps). Low effort, low value (indirect user impact). |

## Priority Rationale

### Top 5 Priorities

1. **RFC-054 (E2E Mock Response Strategy)**: **Critical blocker** - Enables comprehensive error testing. Blocks provider hardening work (#399). High effort but essential infrastructure.

2. **RFC-040 (Audio Preprocessing)**: **High impact, low effort** - Well-defined scope, clear implementation path. Immediate cost savings (10-30% reduction) and quality improvements. No dependencies.

3. **RFC-044 (Model Registry)**: **Low effort, high value** - Centralized change that eliminates technical debt and enables easier provider additions. Low effort because it's a refactoring task with clear scope.

4. **RFC-053 (Adaptive Summarization Routing)**: **Quality improvement** - Episode-specific routing improves summary quality. Medium effort, high impact on user experience. No dependencies.

5. **RFC-042 (Hybrid Summarization Pipeline)**: **Quality improvement** - Addresses persistent summary quality issues. Higher effort but high impact. No dependencies.

### Provider Integration Priority

Provider integrations are prioritized by:

- **Unique capabilities** (Gemini: 2M context, native audio)
- **Cost optimization** (Ollama: zero cost, DeepSeek: ultra low-cost)
- **Quality options** (Anthropic: high quality, Mistral: complete alternative)
- **Niche value** (Grok: real-time info)

All provider integrations benefit from RFC-044 (Model Registry) but are not blocked by it.

## Implementation Status

### In Progress

- Provider metrics tracking (recently completed)
- Provider hardening (retry policies, timeouts - in progress via #399)

### Planned

- Audio preprocessing (RFC-040)
- Adaptive summarization routing (RFC-053)
- Provider integrations (various PRDs/RFCs)

### Future

- Knowledge graph (PRD-017)
- Experimentation platform (PRD-007)
- Advanced observability (PRD-016)

## Related Documents

- **[PRDs](prd/index.md)** - Product requirements documents
- **[RFCs](rfc/index.md)** - Technical design documents
- **[Architecture](ARCHITECTURE.md)** - System design and module responsibilities
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
- Provider integrations (Gemini, Ollama, DeepSeek) - Similar patterns, medium effort each

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

**Last Updated**: 2026-02-05
**Next Review**: Quarterly (or as priorities shift)
