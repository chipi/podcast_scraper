# RFC-NNN: [Feature Name]

- **Status**: Draft | Accepted | Completed
- **Authors**: [Author names]
- **Stakeholders**: [Who needs to review/approve this]
- **Related PRDs**:
  - `docs/prd/PRD-XXX-feature-name.md` (if applicable)
- **Related ADRs**:
  - `docs/adr/ADR-NNN-decision-name.md` (if applicable)
- **Related RFCs**:
  - `docs/rfc/RFC-XXX-related-feature.md` (if applicable)
- **Related Documents**:
  - `docs/ARCHITECTURE.md` (if architecture changes)
  - `docs/TESTING_STRATEGY.md` (if testing changes)

## Abstract

[2-3 sentences summarizing:]

- [What this RFC proposes]
- [Why it's needed]
- [Key technical approach]

**Architecture Alignment:** [If applicable, describe how this aligns with existing architecture or RFC-016 modularization principles.]

## Problem Statement

[2-3 paragraphs explaining:]

- [What problem we're solving]
- [Why current approach is insufficient]
- [What gaps exist]
- [Impact of not solving this]

**Use Cases:**

1. **[Use Case 1]**: [Description]
2. **[Use Case 2]**: [Description]
3. **[Use Case 3]**: [Description]

## Goals

[Bulleted list of technical goals:]

1. **[Goal 1]**: [Technical objective]
2. **[Goal 2]**: [Technical objective]
3. **[Goal 3]**: [Technical objective]

## Constraints & Assumptions

**Constraints:**

- [Technical constraint 1, e.g., "Must not hit external networks"]
- [Performance constraint, e.g., "Must complete in < 5s"]
- [Compatibility constraint, e.g., "Must be backward compatible"]

**Assumptions:**

- [Assumption 1, e.g., "Local HTTP server is sufficient for HTTP testing"]
- [Assumption 2, e.g., "Small ML models are acceptable for integration tests"]

## Design & Implementation

[Main technical design section. Organize by component or phase as needed.]

### 1. [Component/Phase Name]

[Description of design and implementation approach:]

- [Design decision 1]
- [Design decision 2]
- [Design decision 3]

**Example code or structure:**

````python

# Example code showing key design patterns

def example_function():
    """Example implementation."""
    pass
```text

component/
  module1.py
  module2.py

```text

## 3. [Integration Points]

[How this integrates with existing systems:]

- **[Integration Point 1]**: [How it integrates]
- **[Integration Point 2]**: [How it integrates]

## Key Decisions

[Document important design decisions and rationale:]

1. **[Decision Topic]**
   - **Decision**: [What was decided]
   - **Rationale**: [Why this decision was made]

2. **[Another Decision]**
   - **Decision**: [What was decided]
   - **Rationale**: [Why this decision was made]

## Alternatives Considered

[Document alternatives that were considered but rejected:]

1. **[Alternative Approach]**
   - **Description**: [What this alternative was]
   - **Pros**: [Benefits of this approach]
   - **Cons**: [Drawbacks of this approach]
   - **Why Rejected**: [Reason for not choosing this]

2. **[Another Alternative]**
   - [Similar structure...]

## Testing Strategy

[How this feature will be tested:]

**Test Coverage:**
- **[Test Category 1]**: [Description, e.g., "Unit tests for core functionality"]
- **[Test Category 2]**: [Description, e.g., "Integration tests with real HTTP server"]
- **[Test Category 3]**: [Description, e.g., "E2E tests for complete workflows"]

**Test Organization:**
- [Where tests will be located]
- [What markers will be used]
- [What fixtures are needed]

**Test Execution:**
- [When tests run in CI]
- [What test data is needed]
- [Performance considerations]

## Rollout & Monitoring

**Rollout Plan:**
- [Phase 1]: [Description and timeline]
- [Phase 2]: [Description and timeline]
- [Phase 3]: [Description and timeline]

**Monitoring:**
- [What metrics to track]
- [How to detect issues]
- [What success looks like]

**Success Criteria:**
1. ✅ [Criterion 1]
2. ✅ [Criterion 2]
3. ✅ [Criterion 3]

## Relationship to Other RFCs

[If applicable, describe how this RFC relates to other RFCs:]

This RFC (RFC-NNN) is part of [broader initiative] that includes:

1. **RFC-XXX: [Related RFC]** - [How they relate]
2. **RFC-YYY: [Another Related RFC]** - [How they relate]

**Key Distinction:**
- **[This RFC]**: [What it focuses on]
- **[Related RFC]**: [What it focuses on]

Together, these RFCs provide:
- [Benefit 1]
- [Benefit 2]

## Benefits

[Summary of benefits this RFC provides:]

1. **[Benefit 1]**: [Description]
2. **[Benefit 2]**: [Description]
3. **[Benefit 3]**: [Description]

## Migration Path

[If applicable, describe how to migrate from old approach to new approach:]

1. **Phase 1**: [Migration step 1]
2. **Phase 2**: [Migration step 2]
3. **Phase 3**: [Migration step 3]

## Open Questions

[Questions that need to be resolved during implementation:]

1. [Question 1]
2. [Question 2]
3. [Question 3]

## References

- **Related PRD**: `docs/prd/PRD-XXX-feature-name.md`
- **Related RFC**: `docs/rfc/RFC-XXX-related-feature.md`
- **Source Code**: `podcast_scraper/[module].py`
- **External Documentation**: [Link to external docs if applicable]
````
