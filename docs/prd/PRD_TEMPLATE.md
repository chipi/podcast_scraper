# PRD-NNN: [Feature Name]

## Summary

[One paragraph describing what this feature does and why it matters. Should be clear and concise, suitable for executive summary.]

## Background & Context

[2-3 paragraphs explaining:]

- [What problem this solves]
- [Why this is needed now]
- [How it relates to existing features]
- [Reference to related architecture/docs if relevant]

## Goals

[Bulleted list of what this feature aims to achieve:]

- [Goal 1: Clear, measurable outcome]
- [Goal 2: User benefit or system improvement]
- [Goal 3: Integration or compatibility goal]

## Non-Goals

[Explicitly state what this feature does NOT do:]

- [Out of scope item 1]
- [Out of scope item 2]
- [Future consideration that's not in this PRD]

## Personas

[Describe 2-3 key user personas who will benefit from this feature:]

- **[Persona Name]**: [Role/description]
  - [What they need]
  - [How this feature helps them]

- **[Persona Name]**: [Role/description]
  - [What they need]
  - [How this feature helps them]

## User Stories

[Format: "As [persona], I can [action] so that [benefit]."]

- _As [Persona], I can [do something] so that [benefit]._
- _As [Persona], I can [do something] so that [benefit]._
- _As [any operator], I can [do something] so that [benefit]._

## Functional Requirements

[Organize by feature area or component. Use FR1, FR1.1, FR1.2 format for hierarchical requirements.]

### FR1: [Feature Area Name]

- **FR1.1**: [Specific requirement with clear acceptance criteria]
- **FR1.2**: [Specific requirement with clear acceptance criteria]
- **FR1.3**: [Specific requirement with clear acceptance criteria]

### FR2: [Another Feature Area]

- **FR2.1**: [Specific requirement]
- **FR2.2**: [Specific requirement]

[Continue as needed...]

## Success Metrics

[Measurable criteria for success:]

- [Metric 1: e.g., ">=95% of episodes with published transcripts complete without manual retry"]
- [Metric 2: e.g., "Processing overhead: <20% increase in total processing time"]
- [Metric 3: e.g., "Zero impact on existing workflows when disabled"]

## Dependencies

[What this feature depends on:]

- [PRD-XXX: Related PRD]
- [RFC-XXX: Related RFC]
- [External dependency or system requirement]

## Constraints & Assumptions

[If applicable, document constraints and assumptions:]

**Constraints:**

- [Hardware constraint, e.g., "Must run on Apple M4 Pro with 48 GB RAM"]
- [Performance constraint, e.g., "Must complete in < 5 seconds"]
- [Compatibility constraint, e.g., "Must be backward compatible"]

**Assumptions:**

- [Assumption 1, e.g., "Users have internet connectivity"]
- [Assumption 2, e.g., "Local models are preferred over API-based solutions"]

## Design Considerations

[If applicable, document design decisions and trade-offs:]

### [Design Decision Topic]

- **Option A**: [Description]
  - **Pros**: [Benefits]
  - **Cons**: [Drawbacks]
  - **Decision**: [Which option chosen and why]

### [Another Design Decision]

- [Similar structure...]

## Integration with [Related Feature]

[If applicable, describe how this feature integrates with existing features:]

[Feature Name] enhances the pipeline by:

- **[Integration Point 1]**: [How it integrates]
- **[Integration Point 2]**: [How it integrates]

## Example Output

[If applicable, show example of what the feature produces:]

```json
{
  "example": "output structure"
}
```json

- [Question 3: e.g., "How to handle multi-speaker transcripts in summaries?"]

## Related Work

[Links to related issues, PRDs, RFCs:]

- Issue #XXX: [Related issue description]
- PRD-XXX: [Related PRD]
- RFC-XXX: [Related RFC]

## Release Checklist

[Checklist for release readiness:]

- [ ] PRD reviewed and approved
- [ ] RFC-XXX created with technical design
- [ ] Implementation completed
- [ ] Tests cover [key scenarios]
- [ ] Documentation updated (README, config examples)
- [ ] Integration with [related feature] verified

```
