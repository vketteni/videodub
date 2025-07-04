# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for the VideoDub project. ADRs document important architectural decisions, their context, and consequences.

## ADR Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [0001](0001-pipeline-architecture.md) | Pipeline Architecture Design | Accepted | 2025-07-03 |
| [0002](0002-cost-tracking-integration.md) | Real-Time Cost Tracking Integration | Accepted | 2025-07-03 |
| [0003](0003-transcript-processor-redesign.md) | Transcript Processing Service Redesign | Accepted | 2025-07-03 |

## ADR Template

When creating new ADRs, use this structure:

```markdown
# ADR XXXX: [Short Title]

## Status
[Proposed | Accepted | Deprecated | Superseded by ADR-YYYY]

## Context
[What is the issue that we're seeing? What factors affect this decision?]

## Decision
[What is the change that we're proposing and/or doing?]

## Consequences
[What becomes easier or more difficult to do because of this change?]

### Positive
- [Benefit 1]
- [Benefit 2]

### Negative  
- [Drawback 1]
- [Drawback 2]

## Date
[YYYY-MM-DD when the decision was made]
```

## Contributing

- ADRs should be numbered sequentially
- Use meaningful, descriptive titles
- Focus on architectural decisions, not implementation details
- Include both positive and negative consequences
- Update the index when adding new ADRs