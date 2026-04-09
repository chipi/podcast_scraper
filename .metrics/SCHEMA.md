# Schema for `rule-adherence.jsonl`

One JSON object per line. All keys are strings; list values are arrays of strings.

**Sessions may contain many lines** — append after each meaningful milestone, not only once at the end.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `timestamp` | string | Yes | ISO 8601 (e.g. `2026-02-10T14:30:00Z`) |
| `task_summary` | string | Yes | Short description of what was done |
| `rules_applied` | list of strings | Yes | Rule IDs followed (e.g. `["0", "2a", "8a", "13"]`) |
| `rules_missed` | list of strings | Yes | Rule IDs missed or N/A (can be `[]`) |
| `skills_used` | list of strings | No | Skills invoked (e.g. `["commit-with-approval"]`) |
| `subagents_used` | list of strings | No | Subagents delegated to (e.g. `["verifier"]`) |

## Example (full)

```json
{"timestamp": "2026-02-10T15:00:00Z", "task_summary": "Expand metrics for skills and subagents", "rules_applied": ["0", "2a", "8a", "13", "16"], "rules_missed": [], "skills_used": ["test-scope-decision-tree"], "subagents_used": []}
```

## Example (minimal)

```json
{"timestamp": "2026-02-10T14:00:00Z", "task_summary": "Doc fix", "rules_applied": ["8a", "13"], "rules_missed": []}
```
