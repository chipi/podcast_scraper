# Pull request

## Summary

<!-- 1-3 bullet points: what changed and why -->

## Test plan

<!-- Markdown checklist of what was tested, including manual checks where relevant -->

- [ ] `make ci-fast` green
- [ ] Tests added/updated for the change

## Viewer changes (delete if N/A)

If this PR touches `web/gi-kg-viewer/**` graph navigation, handoff stores,
entry-point components, or `GraphCanvas.vue`:

- [ ] **Tier 1** fast matrix green: `cd web/gi-kg-viewer && node_modules/.bin/playwright test e2e/handoff/`
- [ ] **Tier 2** production-shaped matrix green: `cd web/gi-kg-viewer && node_modules/.bin/playwright test e2e/handoff-production/`
- [ ] **Tier 3** validation walk run against a real corpus (RFC-086 / ADR-095):
      `make ci-ui-validation CORPUS=/path/to/corpus`
      Corpus used: `<name>`
      Date: `<YYYY-MM-DD>`

If this PR fixes a bug surfaced by Tier 3 (real-backend validation), per
the [ADR-095 institutional rule](docs/adr/ADR-095-viewer-test-pyramid.md):

- [ ] A Tier-2 matrix row under `web/gi-kg-viewer/e2e/handoff-production/`
      reproduces the bug. The row fails pre-fix and passes post-fix in
      this PR.

## Related

<!-- Issue references (Fixes #123), RFCs, ADRs -->
