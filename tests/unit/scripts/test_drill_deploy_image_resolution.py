"""Regression guard for the DR-drill deploy image-SHA resolution (#1088).

The scheduled DR drill used to deploy ``github.sha`` (main HEAD) blindly. When
that commit's ``stack-test`` build failed (or main advanced past the last green
build), the image was absent and ``docker manifest inspect`` hit
``manifest unknown`` — the drill's real failure mode (run 28072149152), which the
issue's original "empty SHA / workflow_call" theory misdiagnosed.

The fix: resolve to the newest ``sha-<7>`` tag actually published to GHCR. These
content checks stop the blind-HEAD fallback from creeping back.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DRILL_DEPLOY = REPO_ROOT / ".github" / "workflows" / "drill-deploy.yml"
DEPLOY_PROD = REPO_ROOT / ".github" / "workflows" / "deploy-prod.yml"


def test_drill_resolves_to_published_ghcr_image_not_main_head() -> None:
    text = DRILL_DEPLOY.read_text(encoding="utf-8")
    # Must query GHCR for the newest published sha tag...
    assert "PKG: podcast-scraper-stack-api" in text
    assert "packages/container/${PKG}/versions" in text
    assert 'select(test("^sha-[0-9a-f]{7,40}$"))' in text
    # ...and must NOT fall back to blind main HEAD (the #1088 bug).
    assert 'echo "${{ github.sha }}" | cut' not in text, "blind github.sha fallback reintroduced"


def test_drill_override_still_takes_precedence() -> None:
    text = DRILL_DEPLOY.read_text(encoding="utf-8")
    assert "OVERRIDE: ${{ inputs.override_image_sha }}" in text
    # An empty resolution must fail loudly, not silently deploy a bad tag.
    assert "no published sha-<7> image" in text


def test_deploy_prod_reads_inputs_not_event_inputs() -> None:
    # `github.event.inputs.*` is null under workflow_call; `inputs.*` is correct
    # under both dispatch and a future workflow_call trigger (RFC-082 Decision 6).
    text = DEPLOY_PROD.read_text(encoding="utf-8")
    assert "OVERRIDE: ${{ inputs.override_image_sha }}" in text
    assert "github.event.inputs.override_image_sha" not in text
