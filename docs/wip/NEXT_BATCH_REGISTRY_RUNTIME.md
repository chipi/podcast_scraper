# Next batch — wire the registry resolver into the runtime (Option B)

**Status**: planned post-#975 merge.
**Scope**: take `resolve_profile_to_settings()` from documentation-only to load-bearing — `Config` reads from the registry, drift between profile YAMLs and registry presets becomes a runtime / CI error.
**Prereq**: PR #975 must be merged to main.
**Estimated effort**: 2-3 hr (one session).

---

## Why

#975 set up the registry as canonical: every `StageOption` carries `research_ref`, every `ProfilePreset` names one option per stage, and `resolve_profile_to_settings(name)` returns a flat dict the pipeline could ingest.

But **nothing actually consumes the resolver**. Profile YAMLs are still parsed the old way (every field explicit, no registry awareness). The materialize-decisions loop is documentation-true, not runtime-true. Drift between a profile YAML and its declared preset is currently a documentation problem; this batch turns it into a CI failure.

The win after this batch:

```yaml
# config/profiles/cloud_with_dgx_primary.yaml — eventually
profile: cloud_with_dgx_primary
dgx_tailnet_host: ${DGX_TAILNET_HOST}
# everything else (transcription provider/model/endpoint, summary provider/model,
# all the GI/KG/clustering defaults) resolved from the registry preset
```

— with the option to add explicit fields that **override** the registry default per profile, so research-driven defaults and operator-specific tweaks coexist cleanly.

---

## Design

### Top-level: opt-in `profile:` field on `Config`

Profile YAMLs gain an optional top-level `profile: <name>` field. When set, the YAML's other fields **override** registry defaults (existing fields win); unset fields get filled from `resolve_profile_to_settings(name, dgx_tailnet_host=...)`.

Existing profile YAMLs without a `profile:` field continue to behave exactly as today. No backward-compat shims needed — the registry path is purely additive.

### Resolution order (highest wins)

1. Explicit field in the YAML (e.g. `transcription_provider: openai` on the YAML).
2. Registry preset's StageOption for that stage.
3. `Config` field default (Pydantic-level fallback that already exists).

### Hostname threading

`resolve_endpoint()` already supports an explicit `dgx_tailnet_host` arg → `DGX_TAILNET_HOST` env → sentinel. When `Config` calls `resolve_profile_to_settings()`, it must pass its own `dgx_tailnet_host` value so endpoints get substituted from one consistent source.

`resolve_profile_to_settings()` itself needs to accept a `dgx_tailnet_host: Optional[str]` arg and pass it through to `resolve_endpoint()` for each stage option.

### Drift detection (new CI gate)

A new test under `tests/integration/providers/ml/test_profile_yaml_registry_drift.py`:

1. Discover every YAML under `config/profiles/*.yaml`.
2. For each, parse it. If it has a `profile:` field, look up the matching `ProfilePreset` in the registry.
3. Compare the YAML's stage routing (`transcription_provider`, `transcription_model`, `summary_provider`, `summary_model`, `dgx_whisper_port`, `dgx_whisper_model`) against `resolve_profile_to_settings(profile_name, dgx_tailnet_host=YAML_HOST_OR_PLACEHOLDER)`.
4. If they disagree without an explicit override comment, fail with the diff. This catches accidental hand-edits that don't match the registry.

A `make profile-drift-check` target wires this into the existing `make ci-fast` chain (or runs standalone).

---

## File-by-file changes

### 1. `src/podcast_scraper/providers/ml/model_registry.py`

Update `resolve_profile_to_settings()` to accept the host arg:

```python
def resolve_profile_to_settings(
    name: str,
    dgx_tailnet_host: Optional[str] = None,
) -> Dict[str, Any]:
    """... (existing docstring + add note about host threading)"""
    preset = get_profile_preset(name)
    tx = get_transcription_option(preset.transcription)
    sm = get_summary_option(preset.summary)

    settings: Dict[str, Any] = {
        "transcription_provider": tx.provider,
    }
    if tx.model is not None:
        settings.setdefault("transcription_model", tx.model)
    resolved_tx_endpoint = resolve_endpoint(tx.endpoint, dgx_tailnet_host)
    if resolved_tx_endpoint is not None:
        settings["transcription_endpoint"] = resolved_tx_endpoint
    if tx.extra_settings:
        settings["transcription_extra"] = dict(tx.extra_settings)

    settings["summary_provider"] = sm.provider
    if sm.model is not None:
        settings["summary_model"] = sm.model
    resolved_sm_endpoint = resolve_endpoint(sm.endpoint, dgx_tailnet_host)
    if resolved_sm_endpoint is not None:
        settings["summary_endpoint"] = resolved_sm_endpoint
    if sm.extra_settings:
        settings["summary_extra"] = dict(sm.extra_settings)

    settings["_profile_preset"] = preset.name
    settings["_transcription_research_ref"] = tx.research_ref
    settings["_summary_research_ref"] = sm.research_ref

    return settings
```

Update tests in `test_model_registry_stage_options.py` to assert that:
- `resolve_profile_to_settings("cloud_with_dgx_primary", dgx_tailnet_host="my.host.ts.net")` returns the endpoint with the explicit host substituted.
- `resolve_profile_to_settings(...)` without `dgx_tailnet_host` falls back to the sentinel (so callers without context get fail-fast URLs).

### 2. `src/podcast_scraper/config.py`

Add a `profile` field + a model validator:

```python
profile: Optional[str] = Field(
    default=None,
    description=(
        "Optional registry preset name. When set, unset fields are filled "
        "from src/podcast_scraper/providers/ml/model_registry.py's "
        "_PROFILE_PRESETS via resolve_profile_to_settings(). Explicit YAML "
        "fields override registry defaults. See "
        "docs/wip/RESEARCH_POWERED_REGISTRY_PLAN.md."
    ),
)
```

Add a `mode="before"` model validator (runs before per-field validation, sees raw input dict):

```python
@model_validator(mode="before")
@classmethod
def _apply_registry_profile(cls, values: Any) -> Any:
    """If a profile name is set, populate missing fields from the registry."""
    if not isinstance(values, dict):
        return values
    profile_name = values.get("profile")
    if not profile_name:
        return values
    from podcast_scraper.providers.ml.model_registry import resolve_profile_to_settings
    # Pull the host from the YAML (already set) or env (registry resolver
    # falls back to env / sentinel on its own).
    host = values.get("dgx_tailnet_host")
    resolved = resolve_profile_to_settings(profile_name, dgx_tailnet_host=host)
    # Existing YAML fields win — only fill what's not present.
    for key, value in resolved.items():
        if key.startswith("_"):
            continue  # skip metadata fields
        values.setdefault(key, value)
    return values
```

Notes:
- This runs BEFORE Pydantic's per-field type coercion, so `values` is a plain dict.
- Existing YAMLs without `profile:` set are untouched.
- The resolver's `transcription_provider` / `summary_provider` / etc. keys must map cleanly to existing `Config` field names. Where they don't (currently the resolver returns e.g. `transcription_endpoint` but `Config` has `dgx_whisper_port` + builds endpoint elsewhere), the resolver may need adjustment to emit the field names `Config` already expects — OR `Config` gains pass-through fields for the new keys. **Decision point**: prefer to make the resolver emit existing field names (`dgx_whisper_port`, `dgx_whisper_model`) rather than inventing new ones — fewer downstream changes.

### 3. `tests/integration/providers/ml/test_profile_yaml_registry_drift.py` (NEW)

Discover-and-check pattern:

```python
"""Drift detection — every profile YAML that names a registry preset must
match the preset's routing on key fields. Catches hand-edits that disagree
with the eval-report-justified registry default."""

import os
from pathlib import Path

import pytest
import yaml

from podcast_scraper.providers.ml.model_registry import (
    _PROFILE_PRESETS,
    resolve_profile_to_settings,
)


_PROFILE_DIR = Path("config/profiles")
# Fields that must match between YAML and the registry preset.
_ROUTING_FIELDS = (
    "transcription_provider",
    "summary_provider",
    # Optionally — but skip if the YAML uses a different routing field name:
    # "transcription_model",
    # "summary_model",
)


def _profile_yaml_files():
    """Yield (path, parsed_dict) for every yaml under config/profiles/ that
    declares a `profile:` field."""
    for path in sorted(_PROFILE_DIR.glob("*.yaml")):
        with path.open() as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            continue
        if "profile" in data:
            yield path, data


@pytest.mark.parametrize(
    "path,data",
    list(_profile_yaml_files()),
    ids=lambda v: v.name if hasattr(v, "name") else "",
)
def test_profile_yaml_matches_registry_preset(path: Path, data: dict) -> None:
    profile_name = data["profile"]
    assert profile_name in _PROFILE_PRESETS, (
        f"{path.name} declares profile={profile_name!r} which is not in the registry. "
        f"Known: {sorted(_PROFILE_PRESETS)}"
    )
    expected = resolve_profile_to_settings(profile_name, dgx_tailnet_host="drift-check.example")
    for field in _ROUTING_FIELDS:
        if field not in data:
            continue  # field omitted; registry default applies — no drift
        assert data[field] == expected[field], (
            f"{path.name}: field {field!r} = {data[field]!r} disagrees with "
            f"registry preset {profile_name!r} (expected {expected[field]!r}). "
            f"Either update the YAML to match the registry, or update the registry "
            f"with new research_ref evidence."
        )
```

### 4. `Makefile` (target)

```make
profile-drift-check: ## Verify config/profiles/*.yaml match their declared registry presets
	@.venv/bin/python -m pytest tests/integration/providers/ml/test_profile_yaml_registry_drift.py -q
```

Wire into `test-fast` (and therefore `ci-fast`) so a stale profile YAML breaks CI loudly.

### 5. Convert one profile YAML as the proof (the only profile YAML change in this batch)

Edit `config/profiles/cloud_with_dgx_primary.yaml` to declare its profile + drop the now-redundant fields:

```yaml
# Profile: cloud_with_dgx_primary — DGX-hosted Whisper + pyannote, cloud
# fallback. The transcription + summary routing comes from the registry
# preset; this file only carries the operator-specific overrides
# (dgx_tailnet_host, GI/KG settings, diarization config).

profile: cloud_with_dgx_primary

# Tailscale hostname (operator-specific). Set DGX_TAILNET_HOST in your .env
# or replace this default per checkout.
dgx_tailnet_host: your-dgx.tailnet.ts.net

transcription:
  primary: tailnet_dgx_whisper
  fallback: openai

# transcription_fallback_provider, screenplay, diarize, diarization_provider,
# dgx_diarize_port, dgx_diarize_model, speaker_detector_provider, etc. —
# leave as-is; they're operator policy, not registry-managed.
transcription_fallback_provider: openai
screenplay: true
diarize: true

# ... rest of the file (diarization, GI, KG, etc.) stays unchanged
```

Note: `dgx_whisper_model: large-v3` and `dgx_whisper_port: 8002` (the registry-driven choices) can now be DROPPED from the YAML — the registry preset already names them. Drift-check test verifies the resolved `transcription_endpoint` field on `Config` (which the existing routing code reads) matches what the YAML expected.

Don't migrate the other profile YAMLs in this batch — one convert + the drift test is enough to prove the mechanism. The remaining migrations are mechanical and can be one follow-up commit each.

### 6. `docs/guides/EXPERIMENT_GUIDE.md` — update Step 6

The "Step 6: Materialize" section already exists. Add a "Step 6b: Verify the runtime sees the change" subsection:

```markdown
### Step 6b: Verify the runtime sees the change

`make profile-drift-check` runs the test that asserts every
`config/profiles/*.yaml` declaring a `profile:` field matches its
registry preset. Run it locally after a registry edit:

\`\`\`bash
make profile-drift-check
\`\`\`

If it fails, either:
- Update the YAML to match the new registry default (most common —
  what the materialize-decisions flow expects).
- Add an explicit override comment in the YAML and silence the field
  via the test's per-field skip list (rare — for profile-specific
  deviation that doesn't deserve a separate registry preset).
\`\`\`
```

### 7. `AGENTS.md` — update the materialize flow

The materialize-decisions section in AGENTS.md lists 6 steps. Add a 7th:

```
7. Verify the runtime               →  make profile-drift-check
     - Catches any profile YAML that disagrees with its declared
       registry preset.
     - Wired into make ci-fast.
```

### 8. `CONTRIBUTING.md`

Add `make profile-drift-check` to the validation workflow bullet list (line ~470).

---

## Acceptance criteria

- [ ] `resolve_profile_to_settings(name, dgx_tailnet_host=...)` threads the host through `resolve_endpoint()` for transcription + summary stages.
- [ ] `Config` has a new optional `profile: str | None` field with a `mode="before"` validator that fills missing fields from the resolver when set.
- [ ] `config/profiles/cloud_with_dgx_primary.yaml` declares `profile: cloud_with_dgx_primary` and drops redundant registry-managed fields. Profile still parses; routing still matches what pre-change YAML said.
- [ ] New drift-detection test under `tests/integration/providers/ml/test_profile_yaml_registry_drift.py` discovers + checks all profile YAMLs.
- [ ] `make profile-drift-check` target exists and is wired into `test-fast` (so `make ci-fast` runs it).
- [ ] EXPERIMENT_GUIDE.md Step 6b, AGENTS.md step 7, CONTRIBUTING.md bullet — all reference the new check.
- [ ] All existing tests still pass (`make ci-fast` green).
- [ ] One round of manual validation: operator runs the pipeline against `cloud_with_dgx_primary` and confirms transcription routes to `:8002` (whisper-openai) as before.

## What this batch does NOT do

- **Migrate the remaining profile YAMLs** (`local_dgx_balanced.yaml`, etc.) to use `profile:` declarations. Each migration is mechanical and can be a one-line follow-up commit. The point of this batch is the mechanism, not exhaustive coverage.
- **Add GI / KG / NER / clustering stage options to the registry.** Currently only transcription + summary are registered. Adding those is its own scope — they're not routing concerns the resolver/drift-check care about.
- **Build a `make profile-resolve` CLI** that prints the resolved settings for a given profile. Useful for ops debugging but not load-bearing.
- **Templatize the `dgx_tailnet_host` field default** in profile YAMLs to use env-var expansion. The current `your-dgx.tailnet.ts.net` placeholder + operator-edit-per-checkout pattern is fine; env-var expansion would need a custom YAML loader.

## Order of execution (for the automating session)

1. `git checkout main && git pull` — pick up #975's merge.
2. `git checkout -b feat/registry-runtime-wiring`.
3. Edit `model_registry.py` — add `dgx_tailnet_host` arg to `resolve_profile_to_settings()`. Run `pytest tests/integration/providers/ml/test_model_registry_stage_options.py` to confirm the existing 18 tests still pass; add 1-2 new tests for host threading.
4. Edit `config.py` — add `profile` field + `_apply_registry_profile` validator. Run `pytest tests/unit/podcast_scraper/test_config*.py` — if any existing config test fails, fix; if not, add a focused test that loads a YAML with `profile: cloud_with_dgx_primary` and asserts the routing fields populate.
5. Write the drift-detection test file (full text above). Run it standalone: `pytest tests/integration/providers/ml/test_profile_yaml_registry_drift.py -v`. Should pass with zero YAMLs migrated (no YAML declares `profile:` yet).
6. Migrate `cloud_with_dgx_primary.yaml` (the proof). Re-run the drift test; should still pass — but now with one parametrized case. Run the operator-side smoke (`make experiment-run` or whatever the equivalent is for routing-only validation).
7. Add `make profile-drift-check` target. Wire into `test-fast`.
8. Update EXPERIMENT_GUIDE.md, AGENTS.md, CONTRIBUTING.md.
9. `make ci-fast` — must be green.
10. Commit. Push. Open PR with a description referencing this plan doc.

## References

- The registry the resolver lives in: `src/podcast_scraper/providers/ml/model_registry.py` (#975 commit `49a1a8c9`).
- The 6-step materialize flow: `AGENTS.md` "Materialize autoresearch decisions in the registry" section.
- The RFC vision: `docs/wip/RESEARCH_POWERED_REGISTRY_PLAN.md`.
- The ADR amendment: `docs/adr/ADR-048-centralized-model-registry.md` 2026-06-12 block.
- Existing test patterns to mirror: `tests/integration/providers/ml/test_model_registry_stage_options.py` (#975 commit `49a1a8c9`).
