"""THE INTEGRITY CHECK. The registry is the source of truth; everything else is a view of it.

Every guard we had before this file was of the form "check the things we listed" — and the failure
that keeps happening is *a thing nobody listed*. A param invented tomorrow, never registered, lives
as a code default nobody can see, and no test fails, because no test knows it should exist. That is
how all of the following shipped, each one silent:

  * ``gi_max_insights`` was **12** in the registry, **50** in the profiles, **20** in the Config
    default. Three doors, three answers, and production ran whichever one you came in through.
  * ``provider_chunked_gated_v3`` — the entire researched v3 tuning — was an **orphan**: no preset
    pointed at it, so the measured configuration reached zero production profiles.
  * ``gi_insight_temperature`` was *recorded in the registry and plumbed nowhere*, so every scored
    bake-off arm sampled at 0.3 while its config said 0.0 — and a model that disagrees with ITSELF
    between runs was about to be credited with "finding knowledge the other model missed".
  * ``gi_value_gate_enabled`` defaults to False, so the judge we spent two days fixing never ran in
    production at all.

So this suite is CLOSED-WORLD. It does not ask "are the listed fields correct?" It asks "is every
tunable field accounted for?" — and a new one that is neither governed nor explicitly exempted fails
the build with instructions. You cannot forget to register a param, because forgetting is a red
test.

THE LOOP IT PROTECTS
    an eval finds a better value
      -> it goes in the REGISTRY (a StageOption, with the research_ref that justified it)
      -> `make profiles-materialize`
      -> every profile inherits it; that is the new default, everywhere, in one commit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import yaml

from podcast_scraper import config as cfgmod
from podcast_scraper.providers.ml.model_registry import (
    _GI_OPTIONS,
    _LLM_PROVIDERS as LLM_PROVIDERS,
    _LOCAL_ONLY_LLM,
    _PREFERRED_GATE_MODEL,
    _PROFILE_PRESETS,
    REGISTRY_GOVERNED_FIELDS,
    resolve_profile_to_settings,
    resolve_value_gate,
)

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]

REPO = Path(__file__).resolve().parents[4]
PROFILE_DIR = REPO / "config" / "profiles"


# Quality-affecting tunables that are deliberately NOT registry-governed, each with the reason it is
# exempt. This list is the ONLY escape hatch, and it is meant to be short and argued.
#
# To add a field here you must be able to say why a measured value could never belong to it. If you
# cannot, the field belongs in the registry.
NOT_REGISTRY_GOVERNED: Dict[str, str] = {
    # DERIVED, not chosen: the judge must be vendor-disjoint from the model it grades (#939), so its
    # value depends on the summariser and cannot be one literal shared by every profile. The
    # resolver computes it; test_the_judge_is_never_the_defendant below proves it.
    "gi_value_gate_provider": "derived per-profile to be vendor-disjoint from the summariser",
    "gi_value_gate_model": "derived per-profile alongside gi_value_gate_provider",
    # DERIVED: an LLM grounds with an LLM. These follow the summary provider (a Config validator
    # auto-aligns them), so pinning one literal would hand a Gemini profile an Anthropic grounder.
    "quote_extraction_provider": "auto-aligned to the summary provider",
    "entailment_provider": "auto-aligned to the summary provider",
    # Local-model plumbing, not a quality choice: which torch device the transformers stack lands on
    # is a property of the MACHINE, not of the researched configuration.
    "extractive_qa_device": "hardware placement, not a tuned value",
    "nli_device": "hardware placement, not a tuned value",
    "gi_qa_model": "transformers-stack model id; superseded by LLM grounding on every LLM profile",
    "gi_nli_model": "transformers-stack model id; superseded by LLM grounding on every LLM profile",
    "gi_embedding_model": "embedding backend for dedupe; shared across profiles, not per-eval",
    "gi_evidence_extract_retries": "transport-level retry budget, not a quality parameter",
    "gi_fail_on_missing_grounding": "a run-mode switch (fail loudly vs continue), not a tuned value",
    "gil_evidence_nli_chunk_size": "batching for the local NLI stack; a throughput knob",
    "gi_insight_source": "governed — routing (which provider extracts); asserted separately",
    # Windowing for the STAGED extractive-QA stack. Every LLM profile grounds with the LLM
    # (`evidence_match_summary_provider`), so these tune a path those profiles do not take.
    "gi_qa_window_chars": "staged transformers-QA windowing; unused when the LLM grounds",
    "gi_qa_window_overlap_chars": "staged transformers-QA windowing; unused when the LLM grounds",
    # A KG-stage feature switch (typed MENTIONS_PERSON / MENTIONS_ORG edges), not part of the GI
    # researched configuration. It belongs to the KG option set, which this suite does not govern.
    "gi_typed_mentions_use_ner": "KG-stage feature switch; not a GI tuned value",
    # READ-TIME, not pipeline: the shared cap a SURFACE shows before 'show more' (ADR-135/#1191 §6).
    # Consumed by the server GI-view + player/viewer, each of which OVERRIDES it (player 6, viewer
    # 12, graph 8, search none). The GI extraction pipeline never reads it and no profile emits it,
    # so there is no per-profile pipeline value to materialize — #1191's eval tuned the surface cut,
    # not a StageOption. Same shape as gi_typed_mentions_use_ner: a gi_-prefixed field of another
    # layer.
    "gi_surface_default_limit": "read-time surface display cap (each surface overrides it); not a GI "
    "extraction-pipeline value the profiles materialize",
}


def _governed_profiles() -> List[Tuple[str, Path, Dict[str, Any]]]:
    out = []
    for path in sorted(PROFILE_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(data, dict) and data.get("profile") in _PROFILE_PRESETS:
            out.append((str(data["profile"]), path, data))
    return out


GOVERNED_PROFILES = _governed_profiles()
_IDS = [name for name, _, _ in GOVERNED_PROFILES]


# ---------------------------------------------------------------------------------------------
# 1. CLOSED WORLD — the check that catches the param nobody thought to list.
# ---------------------------------------------------------------------------------------------


def test_every_gi_tunable_is_either_governed_or_explicitly_exempt() -> None:
    """THE ONE THAT MAKES FUTURE DRIFT IMPOSSIBLE.

    Add a new ``gi_*`` / ``gil_*`` knob to Config and do nothing else, and this goes red. There is
    no way to introduce a quality parameter that quietly lives on a code default, because a field
    that is in neither bucket is a build failure with instructions attached.

    Every other guard in this repo checks a list someone remembered to update. This one checks the
    list itself.
    """
    tunables = {
        name
        for name in cfgmod.Config.model_fields
        if name.startswith(("gi_", "gil_"))
        or name in ("quote_extraction_provider", "entailment_provider")
    }
    unclassified = sorted(tunables - set(REGISTRY_GOVERNED_FIELDS) - set(NOT_REGISTRY_GOVERNED))
    assert not unclassified, (
        "These Config fields are quality parameters that NOTHING governs:\n\n"
        + "".join(f"    - {f}\n" for f in unclassified)
        + "\nA param that is neither registered nor exempted lives on a code default, invisible to "
        "the profiles and unpinnable by an eval — which is how `gi_value_gate_enabled` shipped "
        "switched off and how every bake-off arm sampled at 0.3 while its config said 0.0.\n\n"
        "Either:\n"
        "  (a) add it to a StageOption in model_registry.py (with a research_ref), map it in\n"
        "      _GI_SETTING_TO_CONFIG_KEY, list it in REGISTRY_GOVERNED_FIELDS, and run\n"
        "      `make profiles-materialize`; or\n"
        "  (b) add it to NOT_REGISTRY_GOVERNED with the reason a measured value could never\n"
        "      belong to it."
    )


# ---------------------------------------------------------------------------------------------
# 2. REGISTRY -> PROFILE — every governed field is WRITTEN, and says what the registry says.
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name,path,data", GOVERNED_PROFILES, ids=_IDS)
def test_every_governed_field_is_written_into_the_profile(
    name: str, path: Path, data: Dict[str, Any]
) -> None:
    """A profile must SAY what it runs. Omission is how a value hides.

    The old drift test only compared fields that were PRESENT in the YAML — so a profile that simply
    left a knob out was never checked, and silently inherited whatever the code default happened to
    be. Absence was indistinguishable from agreement.
    """
    resolved = resolve_profile_to_settings(name)
    expected = {k: resolved[k] for k in REGISTRY_GOVERNED_FIELDS if k in resolved}
    missing = sorted(k for k in expected if k not in data)
    assert not missing, (
        f"{path.name} does not write {missing}. Run `make profiles-materialize`. A profile that "
        f"omits a governed field is not 'using the default' — it is running a value nobody declared."
    )


@pytest.mark.parametrize("name,path,data", GOVERNED_PROFILES, ids=_IDS)
def test_the_profile_never_contradicts_the_registry(
    name: str, path: Path, data: Dict[str, Any]
) -> None:
    """The YAML is a VIEW. It gets no vote."""
    resolved = resolve_profile_to_settings(name)
    disagree = {
        k: (data[k], resolved[k])
        for k in REGISTRY_GOVERNED_FIELDS
        if k in resolved and k in data and data[k] != resolved[k]
    }
    assert not disagree, (
        f"{path.name} contradicts the registry:\n"
        + "".join(f"    {k}: profile={p!r} registry={r!r}\n" for k, (p, r) in disagree.items())
        + "\nThe registry is the source of truth. Change the StageOption (with a research_ref) and "
        "run `make profiles-materialize` — do not hand-edit the YAML."
    )


# Fields the resolver OMITS when empty (no chain, or fully stripped by allow_cloud_fallback=False).
# For these, absence-in-the-registry must mean absence-in-the-YAML — a stale key left behind would
# otherwise inject a fallback (including a cloud tier a fail-closed profile forbids) that the
# registry never emitted, and the contradiction test above skips it because it is not in `resolved`.
_OMITTED_WHEN_EMPTY = (
    "transcription_fallback_providers",
    "diarization_fallback_providers",
    "summary_fallback_providers",
)


@pytest.mark.parametrize("name,path,data", GOVERNED_PROFILES, ids=_IDS)
def test_no_orphan_fallback_chain_in_the_profile(
    name: str, path: Path, data: Dict[str, Any]
) -> None:
    """A ``*_fallback_providers`` key must never appear in a profile the registry did not emit it for
    (RFC-106 #1198). Guards the fail-closed invariant: a hand-added or stale chain — e.g. a cloud
    tier in an ``allow_cloud_fallback=False`` profile — is caught even though the registry omits the
    key (so the general contradiction check cannot see it)."""
    resolved = resolve_profile_to_settings(name)
    orphans = {k: data[k] for k in _OMITTED_WHEN_EMPTY if k in data and k not in resolved}
    assert not orphans, (
        f"{path.name} carries fallback chains the registry did not emit: {orphans}. "
        f"Remove them (or set them on the ProfilePreset) and run `make profiles-materialize` — a "
        f"stale/hand-added chain bypasses the registry and can defeat allow_cloud_fallback."
    )


# ---------------------------------------------------------------------------------------------
# 3. THE DEFAULT IS THE DEFAULT — a code path with no profile must not run a different pipeline.
# ---------------------------------------------------------------------------------------------


def test_the_config_default_is_not_a_trap() -> None:
    """Any caller that does not load a profile still has to get the researched configuration.

    Before this, ``quote_extraction_provider`` and ``entailment_provider`` defaulted to
    ``transformers`` and the evidence modes defaulted to ``staged`` — so code that skipped the
    profile silently ground with the local DeBERTa stack instead of the LLM. The default was not a
    default; it was a different product.
    """
    ref = resolve_profile_to_settings("cloud_balanced")
    mismatched = []
    for key in (
        "gi_max_insights",
        "gi_require_grounding",
        "gil_evidence_quote_mode",
        "gil_evidence_nli_mode",
        "gi_insight_prompt_version",
        "gi_insight_dedupe_threshold",
        "gi_value_gate_enabled",
    ):
        field = cfgmod.Config.model_fields.get(key)
        if field is None or key not in ref:
            continue
        if field.default != ref[key]:
            mismatched.append((key, field.default, ref[key]))
    assert not mismatched, (
        "Config defaults disagree with the registry's researched configuration:\n"
        + "".join(f"    {k}: default={d!r} registry={r!r}\n" for k, d, r in mismatched)
        + "\nA caller that does not load a profile runs a DIFFERENT pipeline than the one we "
        "measured. Set the Config default to the registry's value."
    )


# ---------------------------------------------------------------------------------------------
# 4. THE JUDGE IS NEVER THE DEFENDANT (#939).
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_PROFILE_PRESETS))
def test_the_gate_never_runs_where_there_is_no_llm_to_run_it(name: str) -> None:
    """The value gate is an LLM asking "is this worth surfacing". On the pure-ML path — sentence
    transformers, summllama, the local extractive stack — there is no LLM, so there is no judge and
    there cannot be one. The gate is INAPPLICABLE there, not merely switched off.

    This is not pedantry. Materialising the researched configuration blindly turned the gate on for
    `airgapped`, `airgapped_thin` and `dev`, and the derived judge came back as **anthropic** — a
    hosted network call inside an AIRGAPPED profile, which is the one thing airgapped means it
    cannot do, and real paid LLM calls inside `dev`, which CI must never make.
    """
    resolved = resolve_profile_to_settings(name)
    summariser = str(resolved.get("summary_provider"))
    if summariser in LLM_PROVIDERS:
        return
    assert resolved.get("gi_value_gate_enabled") is False, (
        f"{name} extracts with '{summariser}', which is not an LLM — there is nothing to judge "
        f"with. The gate must be off."
    )
    assert resolved.get("gi_value_gate_provider") is None, (
        f"{name} extracts with '{summariser}' (no LLM) but was handed the judge "
        f"'{resolved.get('gi_value_gate_provider')}'. On an airgapped profile that is a network "
        f"call; in dev it is a paid API call from CI."
    )


@pytest.mark.parametrize("name", sorted(_PROFILE_PRESETS))
def test_an_offline_profile_never_reaches_for_a_hosted_judge(name: str) -> None:
    """Airgapped means airgapped. The gate never phones out.

    Previously asserted ``is None`` as a PROXY for "not hosted", which held only because the old
    resolver refused to name a rater for local providers. Same-route resolution now records the
    local model explicitly — safer and more visible, but it means the proxy no longer tracks the
    property. Assert the property itself.
    """
    if "airgapped" not in name and not name.startswith("local"):
        return
    rater = resolve_profile_to_settings(name).get("gi_value_gate_provider")
    assert rater is None or rater in _LOCAL_ONLY_LLM, (
        f"{name} is an offline profile but would rate insights via '{rater}', which is hosted. An "
        f"offline run rates locally or does not rate — it does not phone out."
    )


@pytest.mark.parametrize("name", sorted(_PROFILE_PRESETS))
def test_the_gate_model_rides_the_summarisers_route(name: str) -> None:
    """The value-gate rater uses the summariser's OWN provider — same proxy, same credential.

    This is the inverse of the rule that used to live here. The old invariant ("never the same
    vendor") is an EVALUATION rule (#939): a bake-off judge must not grade its own family or the
    scoreboard reports our judge assignment as model quality. It was wrongly applied to the
    production pipeline, where this stage is a small inline tier rater and not a competition. The
    result was that cloud_with_dgx_primary and cloud_openrouter — both litellm-routed — resolved to
    a DIRECT Anthropic call mid-pipeline: a second vendor, a second credential, and spend invisible
    to the gateway's SpendLogs.

    #939 still governs autoresearch cohorts. It does not govern this.
    """
    resolved = resolve_profile_to_settings(name)
    summariser = resolved.get("summary_provider")
    rater = resolved.get("gi_value_gate_provider")
    if rater is None:
        # No LLM on this path at all — the resolver records that rather than leaving it implicit.
        assert resolve_value_gate(str(summariser))[1] is None
        return
    assert rater == summariser, (
        f"{name}: the value gate rates via '{rater}' but the summariser is '{summariser}'. The "
        f"rater must ride the summariser's route so there is one credential and one spend ledger."
    )


@pytest.mark.parametrize("name", sorted(_PROFILE_PRESETS))
def test_a_curated_route_rates_with_a_distinct_model(name: str) -> None:
    """Where a route has a curated sibling, the rater must not be the summariser's own model.

    Same route is the rule; same MODEL is the thing we are still trying to avoid, because
    self-grading is measurably lenient (~10% of insights dropped against ~25%). This is what stops
    the fix from quietly turning every profile into a self-grader — which is precisely what
    cloud_balanced already was (podcast-flash-0731 rating podcast-flash-0731) across all 765
    episodes before this change.
    """
    resolved = resolve_profile_to_settings(name)
    summariser = str(resolved.get("summary_provider"))
    if summariser not in _PREFERRED_GATE_MODEL:
        return  # uncurated route: self-grade is the accepted, logged outcome
    rater_model = resolved.get("gi_value_gate_model")
    summary_model = resolved.get(f"{summariser}_summary_model") or resolved.get("summary_model")
    assert rater_model == _PREFERRED_GATE_MODEL[summariser], (
        f"{name}: curated route {summariser!r} must rate with "
        f"{_PREFERRED_GATE_MODEL[summariser]!r}, got {rater_model!r}"
    )
    if summary_model:
        assert rater_model != summary_model, (
            f"{name}: {summariser} is rating its own output with {rater_model!r} — self-grading, "
            f"roughly half as strict as a distinct rater."
        )


@pytest.mark.parametrize("provider", ["openai", "deepseek", "litellm", "qwen", "vllm", "groq"])
def test_value_gate_runs_for_every_hosted_llm_sibling(provider: str) -> None:
    """The gate is membership-gated on ``_LLM_PROVIDERS``; a summary provider missing from that set
    SILENTLY fails open — no grading, no error. The entire v2.5 finale ran via ``litellm`` and its
    ``gi_value_gate_enabled: true`` was a NO-OP because ``litellm`` (and the ``qwen`` ADR-147 sibling)
    were absent here, while ``deepseek`` was present — so a native-deepseek arm graded and its
    gateway twin did not, making their insight counts and cost non-comparable. Guard every real LLM
    sibling so a new provider can never re-disable this quality stage by omission.
    """
    enabled, _ = resolve_value_gate(provider)
    assert enabled, (
        f"value gate silently disabled for summary_provider={provider!r}: it is not in "
        f"_LLM_PROVIDERS, so gi_value_gate_enabled becomes a no-op for every run that uses it."
    )


# ---------------------------------------------------------------------------------------------
# 5. PROVENANCE — a registered value must say which eval justified it.
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("option_id", sorted(_GI_OPTIONS))
def test_every_registered_gi_option_cites_its_research(option_id: str) -> None:
    """A number in the registry without a report behind it is a guess with good posture."""
    opt = _GI_OPTIONS[option_id]
    assert opt.research_ref, f"GI option '{option_id}' carries tuned params but cites no research"


def test_no_gi_option_is_an_orphan() -> None:
    """A researched configuration that no preset points at never reaches production.

    ``provider_chunked_gated_v3`` sat in this registry — with the temperature pin, the dedupe
    threshold and the value gate all measured — while all 14 production presets pointed at the
    stale ``provider_n12_grounded_bundled``. The research was done, written down, and wired to
    nothing.
    """
    used = {p.gi for p in _PROFILE_PRESETS.values() if p.gi}
    # A DEPRECATED option is allowed to be unused — that is what deprecated means. Superseded
    # research is kept as history (with the reason it was abandoned), not deleted; knowing why n=12
    # was wrong is worth more than the 12 ever was.
    live = {k for k, v in _GI_OPTIONS.items() if v.tier != "deprecated"}
    orphans = sorted(live - used)
    assert not orphans, (
        f"GI options no preset uses: {orphans}. Point a preset at them, or mark them "
        f"tier='deprecated' with a deprecation_reason — a live registry entry nobody references is "
        f"research that never shipped, which is exactly how the v3 tuning reached zero profiles."
    )
