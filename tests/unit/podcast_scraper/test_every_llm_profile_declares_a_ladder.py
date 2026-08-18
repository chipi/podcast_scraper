"""A failover capability nothing configures is not a capability (#1657).

The wrapper was fixed to fail over on every LLM call. That work was worth nothing in production
because only 2 of 39 profiles declared a ladder — ``cloud_balanced`` and ``prod_dgx_full``.
Everything else, including ``cloud_openrouter`` (the profile named after the vendor whose budget
ran out) and both corpus-repair ``reprocess_*`` profiles, ran with no failover at all. Working
code plus unconfigured code is broken behaviour.

This test closes the loop the other tests do not reach: ``test_failover_covers_every_llm_call``
proves the mechanism works, and this proves it is actually switched on where it matters.

THE RULE
A profile that makes CLOUD LLM calls must declare ``summary_fallback_providers``, with one
deliberate exception: profiles that exist to MEASURE a single provider must not have one.

* ``bakeoff_*`` pins one provider to score it. A fallback would silently blend two providers'
  output into one measurement and quietly invalidate the eval.
* ``freeze/*`` pins a released configuration for reproducibility. A fallback would make a frozen
  run non-deterministic, which defeats the freeze.

A third exemption exists for a different reason: a profile whose ENVIRONMENT has no second
vendor it can authenticate against. A ladder is worth only what it can CONSTRUCT — 612fd451
declared a ``deepseek`` tier across 11 profiles and not one of them could be built, so every
failover logged "DeepSeek API key required" and recovered nothing (the incident that produced
``summarization.fallback.preflight_fallback_chain``). Declaring a tier for such a profile does
not add failover; it trades a clear "the gateway is down" error for a confusing auth error at
the worst moment.

Those are the only exemptions, and they are matched by path so a new production profile cannot
accidentally inherit the exemption by being named carelessly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import yaml

pytestmark = [pytest.mark.unit]

PROFILE_DIR = Path(__file__).resolve().parents[3] / "config" / "profiles"

#: Providers that run on hardware we own or in-process — no vendor account to exhaust, so a
#: single-vendor outage is not a risk in the same way.
LOCAL_PROVIDERS = {"spacy", "transformers", "summllama", "ollama", "vllm", "hybrid_ml", None, ""}


def _profiles() -> List[Tuple[Path, Dict[str, Any]]]:
    out = []
    for path in sorted(PROFILE_DIR.rglob("*.yaml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(doc, dict):
            out.append((path, doc))
    return out


def _is_measurement_profile(path: Path) -> bool:
    rel = path.relative_to(PROFILE_DIR)
    return rel.name.startswith("bakeoff_") or rel.parts[0] == "freeze"


#: Profiles whose environment cannot authenticate a SECOND vendor, so no declarable tier could be
#: built. Listed BY PATH, one entry at a time, so the exemption cannot be inherited by naming and
#: cannot quietly grow — ``test_the_no_second_vendor_exemption_stays_narrow`` pins the contents.
#:
#: ``homelab_balanced`` reaches its models through the homelab LiteLLM gateway, whose virtual key
#: resolves ``homelab-flash-0731`` and nothing else; the direct-DeepSeek tier cloud_balanced falls
#: back to needs a DEEPSEEK_API_KEY that does not exist in that environment. Vendor-level outage IS
#: still covered one layer down, by the gateway's own ``provider.order`` chain. What is not covered
#: is the gateway itself going down, and the profile says so in as many words.
_NO_AUTHENTICABLE_SECOND_VENDOR = frozenset({"homelab_balanced.yaml"})


def _makes_cloud_llm_calls(doc: Dict[str, Any]) -> bool:
    if not (doc.get("generate_summaries", True) or doc.get("generate_gi", False)):
        return False
    return (doc.get("summary_provider") not in LOCAL_PROVIDERS) or (
        doc.get("speaker_detector_provider") not in LOCAL_PROVIDERS
    )


class TestEveryOperationalProfileCanFailOver:
    def test_no_cloud_llm_profile_is_missing_a_ladder(self) -> None:
        missing = [
            str(p.relative_to(PROFILE_DIR))
            for p, doc in _profiles()
            if _makes_cloud_llm_calls(doc)
            and not _is_measurement_profile(p)
            and str(p.relative_to(PROFILE_DIR)) not in _NO_AUTHENTICABLE_SECOND_VENDOR
            and not doc.get("summary_fallback_providers")
        ]
        assert not missing, (
            f"profiles make cloud LLM calls with no failover ladder: {missing}. Add "
            "summary_fallback_providers naming a DIFFERENT vendor. If the profile is "
            "registry-governed set it on its ProfilePreset and run `make profiles-materialize`; "
            "hand-maintained profiles set it directly in the YAML."
        )

    def test_the_no_second_vendor_exemption_stays_narrow(self) -> None:
        """The exemption above is the dangerous kind: it makes a profile invisible to the rule.

        So pin it. The contents are asserted exactly, the profiles the rule exists FOR are asserted
        out of it, and the exempt profile must carry an EMPTY ladder rather than a partial one — an
        empty list is a decision, a half-declared ladder is an oversight, and only the first earns
        the exemption.
        """
        assert _NO_AUTHENTICABLE_SECOND_VENDOR == {"homelab_balanced.yaml"}
        for name in (
            "cloud_balanced.yaml",
            "cloud_openrouter.yaml",
            "reprocess_v22_community1.yaml",
            "reprocess_v23_turbo.yaml",
        ):
            assert name not in _NO_AUTHENTICABLE_SECOND_VENDOR, name
        doc = yaml.safe_load((PROFILE_DIR / "homelab_balanced.yaml").read_text(encoding="utf-8"))
        assert (
            doc.get("summary_fallback_providers") == []
        ), "the exempt profile must declare an empty ladder deliberately, not omit the key"

    def test_the_corpus_repair_profiles_have_one(self) -> None:
        """Called out separately because these re-derive an EXISTING corpus. An outage mid-run
        does not merely lose new work — it can replace good artifacts with degraded ones."""
        for name in ("reprocess_v22_community1.yaml", "reprocess_v23_turbo.yaml"):
            doc = yaml.safe_load((PROFILE_DIR / name).read_text(encoding="utf-8"))
            assert doc.get("summary_fallback_providers"), f"{name} would repair the corpus blind"

    def test_the_openrouter_profile_has_one(self) -> None:
        """The specific hole: the profile named after OpenRouter had no protection from
        OpenRouter."""
        doc = yaml.safe_load((PROFILE_DIR / "cloud_openrouter.yaml").read_text(encoding="utf-8"))
        assert doc.get("summary_fallback_providers")

    def test_no_ladder_names_its_own_primary(self) -> None:
        """Failing over to the provider that just failed is not failover. The runtime drops it
        anyway, but a config that says so is a config nobody trusts."""
        offenders = []
        for path, doc in _profiles():
            chain = [str(x).strip().lower() for x in (doc.get("summary_fallback_providers") or [])]
            primary = str(doc.get("summary_provider") or "").strip().lower()
            if primary and primary in chain:
                offenders.append(f"{path.relative_to(PROFILE_DIR)} ({primary})")
        assert not offenders, f"ladder names its own primary: {offenders}"

    def test_every_named_tier_is_a_real_provider(self) -> None:
        """A typo'd tier fails over into an unbuildable provider — a second failure at the worst
        moment, discovered only during an outage."""
        buildable = {
            "anthropic",
            "deepseek",
            "gemini",
            "grok",
            "groq",
            "litellm",
            "mistral",
            "ollama",
            "openai",
            "qwen",
            "summllama",
            "transformers",
            "vllm",
        }
        bad = []
        for path, doc in _profiles():
            for tier in doc.get("summary_fallback_providers") or []:
                if str(tier).strip().lower() not in buildable:
                    bad.append(f"{path.relative_to(PROFILE_DIR)} -> {tier}")
        assert not bad, f"unbuildable failover tier(s): {bad}"


class TestMeasurementProfilesStayPinned:
    """The exemption, asserted rather than assumed — so nobody 'helpfully' adds a ladder to an
    eval profile and silently blends two providers into one score."""

    def test_bakeoff_profiles_have_no_ladder(self) -> None:
        offenders = [
            str(p.relative_to(PROFILE_DIR))
            for p, doc in _profiles()
            if p.name.startswith("bakeoff_") and doc.get("summary_fallback_providers")
        ]
        assert not offenders, (
            f"bakeoff profiles must pin ONE provider; a ladder blends two into one "
            f"measurement: {offenders}"
        )

    def test_freeze_profiles_have_no_ladder(self) -> None:
        offenders = [
            str(p.relative_to(PROFILE_DIR))
            for p, doc in _profiles()
            if p.relative_to(PROFILE_DIR).parts[0] == "freeze"
            and doc.get("summary_fallback_providers")
        ]
        assert not offenders, f"a frozen profile with a ladder is not frozen: {offenders}"


class TestTheRegistryStaysTheSourceOfTruth:
    """ADR-112: for registry-governed profiles the ladder is materialized, never hand-edited."""

    def test_registry_governed_profiles_match_the_registry(self) -> None:
        import subprocess
        import sys

        repo = PROFILE_DIR.parents[1]
        result = subprocess.run(
            [sys.executable, "scripts/config/materialize_profiles.py", "--check"],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            "profiles are stale against the registry — run `make profiles-materialize`:\n"
            f"{result.stdout}\n{result.stderr}"
        )
