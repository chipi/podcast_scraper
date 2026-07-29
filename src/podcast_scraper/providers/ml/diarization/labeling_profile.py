"""ADR-138 — versioned labeling profiles (a knob-bundle, not a code fork).

Cleaning is versioned as a registry of pure ``text -> text`` functions (ADR-017). Labeling is a
stateful, LLM-coupled pipeline, so we version its *configuration* instead: a frozen
:class:`LabelingProfile` bundles the tunables + per-fix feature flags that used to live as scattered
constants in ``roster.py``, behind one greppable, comparable unit selected by ID.

This is what makes A/B automatable — an agent proposes a new profile (different flags/knobs), and
the corpus is re-run with ``labeling_profile: <id>`` and compared to the baseline, no code edit. A
future *algorithm* (not expressible as a knob) is added as a strategy on the same profile — an
extension, not a rebuild (ADR-138 tier 3).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List


@dataclass(frozen=True)
class LabelingProfile:
    """A versioned bundle of labeling knobs + per-fix feature flags.

    ``version`` aligns with ``METHOD_VERSIONS["naming"]`` and is stamped on every episode's sidecar,
    so a run declares which labeling behaviour produced it. Flags default ON (the ``naming-4``
    behaviour); a legacy/experimental profile flips them to isolate a variable.
    """

    name: str
    version: str

    # --- per-fix feature flags (ADR-137) ---------------------------------------------------------
    narrator_cue_binding: bool = True  # case-blind metadata-anchored cue path in the intro reader
    case_blind_self_intro: bool = True  # lowercase "i'm X" -> stated name (metadata-anchored)
    nickname_fuzzy_binding: bool = True  # Rich<->Richard + ASR-fuzzy surname in the canonicalizer
    pattern_b_bounded_promotion: bool = True  # bound `unknown` to top-N; excess is `unidentified`
    alarm_on_defect_share: bool = (
        True  # alarm on the `unknown` defect share, not total unattributed
    )
    # host-intro by a bare FIRST name ("here with akshat") binds when exactly one stated name has
    # it (flightcast group intros). Cue path only — a bare "i'm rich" self-intro must not bind.
    first_name_only_intro: bool = True
    # a cluster whose self-intros map to 2+ different stated people is a diarization MERGE ("I'm
    # Lucas and I'm Axel" in the host's cluster) — suppress its name rather than paint the first one
    # onto it. Removes a wrong name; does not recover the merged guests (that needs re-diarization).
    suppress_merged_speaker_clusters: bool = True

    # --- numeric knobs ---------------------------------------------------------------------------
    unattributed_alarm_threshold: float = 0.25  # CONSUMED by build_speaker_diagnostics
    # CONSUMED: threaded to all five cameo sites (classify_voices, _classify_voice_types,
    # _name_guest_voices, _self_intro_voice_names -> host-candidate). Default equals the module
    # constant CAMEO_MAX_TALK_S, so naming-4 is unchanged; a profile that raises it re-tiers cameos.
    cameo_max_talk_s: float = 20.0


_REGISTRY: Dict[str, LabelingProfile] = {}


def register_profile(profile: LabelingProfile) -> None:
    """Register a labeling profile under its ``version`` id (mirrors ADR-017's registry API)."""
    _REGISTRY[profile.version] = profile


def get_profile(version: str) -> LabelingProfile:
    """Look up a registered profile by version id; raises ValueError on an unknown id."""
    if version not in _REGISTRY:
        raise ValueError(f"Labeling profile {version!r} not found. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[version]


def list_profiles() -> List[str]:
    """All registered profile version ids."""
    return sorted(_REGISTRY)


# The production profile — every fix on, current thresholds. Its version is the reprocess key.
NAMING_4 = LabelingProfile(name="production", version="naming-4")

# A legacy baseline: every ADR-137 fix + Pattern-B OFF, alarm on TOTAL unattributed — i.e. the
# pre-today behaviour. Exists so "naming-4 vs naming-3-legacy" is a one-line A/B on the same corpus.
NAMING_3_LEGACY = replace(
    NAMING_4,
    name="legacy",
    version="naming-3-legacy",
    narrator_cue_binding=False,
    case_blind_self_intro=False,
    nickname_fuzzy_binding=False,
    pattern_b_bounded_promotion=False,
    alarm_on_defect_share=False,
    first_name_only_intro=False,
    suppress_merged_speaker_clusters=False,
)

register_profile(NAMING_4)
register_profile(NAMING_3_LEGACY)

DEFAULT_LABELING_PROFILE = NAMING_4
