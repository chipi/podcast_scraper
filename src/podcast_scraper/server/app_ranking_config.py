"""Ranking-signal registry — the single source of truth for discovery/digest ranking.

Every contributing signal is independently **on/off** and **weight-tunable**, with optional
signal-specific ``params``, so ranking can be tuned and A/B'd without code changes. This is the
config the operator surface edits (#11 of the consumer-remember batch: "build recommendations so
all contributing elements can be on/off and, where they carry a value, configurable, and manage
it all in one place"). ``rank_discover`` composes the enabled signals; a new signal slots in by
adding a :class:`RankingSignal` to :data:`DEFAULT_RANKING_CONFIG` plus one term in the composition.

An A/B variant is just a different :class:`RankingConfig` instance — no code branch. Parsing from
a config dict (operator-config / admin view) is total: any missing or invalid field falls back to
the default for that signal, so a bad override can never empty the ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Signal names — the composition in ``rank_discover`` references these constants.
SIGNAL_SIGNIFICANCE = "significance"
SIGNAL_INTEREST_AFFINITY = "interest_affinity"
SIGNAL_TREND_VELOCITY = "trend_velocity"
SIGNAL_RECENCY = "recency"


@dataclass(frozen=True)
class RankingSignal:
    """One ranking contributor: on/off, a tunable weight, and signal-specific params."""

    name: str
    enabled: bool = True
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RankingConfig:
    """An immutable snapshot of the ranking signals. An A/B variant is a different instance."""

    signals: tuple[RankingSignal, ...]

    def get(self, name: str) -> RankingSignal | None:
        for s in self.signals:
            if s.name == name:
                return s
        return None

    def is_enabled(self, name: str) -> bool:
        s = self.get(name)
        return bool(s and s.enabled)

    def weight_of(self, name: str, default: float = 0.0) -> float:
        """The configured weight when the signal is enabled, else ``default`` (0 = signal off)."""
        s = self.get(name)
        return s.weight if (s is not None and s.enabled) else default

    def params_of(self, name: str) -> dict[str, Any]:
        s = self.get(name)
        return dict(s.params) if s is not None else {}


DEFAULT_RANKING_CONFIG = RankingConfig(
    signals=(
        RankingSignal(
            SIGNAL_SIGNIFICANCE,
            enabled=True,
            weight=1.0,
            params={"gi_bonus": 2.0, "kg_bonus": 1.0, "bullet_step": 0.2, "bullet_cap": 5},
        ),
        RankingSignal(SIGNAL_INTEREST_AFFINITY, enabled=True, weight=2.0),
        # Trend defaults OFF until tuned on real engagement — like the whole personalization path.
        RankingSignal(SIGNAL_TREND_VELOCITY, enabled=False, weight=0.4, params={"cap": 1.5}),
        # Recency: the stable newest-first tie-break / no-interest fallback (weight unused today).
        RankingSignal(SIGNAL_RECENCY, enabled=True, weight=0.0),
    )
)


def _coerce_signal(item: Any, base: RankingSignal | None) -> RankingSignal | None:
    """One config-dict entry → a RankingSignal, defaulting each field from *base* when invalid."""
    if not isinstance(item, dict):
        return None
    name = item.get("name")
    if not isinstance(name, str) or not name:
        return None
    enabled = bool(item.get("enabled", base.enabled if base else True))
    try:
        weight = float(item.get("weight", base.weight if base else 1.0))
    except (TypeError, ValueError):
        weight = base.weight if base else 1.0
    params = item.get("params")
    if not isinstance(params, dict):
        params = dict(base.params) if base else {}
    return RankingSignal(name, enabled=enabled, weight=weight, params=params)


def ranking_config_from_dict(data: Any) -> RankingConfig:
    """Parse a config dict (operator-config / admin view) into a :class:`RankingConfig`.

    Overrides **merge** onto the defaults: a signal named in ``data['signals']`` replaces that
    default (any omitted field inherited from it), unnamed defaults are kept, and an unknown
    signal name is appended. Total by construction — a non-dict, a missing ``signals`` list, or
    no valid entry all return :data:`DEFAULT_RANKING_CONFIG`, so a malformed override never
    empties ranking or silently drops a signal the composition relies on.
    """
    if not isinstance(data, dict):
        return DEFAULT_RANKING_CONFIG
    raw_signals = data.get("signals")
    if not isinstance(raw_signals, list):
        return DEFAULT_RANKING_CONFIG
    by_name = {s.name: s for s in DEFAULT_RANKING_CONFIG.signals}
    overrides: dict[str, RankingSignal] = {}
    for item in raw_signals:
        sig = _coerce_signal(item, by_name.get(_name_of(item)))
        if sig is not None:
            overrides[sig.name] = sig
    if not overrides:
        return DEFAULT_RANKING_CONFIG
    merged = [overrides.pop(s.name, s) for s in DEFAULT_RANKING_CONFIG.signals]
    merged.extend(overrides.values())  # unknown signal names appended in encounter order
    return RankingConfig(signals=tuple(merged))


def _name_of(item: Any) -> str:
    if isinstance(item, dict):
        name = item.get("name")
        if isinstance(name, str):
            return name
    return ""


def ranking_config_to_dict(config: RankingConfig) -> dict[str, Any]:
    """Serialize for the operator-config API / admin view."""
    return {
        "signals": [
            {"name": s.name, "enabled": s.enabled, "weight": s.weight, "params": dict(s.params)}
            for s in config.signals
        ]
    }


__all__ = [
    "RankingSignal",
    "RankingConfig",
    "DEFAULT_RANKING_CONFIG",
    "ranking_config_from_dict",
    "ranking_config_to_dict",
    "SIGNAL_SIGNIFICANCE",
    "SIGNAL_INTEREST_AFFINITY",
    "SIGNAL_TREND_VELOCITY",
    "SIGNAL_RECENCY",
]
