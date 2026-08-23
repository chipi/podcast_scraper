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

#: NOT a scoring signal — the ADMISSION policy: how many candidates enter ranking at all.
#:
#: It lives here anyway, and that is deliberate. Everything else in this file re-ORDERS the
#: candidates; this decides who is in the room, and no weight can promote an episode the pool
#: excluded. Keeping it outside the config meant the single most consequential ranking parameter
#: was a module constant nothing could override — which is how it stayed a fixed 48 episodes
#: while the corpus grew to 678 (#1682). Any future sweep (#1795) has to be able to vary it.
SIGNAL_DISCOVER_POOL = "discover_pool"


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
        """The signal with this name, or ``None`` when the config has no such signal."""
        for s in self.signals:
            if s.name == name:
                return s
        return None

    def is_enabled(self, name: str) -> bool:
        """Whether the named signal exists and is enabled."""
        s = self.get(name)
        return bool(s and s.enabled)

    def weight_of(self, name: str, default: float = 0.0) -> float:
        """The configured weight when the signal is enabled, else ``default`` (0 = signal off)."""
        s = self.get(name)
        return s.weight if (s is not None and s.enabled) else default

    def params_of(self, name: str) -> dict[str, Any]:
        """A copy of the named signal's params, or an empty dict when there is no such signal."""
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
        # Affinity SATURATES per matched token; it is no longer matched/len(interests) (#19).
        #
        # That denominator punished engagement: one match was worth x2.0 with two follows and x1.1
        # with twenty, so personalisation faded for the users who had told the product the most
        # about themselves. Following one more show is not a statement that everything else matters
        # less. The boost now depends on HOW MANY of your interests an episode matches, never on
        # how many you hold.
        #
        # derived_ratio 0.5 — an explicit follow is a stated preference, a derived token is an
        # inference, so an inference is worth about half a statement. They are counted separately
        # (see rank_discover): pooled into one denominator, enabling APP_DERIVED_INTERESTS dropped a
        # 2-follow user's per-match affinity from 0.5 to 0.2, i.e. switching implicit
        # personalisation ON made the user's own follows count for LESS. Separate weights mean it
        # can only ever add.
        #
        # cap 1.0 — the saturation ceiling, so a broad episode matching six interests cannot run
        # away with the feed: 1 match -> half the cap, 2 -> three quarters, never beyond it.
        # weight 4.0, NOT the 2.0 it was before saturation. Measured, not guessed: saturation makes
        # one match worth `weight * (1 - 0.5^1)` = half the weight, so 2.0 would have HALVED the
        # boost a single-interest user gets and the eval fell 0.981 -> 0.835. 4.0 restores a
        # one-match boost of exactly 2.0 — the value everything else was tuned against — and scores
        # 0.984. Higher keeps climbing (6.0 -> 0.999) only by letting affinity swamp recency and
        # significance, which would quietly undo #22 and #23.
        RankingSignal(
            SIGNAL_INTEREST_AFFINITY,
            enabled=True,
            weight=4.0,
            params={"derived_ratio": 0.5, "cap": 1.0},
        ),
        # Trend defaults OFF until tuned on real engagement — like the whole personalization path.
        RankingSignal(SIGNAL_TREND_VELOCITY, enabled=False, weight=0.4, params={"cap": 1.5}),
        # Recency is a GRADED boost, not just the newest-first tie-break it used to be (#22).
        #
        # weight 0.5 against affinity's 2.0: a real interest match should still outrank pure
        # freshness — following something has to mean more than "this is new" — but an unrelated
        # older episode should no longer leapfrog today's just because it happens to carry a richer
        # KG. Decay runs from the NEWEST episode in the pool, not wall-clock (see _recency_boost).
        #
        # BOTH NUMBERS ARE MEASURED, not chosen by feel, against
        # scripts/eval/score/rank_discover_v1.py — which now scores the config that actually ships
        # (#21). On the committed 36-episode corpus (span 2024-01-02 -> 2026-07-16, ~925 days),
        # following one topic and looking at the episodes that do NOT match it:
        #
        #   half-life  w=0.5 unrelated-in-time-order   interest still top-3
        #        30d              94.4%  (unchanged)          yes
        #       365d              98.4%                       yes
        #       730d              97.2%                       yes
        #
        # 30 days is DEAD on a corpus this sparse: the second-newest episode is already months old,
        # so its boost is 0.014 and everything below it is 0. That is the trap — a half-life that
        # sounds right for "when does an episode stop feeling current" silently does nothing when
        # the corpus does not publish that often.
        #
        # So the right value scales with PUBLISHING CADENCE, not with intuition about freshness. A
        # corpus that ships weekly wants a far shorter half-life than this one; re-run the eval
        # against your own corpus (--data-dir) rather than inheriting this because it is written
        # here. Gate at these values: mean nDCG 0.390 -> 0.978, uplift +0.588 (floor 0.5 / 0.05).
        # For reference: recency OFF scores 1.000/+0.610 and 30d scores 0.944/+0.555 — so the
        # measured half-life costs LESS personalisation quality than the intuitive one, while
        # being the only one that does anything at all.
        #
        # 365 -> 730, 2026-08-19. The reasoning above still holds; what changed is the question it
        # was answering. 365 was fitted to the corpus we HAD. The half-life should instead encode
        # WHEN PODCAST CONTENT GOES STALE, and hold still while the corpus grows underneath it —
        # otherwise it needs re-tuning every time the archive deepens, which is backwards.
        #
        # The target window is 2-4 years of content, with a tail out to ~10. What each value gives:
        #
        #        age        365d      730d     1095d
        #     1 year        0.50      0.71      0.79
        #    2 years        0.25      0.50      0.63
        #    4 years        0.06      0.25      0.40
        #   10 years        0.00      0.03      0.10
        #
        # At 365 a FOUR-YEAR-OLD episode scores 0.06 — effectively excluded from the freshness
        # signal while sitting inside the window we care about. At 730 it scores 0.25: still
        # competing, clearly aged. A 2-year episode keeps half its freshness, which is the middle
        # of the target window. 1095 was rejected as too generous — a decade-old episode still at
        # 0.10 should win on relevance or not at all.
        #
        # The personalisation cost is ~1pp: the table above measured 730d at 97.2% vs 365d's
        # 98.4% on the same corpus.
        #
        # Note this barely moves TODAY. On the 678-episode production corpus (span 556 days,
        # measured #1683) the oldest episode goes 0.35 -> 0.59, a mild re-ordering. It matters at
        # 10k episodes spanning years, which is the point: set it for where the corpus is going.
        #
        # And keep the scale in mind before over-investing here — recency's whole range is worth
        # ~0.5 in the score while ONE followed interest is worth 3.0. This tunes the shelf for a
        # user with no strong interests; it does not drive a personalised feed.
        RankingSignal(SIGNAL_RECENCY, enabled=True, weight=0.5, params={"half_life_days": 730.0}),
        # The candidate pool. `weight` is unused (this admits rather than scores); the policy is
        # entirely in params, and `enabled=False` would mean "no bound", which is not offered —
        # ranking is one KG artifact load per candidate, so an unbounded pool is a slow endpoint.
        #
        #   corpus_share      fraction of the corpus the window may reach. Measured on production
        #                     2026-08-19: the old fixed window reached 48/678 = 7.1%, so 630
        #                     episodes could not be surfaced at all unless they matched a follow.
        #   page_multiple     the window as a multiple of page size, which is what the window used
        #                     to be, full stop. Still the floor for small corpora.
        #   max_candidates    hard ceiling, so a large corpus cannot make /discover slow.
        #   min_limit_for_share  below this page size only `page_multiple` applies — a request for
        #                     one or two episodes is a probe or a widget, not a discovery feed.
        #
        # 0.15 is a judgement call, not a measurement, and it was chosen against a 678-episode
        # corpus that is expected to reach several thousand. #1795 exists to find out whether it
        # is even a live lever before anything searches over it.
        RankingSignal(
            SIGNAL_DISCOVER_POOL,
            enabled=True,
            weight=0.0,
            params={
                "corpus_share": 0.15,
                "page_multiple": 4,
                "max_candidates": 400,
                "min_limit_for_share": 5,
            },
        ),
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
