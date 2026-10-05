"""Withdraw a language's episodes from DISCOVERY, and put them back (S3.2).

WHAT `enabled` MEANS, AND WHAT IT DOES NOT. `enabled: false` in ``config/languages.yaml`` gates
exactly one thing: INGEST. ``episode_processor`` refuses a feed whose resolved language is not
enabled, with a reason naming the source. Nothing on the read side consults it — not the server
routes, not search — so disabling a language that already has published episodes leaves those
episodes fully served. That is deliberate, decided 2026-10-01:

* A one-line YAML edit must not silently change what listeners can see. If `enabled: false` also
  hid episodes, the blast radius of a typo would be a content outage with no confirmation step.
* Filtering by language on every read path would break D-39 ("stages never branch by language") —
  it would put a language branch in search, the rails, recommendations and the API, which is the
  thing Phase 2 was built to avoid.
* Rollback is an operator ACTION with a blast radius, so it deserves a command with a dry run,
  not a config state whose effect you discover later.

So withdrawal lives here, as something you run on purpose.

HOW IT WORKS, AND WHY IT IS REVERSIBLE BY CONSTRUCTION. Withdrawal removes the language's episode
rows from the SEARCH INDEX and touches nothing else. The index is derived from the corpus, so the
corpus — transcripts, metadata, artwork, the English renders — is left exactly as it was, and the
reversal is an ordinary reindex. There is no `withdrawn` flag to add, no new state for the rest of
the system to learn, and nothing to migrate if the decision is undone.

WHAT A LISTENER SEES. The episodes stop appearing in search, the rails and recommendations, which
is what "withdrawn" has to mean to be worth anything. A saved link or a library entry still
resolves and still plays: the artifacts are on disk and the episode routes read them directly.
That is the honest limit of this procedure and it is stated rather than hidden — if a language has
to become completely unreachable, that is a corpus deletion, which is a different and
irreversible operation this module does not perform.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .corpus_language_audit import assess_languages
from .languages import language_registry, normalize_language_tag

logger = logging.getLogger(__name__)

#: The tiers an episode's rows can live in. `segment_nonen` is the one that holds a non-English
#: episode's source body, so omitting it would leave exactly the rows a withdrawal is aimed at.
_TIERS = ("segment", "segment_nonen", "insight", "aux")


@dataclass
class WithdrawalPlan:
    """What a withdrawal would remove. Produced read-only, so a dry run is the default."""

    language: Optional[str]
    episodes: List[str] = field(default_factory=list)
    feeds: List[str] = field(default_factory=list)
    index_present: bool = True
    error: Optional[str] = None

    @property
    def empty(self) -> bool:
        return not self.episodes


@dataclass
class WithdrawalResult:
    """What a withdrawal actually removed, per tier."""

    language: Optional[str]
    episodes: List[str] = field(default_factory=list)
    rows_removed: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def total_rows(self) -> int:
        return sum(self.rows_removed.values())


def plan_withdrawal(corpus_root: Path, language: str) -> WithdrawalPlan:
    """Which episodes a withdrawal of *language* would remove from the index. Read-only.

    Normalizes the requested tag the same way the audit does, so asking for ``es-ES`` finds the
    episodes stored as ``es`` rather than silently matching nothing — a withdrawal that reports
    "0 episodes" because of a tag mismatch is the worst possible outcome here, since it reads as
    "there was nothing to withdraw".
    """
    normalized = normalize_language_tag(language)
    if not normalized:
        return WithdrawalPlan(language=None, error=f"{language!r} is not a usable language tag")
    # VALIDATED AGAINST THE REGISTRY, not just normalized. `normalize_language_tag` strips the
    # region and nothing more, so `zzz` and a typo like `nto` both survive it and would then match
    # zero episodes — and this tool reporting "nothing to withdraw" for a typo is the worst outcome
    # it has, because it reads as "there was nothing there". A tag the registry has never heard of
    # is an operator mistake, so say so instead.
    if normalized not in language_registry():
        return WithdrawalPlan(
            language=None,
            error=(
                f"{language!r} (normalized {normalized!r}) is not declared in "
                "config/languages.yaml — check the tag rather than reading a zero as "
                "'nothing to withdraw'"
            ),
        )
    if normalized == "en":
        # Refused rather than supported: English is the default on every surface (D-38), so
        # withdrawing it would empty the product. If that is ever genuinely wanted it should be
        # a deliberate corpus operation, not a language flag's rollback path.
        return WithdrawalPlan(
            language=normalized,
            error="refusing to withdraw English — it is the default on every surface (D-38)",
        )

    report = assess_languages(corpus_root)
    matching = [e for e in report.episodes if e.language == normalized]
    plan = WithdrawalPlan(
        language=normalized,
        episodes=sorted({e.episode_id for e in matching}),
        feeds=sorted({e.feed_id for e in matching}),
        index_present=(corpus_root / "search" / "lance_index").is_dir(),
    )
    return plan


def format_plan(plan: WithdrawalPlan) -> str:
    """The dry run an operator reads before committing."""
    if plan.error:
        return f"REFUSED: {plan.error}"
    lines = [f"Withdrawal plan for {plan.language!r}:"]
    if plan.empty:
        lines.append("  no episodes in this language — nothing to withdraw")
        return "\n".join(lines)
    lines.append(f"  {len(plan.episodes)} episode(s) across {len(plan.feeds)} feed(s)")
    lines.append(f"  feeds: {', '.join(plan.feeds)}")
    if not plan.index_present:
        lines.append("  NOTE: no search index on disk, so there is nothing to remove from")
    lines.append("")
    lines.append("  Removes their rows from the search index ONLY. The corpus is untouched, so")
    lines.append("  this is undone by a reindex: `cli index-two-tier --output-dir <corpus>`.")
    lines.append("  Direct links and library entries keep playing — see the module docstring.")
    return "\n".join(lines)


def apply_withdrawal(corpus_root: Path, language: str) -> WithdrawalResult:
    """Remove *language*'s episode rows from the search index.

    Deliberately NOT idempotent-by-marker: it just deletes rows, so running it twice removes
    nothing the second time and reports zero. There is no state to get out of step.
    """
    plan = plan_withdrawal(corpus_root, language)
    if plan.error:
        return WithdrawalResult(language=plan.language, error=plan.error)
    result = WithdrawalResult(language=plan.language, episodes=list(plan.episodes))
    if plan.empty or not plan.index_present:
        return result

    from .search.backends.lancedb_backend import LanceDBBackend

    index_dir = corpus_root / "search" / "lance_index"
    try:
        backend = LanceDBBackend(str(index_dir))
    except Exception as exc:  # noqa: BLE001 - an unopenable index means nothing was withdrawn
        return WithdrawalResult(
            language=plan.language, error=f"could not open the index at {index_dir}: {exc}"
        )

    for tier in _TIERS:
        removed = 0
        for episode_id in plan.episodes:
            # `keep_ids` empty means "remove every row for this episode in this tier", which is
            # exactly a withdrawal. The same primitive the indexer uses to prune a superseded run.
            removed += int(backend.prune_episode_rows(tier, episode_id, keep_ids=set()))
        if removed:
            result.rows_removed[tier] = removed
    logger.info(
        "withdrew language %s: %d episode(s), %d index row(s) across %s",
        plan.language,
        len(result.episodes),
        result.total_rows,
        ", ".join(sorted(result.rows_removed)) or "no tier",
    )
    return result
