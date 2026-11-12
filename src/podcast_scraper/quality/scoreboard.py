"""The corpus scoreboard, from the installed packages that write what it counts.

The change gate (``scripts/measure/corpus_change_gate.py``) and the outcome hook
(``upgrade/outcome_hook.py``) score a corpus before and after a migration. What they count (hidden
insights, quotes off their text, ad copy left in ad-free text, ...) is defined by the package whose
artifacts it is, through :class:`podcast_scraper.extensions.CorpusScoreboard`. With none installed
there is nothing to count; a corpus that still holds files a scoreboard would read
(:data:`SCORED_FILES`) is refused rather than scored as unchanged, so a migration measured without
its scoreboard can never pass as LIGHT.
"""

from __future__ import annotations

import fnmatch
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..extensions import corpus_scoreboards

#: Files a scoreboard reads. Kept here, not in the repo's config, because the outcome hook runs in
#: an installed image that has no ``config/``.
SCORED_FILES: tuple[str, ...] = ("*.gi.json", "*.adfree.txt")


class ScoreboardMissing(RuntimeError):
    """The corpus holds scored files but no installed package scores them."""


def gating_metrics() -> List[str]:
    """Every installed scoreboard's gating metrics, in load order, each once."""
    out: List[str] = []
    for board in corpus_scoreboards():
        out.extend(m for m in board.gating_metrics if m not in out)
    return out


def score_episode(meta_path: Path) -> Optional[Dict[str, Any]]:
    """``{"language", "counts"}`` for one episode, the installed scoreboards' counts summed;
    ``None`` when none is installed or a scoreboard cannot read the metadata."""
    boards = corpus_scoreboards()
    if not boards:
        return None
    language: Optional[str] = None
    counts: Counter = Counter()
    for board in boards:
        row = board.score_episode(meta_path)
        if row is None:
            return None
        language = language or row.get("language")
        counts.update({k: int(v) for k, v in (row.get("counts") or {}).items()})
    return {"language": language or "", "counts": dict(counts)}


def _with_zeros(counts: Counter) -> Dict[str, int]:
    keys = [k for board in corpus_scoreboards() for k in board.keys]
    out = {k: int(counts.get(k, 0)) for k in keys}
    out.update({k: int(v) for k, v in counts.items() if k not in out})
    return dict(sorted(out.items()))


def score_corpus(root: Path) -> Dict[str, Any]:
    """The scoreboard over every served episode under ``root``: totals and per language."""
    from ..upgrade.corpus_selection import select_served_artifacts

    total: Counter = Counter()
    per_lang: Dict[str, Counter] = defaultdict(Counter)
    for meta in select_served_artifacts(root, ".metadata.json")[0]:
        row = score_episode(meta)
        if row is None:
            if corpus_scoreboards():
                total["metadata_unreadable"] += 1
            continue
        total.update(row["counts"])
        per_lang[row["language"]].update(row["counts"])
    return {
        "total": _with_zeros(total),
        "per_language": {k: _with_zeros(v) for k, v in sorted(per_lang.items())},
    }


def scored_files_present(root: Path, patterns: Iterable[str]) -> List[str]:
    """Up to three files under ``root`` matching any of ``patterns`` (``fnmatch`` on the name)."""
    pats = list(patterns)
    found: List[str] = []
    if not pats:
        return found
    for path in Path(root).rglob("*"):
        if path.is_file() and any(fnmatch.fnmatch(path.name, p) for p in pats):
            found.append(str(path.relative_to(root)))
            if len(found) == 3:
                break
    return found


def require_scoreboard(root: Path, patterns: Iterable[str] = SCORED_FILES) -> None:
    """Raise :class:`ScoreboardMissing` when ``root`` holds a scored file and no scoreboard is
    installed: measuring it without one would report every gating number as unchanged."""
    if corpus_scoreboards():
        return
    hits = scored_files_present(root, patterns)
    if hits:
        raise ScoreboardMissing(
            "no corpus scoreboard is installed, but the corpus holds files one would score "
            f"({', '.join(hits)}): install the package that writes them before measuring"
        )
