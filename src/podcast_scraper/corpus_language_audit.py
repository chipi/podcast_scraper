"""What language is every episode in this corpus, and WHERE did that answer come from? (#2175)

Read-only. This is the gate S0.6 must not land without, because after S0.6 a feed whose RSS says
``de`` is genuinely sent to the DGX as German — so the corpus's real distribution has to be known
first, not assumed.

THE FAILURE THIS IS BUILT TO AVOID. "The corpus is English" is an assertion nobody has checked.
The obvious audit re-reads the run configuration and reports a confident 100% ``en`` — which is
indistinguishable from a genuinely English corpus, and is the same shape as
``capability_audit``'s "0/36 openings, defect rate 0.0%": a check that found nothing and a corpus
with nothing wrong look identical in the output.

Two things make this one able to fail:

* it reports the **resolution source** for every episode, not just the language. A corpus where
  every row says ``profile_default`` has told you about the profile, not the corpus;
* ``ok`` is False when NO episode resolved from ``rss``. On a corpus that has been through
  ``m0021`` that means the backfill has not run, and a clean-looking report would be a lie.

Ordering matters for the same reason: run this AFTER m0021. Before it, no artifact carries a
publisher-declared language at all, so the audit can only echo the config.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .languages import is_language_enabled, normalize_language_tag

#: Sources an episode's language can carry, in precedence order. A value outside this set means
#: something wrote a source this audit does not know about, which is itself worth reporting.
KNOWN_SOURCES = ("override", "rss", "profile_default")


@dataclass
class EpisodeLanguage:
    """One episode's resolved language and where it came from."""

    relpath: str
    feed_id: str
    episode_id: str
    language: Optional[str]
    language_raw: Optional[str]
    source: Optional[str]

    @property
    def enabled(self) -> bool:
        return is_language_enabled(self.language)


@dataclass
class LanguageAuditReport:
    """The corpus's language distribution, with provenance."""

    episodes: List[EpisodeLanguage] = field(default_factory=list)
    unparsable: List[str] = field(default_factory=list)

    @property
    def by_language(self) -> Counter:
        return Counter(e.language or "<none>" for e in self.episodes)

    @property
    def by_source(self) -> Counter:
        return Counter(e.source or "<none>" for e in self.episodes)

    @property
    def non_english(self) -> List[EpisodeLanguage]:
        return [e for e in self.episodes if e.language != "en"]

    @property
    def not_enabled(self) -> List[EpisodeLanguage]:
        """Episodes in a language the registry does not enable — what S0.8 would skip."""
        return [e for e in self.episodes if not e.enabled]

    @property
    def unknown_sources(self) -> List[EpisodeLanguage]:
        return [e for e in self.episodes if e.source and e.source not in KNOWN_SOURCES]

    @property
    def measured(self) -> bool:
        """Did ANY episode's language come from a publisher rather than our own config?

        The one question that separates a measurement from an echo.
        """
        return any(e.source == "rss" for e in self.episodes)

    @property
    def ok(self) -> bool:
        """False when the report cannot be trusted as a measurement of the CORPUS.

        Deliberately not "False when a non-English episode exists" — finding one is a result,
        not a defect. What is a defect is a report that only measured our own configuration, or
        one carrying a source this audit does not understand.
        """
        if not self.episodes:
            return False
        return self.measured and not self.unknown_sources


def _load(path: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, str(exc)
    return (payload, "") if isinstance(payload, dict) else (None, "not a JSON object")


def _block(payload: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = payload.get(name)
    return value if isinstance(value, dict) else {}


def assess_languages(corpus_root: Path) -> LanguageAuditReport:
    """Walk every served episode and record its language and resolution source. Read-only."""
    from .upgrade.corpus_selection import select_served_artifacts

    report = LanguageAuditReport()
    served, _superseded = select_served_artifacts(corpus_root, ".metadata.json")
    for path in served:
        payload, err = _load(path)
        if payload is None:
            report.unparsable.append(f"{path.name}: {err}")
            continue
        feed = _block(payload, "feed")
        episode = _block(payload, "episode")
        # The EPISODE's language is the answer; the feed's is the fallback for a corpus written
        # before the episode pair existed. Normalizing on read means a pre-#2174 artifact
        # carrying "en-us" is reported as "en" rather than as a separate language.
        language = episode.get("language") or feed.get("language")
        source = episode.get("language_source") or feed.get("language_source")
        report.episodes.append(
            EpisodeLanguage(
                relpath=str(path.relative_to(corpus_root)),
                feed_id=str(feed.get("feed_id") or "?"),
                episode_id=str(episode.get("episode_id") or path.stem),
                language=normalize_language_tag(language) if isinstance(language, str) else None,
                language_raw=(
                    feed.get("language_raw") if isinstance(feed.get("language_raw"), str) else None
                ),
                source=str(source) if source else None,
            )
        )
    return report


def format_report(report: LanguageAuditReport) -> str:
    """Operator-facing: the distribution, then every item that is not plain enabled English."""
    lines = ["CORPUS LANGUAGE AUDIT", f"  episodes scanned : {len(report.episodes)}"]
    if not report.episodes:
        lines.append("")
        lines.append("  NO EPISODES FOUND — nothing was measured, so this is not a pass.")
        lines.append("VERDICT: FAIL")
        return "\n".join(lines)

    lines.append("")
    lines.append("  by language:")
    for lang, n in sorted(report.by_language.items(), key=lambda kv: (-kv[1], kv[0])):
        mark = "" if is_language_enabled(lang) else "   (not enabled in the registry)"
        lines.append(f"    {lang:<10} {n:5d}{mark}")

    lines.append("")
    lines.append("  by resolution source:")
    for src, n in sorted(report.by_source.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"    {src:<16} {n:5d}")
    if not report.measured:
        lines.append(
            "    ^ NO episode resolved from 'rss'. This report measured the CONFIGURATION, "
            "not the corpus — run the m0021 backfill first."
        )

    if report.non_english:
        lines.append("")
        lines.append(f"  NON-ENGLISH — {len(report.non_english)} episode(s):")
        for e in report.non_english[:50]:
            raw = f" raw={e.language_raw!r}" if e.language_raw else ""
            lines.append(
                f"    {e.feed_id}/{e.episode_id}  language={e.language!r} "
                f"source={e.source!r}{raw}"
            )
        if len(report.non_english) > 50:
            lines.append(f"    … and {len(report.non_english) - 50} more")

    if report.not_enabled:
        feeds = sorted({e.feed_id for e in report.not_enabled})
        lines.append("")
        lines.append(
            f"  NOT ENABLED — {len(report.not_enabled)} episode(s) across {len(feeds)} show(s) "
            "are in a language the registry does not enable. S0.8 would skip these."
        )
        for feed_id in feeds[:20]:
            langs = sorted(
                {e.language or "<none>" for e in report.not_enabled if e.feed_id == feed_id}
            )
            lines.append(f"    {feed_id}: {', '.join(langs)}")

    if report.unknown_sources:
        lines.append("")
        lines.append(
            f"  UNKNOWN SOURCE — {len(report.unknown_sources)} episode(s) carry a "
            f"language_source outside {KNOWN_SOURCES}:"
        )
        for e in report.unknown_sources[:20]:
            lines.append(f"    {e.feed_id}/{e.episode_id}  source={e.source!r}")

    if report.unparsable:
        lines.append("")
        lines.append(f"  UNPARSABLE — {len(report.unparsable)}:")
        for item in report.unparsable[:20]:
            lines.append(f"    {item}")

    lines.append("")
    lines.append(f"VERDICT: {'PASS' if report.ok else 'FAIL'}")
    if report.ok and not report.non_english:
        lines.append("  Every episode is English, and at least one publisher said so itself.")
    return "\n".join(lines)


def check_corpus(corpus_root: Path) -> Tuple[bool, str]:
    """Convenience: ``(ok, formatted_report)``. ``ok`` is the trustworthy-measurement verdict."""
    report = assess_languages(corpus_root)
    return report.ok, format_report(report)
