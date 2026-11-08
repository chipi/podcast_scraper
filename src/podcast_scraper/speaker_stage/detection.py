"""Feed hosts and episode guests: who the metadata says takes part (moved verbatim from
``workflow.stages.processing``, E04 S2).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, cast, Dict, Iterable, List, NamedTuple, Optional, Set

from podcast_scraper import config
from podcast_scraper.kg.speaker_coherence import same_person
from podcast_scraper.languages import transcription_language
from podcast_scraper.models import Episode, RssFeed
from podcast_scraper.speaker_detectors.corroboration import corroborate_guests
from podcast_scraper.speaker_detectors.factory import create_speaker_detector
from podcast_scraper.speaker_detectors.hosts import (
    compose_episode_hosts,
    detect_hosts_from_feed,
    distinct_self_introductions,
    drop_non_person_names,
    hosts_from_episode_description,
    hosts_from_feed_statement,
    is_network_or_org_author,
    normalize_host_names,
    recurrent_hosts_across_episodes,
)
from podcast_scraper.speaker_detectors.normalization import filter_default_speaker_names
from podcast_scraper.utils import filesystem
from podcast_scraper.utils.log_redaction import format_exception_for_log
from podcast_scraper.utils.optional_deps import caused_by_missing_import
from podcast_scraper.workflow import metrics
from podcast_scraper.workflow.stages.processing import extract_episode_description
from podcast_scraper.workflow.types import HostDetectionResult

logger = logging.getLogger(__name__)


def _flatten_speaker_name_entries(value: Any) -> List[str]:
    """Normalize speaker-detector output to flat, non-empty strings.

    LLM JSON occasionally nests names (e.g. ``[\"A\", \"B\"]`` or mixed lists);
    those values are not hashable and must not be used in ``set`` membership
    checks without flattening.
    """
    if value is None:
        return []
    if isinstance(value, str):
        t = value.strip()
        return [t] if t else []
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for v in value:
            out.extend(_flatten_speaker_name_entries(v))
        return out
    t = str(value).strip()
    return [t] if t else []


def _speaker_names_to_str_set(members: Any) -> Set[str]:
    """Build a string set from iterable of possibly nested speaker/host labels."""
    out: Set[str] = set()
    if members is None:
        return out
    for item in members:
        for s in _flatten_speaker_name_entries(item):
            out.add(s)
    return out


def _handle_dry_run_host_detection(
    feed: RssFeed,  # type: ignore[valid-type]
) -> HostDetectionResult:
    """Handle host detection in dry-run mode.

    Args:
        feed: Parsed RssFeed object

    Returns:
        HostDetectionResult with hosts from RSS author tags if available
    """
    logger.info("(dry-run) would initialize speaker detector")
    # Statement-first, matching the real run (audit F4): the feed's "Hosted by ..." blurb is pure
    # regex (no ML — safe in dry-run), so an org-authored feed whose description names its hosts
    # previews the same hosts the real run would find, instead of reporting none. NER is skipped
    # here (it needs the model dry-run deliberately avoids); author tags are the second source.
    cached_hosts: set[str] = set(
        hosts_from_feed_statement(feed.title, feed.description, getattr(feed, "language", "") or "")
    )
    if not cached_hosts and feed.authors:
        cached_hosts = {
            a
            for a in feed.authors
            if not is_network_or_org_author(a, language=_feed_language(feed))
        }
    if cached_hosts:
        logger.info(
            "DETECTED HOSTS (dry-run, feed statement / author tags): %s",
            ", ".join(sorted(cached_hosts)),
        )
    return HostDetectionResult(cached_hosts, None, None, language=_feed_language(feed))


def _create_speaker_detector_if_needed(
    cfg: config.Config, speaker_detector: Optional[Any]
) -> Optional[Any]:
    """Create speaker detector if not provided (backward compatibility).

    Args:
        cfg: Configuration object
        speaker_detector: Optional existing speaker detector

    Returns:
        Speaker detector instance or None if creation failed
    """
    if speaker_detector is not None:
        return speaker_detector

    # Fallback: create speaker detector if not provided (for backward compatibility)
    logger.warning(
        "speaker_detector not provided to detect_feed_hosts_and_patterns, "
        "creating new instance (this should be created in orchestration)"
    )
    try:
        import sys

        workflow_pkg = sys.modules.get("podcast_scraper.workflow")
        if workflow_pkg and hasattr(workflow_pkg, "create_speaker_detector"):
            func = getattr(workflow_pkg, "create_speaker_detector")
            from unittest.mock import Mock

            if isinstance(func, Mock):
                speaker_detector = func(cfg)
            else:
                speaker_detector = create_speaker_detector(cfg)
        else:
            from podcast_scraper.speaker_detectors.factory import (
                create_speaker_detector as factory_create_speaker_detector,
            )

            speaker_detector = factory_create_speaker_detector(cfg)
        # Initialize provider (loads spaCy model)
        speaker_detector.initialize()
        return speaker_detector
    except Exception as exc:
        logger.error(
            "Failed to initialize speaker detector: %s",
            format_exception_for_log(exc),
        )
        return None


def _detect_hosts_from_feed(
    feed: RssFeed,  # type: ignore[valid-type]
    speaker_detector: Any,
) -> set[str]:
    """Detect hosts from feed metadata using speaker detector.

    Args:
        feed: Parsed RssFeed object
        speaker_detector: Speaker detector instance

    Returns:
        Set of detected host names
    """
    # Statement-first order of authority (ADR-130 / Fable-5 audit F2). The deterministic parser
    # reads the feed's own "Hosted by ..." blurb out of the description, then org-filtered author
    # tags, then NER. EVERY LLM provider's detect_hosts short-circuits on RSS author tags — it
    # returns set(feed_authors) verbatim and never reads the description — so calling it first
    # mis-anchors org-authored feeds (the org is returned, then stripped to nothing) AND overrides a
    # feed whose description names hosts different from a personal author tag. Consult the provider
    # only when the deterministic parse finds nothing (it may still add an LLM-NER hit).
    stated = detect_hosts_from_feed(
        feed.title,
        feed.description,
        feed.authors or [],
        language=getattr(feed, "language", "") or "",
    )
    if stated:
        return stated
    try:
        feed_hosts = speaker_detector.detect_hosts(
            feed_title=feed.title,
            feed_description=feed.description,  # show blurb usually names the host (#1169)
            feed_authors=feed.authors if feed.authors else None,
        )
    except Exception as exc:
        if not caused_by_missing_import(exc):
            raise
        # A missing optional PACKAGE degrades; it does not end the run — the same rule the two
        # sites below already apply (utils.optional_deps). This one was missed, and it is reached
        # EARLIER than either: `_setup_pipeline_resources` calls it before any episode is touched,
        # so a box without the [ml] extra died at pipeline startup with a bare
        # ``ModuleNotFoundError: No module named 'spacy'`` out of ``run_pipeline`` — 12 e2e tests,
        # all of them describing error-recovery behaviour, none of them about spaCy.
        #
        # The deterministic parse above already ran and found nothing stated, so the honest result
        # is "no host inferred": whatever the episode metadata states is what it keeps. Host
        # detection is an enhancement, and the pipeline has a no-detector path
        # (``_create_speaker_detector_if_needed`` returning None takes it) — this is that path,
        # reached one step later.
        #
        # LOUDLY, per #1647: a feed whose hosts were never looked for must stay distinguishable
        # from one where the detector ran and honestly found nobody.
        logger.error(
            "Feed host detection UNAVAILABLE — an optional dependency is not installed (%s: %s). "
            "No host was inferred from the feed; the show keeps whatever its metadata states. "
            "Install the [ml] extra to enable it.",
            type(exc).__name__,
            exc,
        )
        return set()
    return _sanitize_detected_hosts(
        cast("set[str]", feed_hosts), feed.title, language=_feed_language(feed)
    )


def _sanitize_detected_hosts(
    names: set[str], feed_title: str | None = None, *, language: Optional[str] = None
) -> set[str]:
    """Put a provider's host names through the same filter the deterministic path uses.

    The deterministic branch above splits multi-person strings (``split_author_names``) and
    rejects organisations (``is_network_or_org_author``). The provider branch did neither, and
    an LLM asked "who hosts this show?" answers in prose: on *The a16z Show* it returned the
    single string ``"Erik Torenberg, Ben Horowitz, Travis Kalanick"``, which became one
    ``Person`` node — ``person:erik-torenberg-ben-horowitz-travis-kalanick``, a human being who
    does not exist, anchoring the roster for the whole episode.

    A composite is worse than no host at all. It can never match a diarized voice (the roster
    compares per name), so it silently disables the known-hosts anchor *and* pollutes the graph
    with a fake entity that cross-episode queries will happily join on.

    Same conservative contract as ``split_author_names``: an over-eager split degrades to "no
    host", which is the safe direction (#876), never to an invented person.

    The rule itself now lives in :func:`~podcast_scraper.speaker_detectors.hosts.
    normalize_host_names`, shared with the episode-authors and config paths — fixing it here
    only, as the first attempt did, left the path that actually fired on a16z untouched.

    *feed_title* carries the show's own name so the SHOW can be rejected as a host (#2064). This
    branch is reached precisely when the deterministic parse found nothing, which is exactly what
    happens once that parse correctly refuses to read a title as a sentence — so an LLM asked "who
    hosts this show?" answers with the show. Three fixes landed before this one (the title pattern,
    the author tag, the episode-authors fallback) and a fresh ingest still produced
    ``host='Africa Tech Summit'`` on all three episodes, because this call passed no title and the
    guard could not fire.
    """
    out = normalize_host_names(names or set(), feed_title=feed_title, language=language)
    if out != set(names or set()):
        logger.info(
            "host detection: provider names %s normalised to %s (split + org-filtered)",
            sorted(names or set()),
            sorted(out),
        )
    return out


def _validate_hosts_with_first_episode(
    feed_hosts: set[str],
    feed: RssFeed,  # type: ignore[valid-type]
    episodes: List[Episode],  # type: ignore[valid-type]
    speaker_detector: Any,
    pipeline_metrics: Optional[metrics.Metrics],
) -> set[str]:
    """Validate hosts by checking if they appear in first episode.

    Args:
        feed_hosts: Hosts detected from feed
        feed: Parsed RssFeed object
        episodes: List of Episode objects
        speaker_detector: Speaker detector instance
        pipeline_metrics: Optional metrics collector

    Returns:
        Validated set of host names
    """
    # Skip validation when the feed carries author tags: on such feeds the hosts came either from a
    # trusted author tag OR (post audit-F2, statement-first) from the feed's own "Hosted by ..."
    # statement — both authoritative, neither needing first-episode corroboration. Only NER-derived
    # hosts on a tag-less feed fall through to validation below.
    if not feed_hosts or not episodes or feed.authors:
        return feed_hosts

    # Only validate if we used NER (not author tags)
    first_episode = episodes[0]
    first_episode_description = extract_episode_description(first_episode.item)
    # Validate hosts by checking if they appear in first episode
    # Use provider's detect_speakers to extract persons from first episode
    # Pass pipeline_metrics for LLM call tracking (if OpenAI provider)
    import inspect

    sig = inspect.signature(speaker_detector.detect_speakers)
    if "pipeline_metrics" in sig.parameters:
        first_episode_speakers, _, _, _ = (
            speaker_detector.detect_speakers(  # type: ignore[call-arg]
                episode_title=first_episode.title,
                episode_description=first_episode_description,
                known_hosts=set(),
                pipeline_metrics=pipeline_metrics,
            )
        )
    else:
        first_episode_speakers, _, _, _ = speaker_detector.detect_speakers(
            episode_title=first_episode.title,
            episode_description=first_episode_description,
            known_hosts=set(),
        )
    first_episode_persons = set(first_episode_speakers)
    # Only keep hosts that also appear in first episode (validation)
    validated_hosts = feed_hosts & first_episode_persons
    if validated_hosts != feed_hosts:
        logger.debug(
            "Host validation: %d hosts from feed, %d validated with first episode",
            len(feed_hosts),
            len(validated_hosts),
        )
        if validated_hosts:
            logger.debug(
                "Validated hosts (appear in feed and first episode): %s",
                list(validated_hosts),
            )
        if feed_hosts - validated_hosts:
            logger.debug(
                "Hosts from feed not found in first episode (discarded): %s",
                list(feed_hosts - validated_hosts),
            )
    return validated_hosts if validated_hosts else feed_hosts


def hosts_for_episode(
    result: HostDetectionResult, episode: Any, episode_people: Iterable[str] = ()
) -> set[str]:
    """The hosts to anchor ONE episode on (#2197, then problem 3 of the naming scoreboard).

    The feed-level set (statement, author tags, config ``known_hosts``, recurrence) is kept; the
    episode-author fallback's contribution is replaced by THIS episode's own byline; and the hosts
    this episode's own description names join the pool — through one composer,
    :func:`~podcast_scraper.speaker_detectors.hosts.compose_episode_hosts`, which also decides that
    a byline listing everybody in the room adds nobody once the description says who hosts (The a16z
    Show: "Erik Torenberg sits down with Amjad Masad and Gagan Biyani" — the byline names all
    three).

    When a feed names nobody, the episode-level ``<itunes:author>`` fallback unions the authors of
    the feed's first episodes into the FEED's host set, and every episode inherits it. On Latent
    Space one post's authors ("Brandon Anderson, RJ Honicky, and Latent.Space") were seated as the
    hosts of 9 episodes; the live feed carries that tag on ONE of its 229 items — 170 say only
    "Latent.Space". Another episode's author is not this episode's host.
    """
    # An operator `hosts` override on THIS episode replaces everything below (#2283).
    from podcast_scraper.workflow.apply_overrides import override_hosts

    forced = override_hosts(episode)
    if forced is not None:
        try:
            episode.unbindable_hosts = []
        except AttributeError:
            pass
        return set(forced)
    hosts = set(result.cached_hosts or ())
    fallback = set(getattr(result, "episode_author_hosts", frozenset()) or ())
    from podcast_scraper.rss import parser as rss_parser

    item = getattr(episode, "item", None)
    own: set[str] = set()
    if fallback and item is not None:
        own_raw = rss_parser.extract_episode_authors(item)
        own = {
            a
            for a in normalize_host_names(
                own_raw, feed_title=result.feed_title, language=result.language
            )
            if not is_network_or_org_author(a, language=result.language)
        }
        own = set(
            drop_non_person_names(
                sorted(own), result.feed_title, result.kind_votes, language=result.language
            )
        )
    # <podcast:person role="host"> names THIS episode's host in so many words (Podcasting 2.0). It
    # is the only per-item field that types the role: measured 2026-10-24 on the labelled episodes'
    # live RSS, its roles were right 6 of 6. Joined after the feed's hosts, through the same person
    # check; a guest entry adds no host.
    stated_hosts: List[str] = []
    if item is not None:
        try:
            stated_hosts = [
                p["name"] or ""
                for p in rss_parser.extract_episode_people_stated(item)
                if p["source"] == "podcast_person" and p["role"] == "host"
            ]
        except (AttributeError, TypeError):
            stated_hosts = []

    def compose(with_statements: bool) -> List[str]:
        return compose_episode_hosts(
            sorted(hosts - fallback),
            sorted(own),
            episode_title=getattr(episode, "title", None),
            episode_description=extract_episode_description(item) if item is not None else None,
            feed_title=result.feed_title,
            more=stated_hosts if with_statements else (),
            episode_people=list(episode_people or ()),
            language=getattr(result, "language", None),
            read_statements=with_statements,
        )

    pool = compose(True)
    # v51: a host the pool has ONLY from this episode's own statement (v43) or podcast:person (v44)
    # anchors the seat, the guard and the spelling, but is not a name the resolver's LLM may bind:
    # in step 3 of the 2026-10-24 DGX block its one served effect through the LLM was a wrong name.
    base = compose(False)
    from podcast_scraper.providers.ml.diarization.roster import _same_person_on_one_episode

    unbindable = [n for n in pool if not any(_same_person_on_one_episode(n, b) for b in base)]
    try:
        episode.unbindable_hosts = unbindable
    except AttributeError:
        pass
    return set(pool)


def _fallback_to_episode_authors(
    cfg: config.Config,
    episodes: List[Episode],  # type: ignore[valid-type]
    feed_title: str | None = None,
    language: Optional[str] = None,
) -> set[str]:
    """Fallback to episode-level authors if no feed-level hosts found.

    Args:
        cfg: Configuration object
        episodes: List of Episode objects

    Returns:
        Set of episode author names (filtered to exclude organizations)
    """
    episode_authors: set[str] = set()
    if not cfg.auto_speakers or not episodes:
        return episode_authors

    from podcast_scraper.rss import parser as rss_parser

    # Check first 3 episodes for episode-level authors
    for episode in episodes[:3]:
        episode_author_list = rss_parser.extract_episode_authors(episode.item)
        # normalize_host_names both SPLITS multi-person tags and applies the shared network/org
        # predicate. Filtering for orgs alone (what this did before) let *The a16z Show*'s
        # ``<itunes:author>Erik Torenberg, Ben Horowitz, Travis Kalanick</itunes:author>``
        # through whole — is_network_or_org_author returns False for it, since a three-person
        # string is neither a mononym nor org-marked — and this is the path that fired on the
        # acceptance run (#1652).
        # #2064: the feed title is the only evidence that tells the SHOW apart from a person on
        # it, and the episode-authors fallback is one of the paths that seated "Africa Tech Summit"
        # as a host.
        episode_authors |= normalize_host_names(
            episode_author_list, feed_title=feed_title, language=language
        )

    return episode_authors


def _log_detected_hosts(
    cached_hosts: set[str],
    feed: RssFeed,  # type: ignore[valid-type]
    episode_authors: set[str],
    cfg: config.Config,
    source: Optional[str] = None,
) -> None:
    """Log detected hosts with their source.

    Args:
        cached_hosts: Set of detected host names
        feed: Parsed RssFeed object
        episode_authors: Set of episode-level authors
        cfg: Configuration object
        source: The branch that actually produced ``cached_hosts``. The caller knows this for
            certain; when omitted it is inferred, which is strictly a guess (see
            :func:`_infer_host_source`).
    """
    if not cached_hosts:
        if getattr(cfg, "auto_speakers", False):
            logger.debug(
                "No hosts detected from feed metadata, episode-level authors, or config known_hosts"
            )
        return

    if source is None:
        source = _infer_host_source(cached_hosts, feed, episode_authors, cfg)
    logger.info("DETECTED HOSTS (from %s): %s", source, ", ".join(sorted(cached_hosts)))


def _infer_host_source(
    cached_hosts: set[str],
    feed: RssFeed,  # type: ignore[valid-type]
    episode_authors: set[str],
    cfg: config.Config,
) -> str:
    """Best-effort guess at which branch produced ``cached_hosts``.

    Only for callers that cannot say. ``detect_feed_hosts_and_patterns`` passes ``source``
    explicitly, because inference here got it wrong in the way that costs debugging time:
    a non-empty ``feed.authors`` does NOT mean the hosts came from it — on an org-authored feed
    those tags are stripped as publisher metadata and the names arrive from the episode-authors
    fallback. Testing ``feed.authors`` first labelled exactly that case "RSS author tags", which
    aimed the a16z composite investigation at the wrong path for a full cycle (#1652).

    So: check the specific sources against what was actually produced, and only then fall back.
    """
    known = list(getattr(cfg, "known_hosts", None) or [])
    if episode_authors and cached_hosts == episode_authors:
        return "episode-level authors"
    # Both spellings: ``cached_hosts`` holds NORMALISED names, so a config entry that needed
    # splitting no longer equals its raw form — comparing only raw would mislabel it.
    if known and cached_hosts in (
        set(known),
        normalize_host_names(known, language=_feed_language(feed)),
    ):
        return "config known_hosts (fallback)"
    if feed.authors:
        return "RSS author tags"
    return "feed metadata (NER)"


def _record_hosts_detected(
    hosts: set[str], source: str, feed_level: set[str], dropped: set[str]
) -> None:
    """Show sidecar: the host set this run uses for the show, and the branch that produced it."""
    from podcast_scraper.workflow.show_events import record_show_event

    record_show_event(
        "hosts_detected",
        hosts=sorted(hosts),
        source=source,
        feed_level_hosts=sorted(feed_level),
        dropped_non_person=sorted(dropped),
    )


def _feed_title(feed: Any) -> Optional[str]:
    """The feed's title, or None. Carried on `HostDetectionResult` so per-episode host parsing can
    refuse a "host" that is really the show — the guard is inert without it, by design."""
    return getattr(feed, "title", None)


def _feed_language(feed: Any) -> Optional[str]:
    """The feed's declared language, or None. Carried on `HostDetectionResult` for the same reason
    as the title: the per-episode host pool is checked against that language's reject words."""
    return getattr(feed, "language", None) or None


# How much of each transcript the recurrence scan reads. Wider than the 2,000 the per-episode
# roster path uses: this scan pays the cost once per feed rather than once per episode, and a show
# that opens with a sponsor read or a cold-open clip puts its host introduction well past 2,000.
# The corpus measurement behind the thresholds was taken at this window.
_INTRO_SCAN_CHARS = 4000


# The detector's raw answer as kept in the naming decision trace (#2276); `raw_chars` keeps the
# uncut length. A detection answer is a short JSON object; the cap only bounds a runaway reply.
_DETECTION_RAW_MAX_CHARS = 8_000


def _newest_run_transcripts(root: Path) -> List[Path]:
    """One transcript per episode under *root*, from the run that SUPERSEDES the others.

    ``root`` is a single feed's workspace (``<corpus>/feeds/<slug>/`` for a corpus run,
    ``output/rss_*`` for a standalone one), so this never crosses feeds. It is deliberately
    ``glob("run_*/...")`` and not ``rglob``: pointed at a corpus ROOT by mistake it finds nothing
    rather than pooling every show's transcripts into one host vote.

    Two things would otherwise double-count. An episode carries up to three ``.txt`` bodies
    (``.adfree``, ``.cleaned``, plain) — ad-free first, because an unstripped pre-roll can push the
    host's self-introduction past the window we read. And a reprocessed episode exists in several
    ``run_*`` dirs at once; the corpus membership rule picks the newest, the same winner every
    catalog-facing surface uses.
    """
    metas = sorted(root.glob(f"run_*/{filesystem.METADATA_SUBDIR}/*.metadata.json"))
    if not metas:
        return []
    try:
        from podcast_scraper.search.corpus_scope import dedupe_metadata_paths_newest_run_per_episode

        metas = list(dedupe_metadata_paths_newest_run_per_episode(root, metas))
    except Exception as exc:  # noqa: BLE001 - a duplicate vote beats no vote at all
        logger.debug("recurrent-host scan could not dedupe runs: %s", exc)
    out: List[Path] = []
    for meta in metas:
        stem = meta.name[: -len(".metadata.json")]
        transcripts = meta.parent.parent / filesystem.TRANSCRIPTS_SUBDIR
        # First rendering with the pre-roll gone; this scan does not care about offsets. Inline
        # and local on purpose: it briefly routed through `workflow.transcript_resolution`, a
        # module written for translation, which made a generic scan depend on a language feature
        # for five lines of precedence it can state itself (reverted 2026-10-02).
        for suffix in (".adfree.txt", ".cleaned.txt", ".txt"):
            candidate = transcripts / f"{stem}{suffix}"
            if candidate.is_file():
                out.append(candidate)
                break
    return out


def _recurrent_hosts_from_disk(output_dir: Optional[str], feed: Any) -> Set[str]:
    """Hosts that self-introduce across MANY of this feed's already-transcribed episodes.

    A GUEST APPEARS ONCE; A HOST APPEARS EVERY WEEK. No per-episode rule can see that, and on the
    shows whose RSS author tag is an organisation it is the only host signal that exists at all.
    Measured across the production snapshot, at the thresholds
    :func:`~podcast_scraper.speaker_detectors.hosts.recurrent_hosts_across_episodes` enforces,
    a name emerges for 30 of 55 feeds and every one of them is that show's presenter.

    Reads transcripts already on disk, so it contributes on a re-process or an incremental ingest
    and contributes NOTHING on a feed's first-ever run — correct in both cases, and never a
    network call.

    The result joins ``known_hosts`` and goes no further. It is a CANDIDATE: it binds to a voice
    only through that episode's own self-introduction or the LLM resolver's existing guards, never
    by talk share and never by elimination, both of which measured far below the safety bar.
    """
    raw = str(output_dir or "")
    root = Path(raw)
    if not raw or not root.is_dir():
        return set()
    per_episode: List[List[str]] = []
    for path in _newest_run_transcripts(root):
        try:
            head = path.read_text(encoding="utf-8", errors="replace")[:_INTRO_SCAN_CHARS]
        except OSError:
            continue
        # EVERY intro in the head, not the first. A two-host show introduces its second host
        # second, and reading one name per episode left every co-host below the threshold.
        per_episode.append(
            distinct_self_introductions(
                head, intro_chars=_INTRO_SCAN_CHARS, feed_title=_feed_title(feed)
            )
        )
    if not per_episode:
        return set()
    return recurrent_hosts_across_episodes(
        per_episode, feed_title=_feed_title(feed), language=_feed_language(feed)
    )


def detect_feed_hosts_and_patterns(
    cfg: config.Config,
    feed: RssFeed,  # type: ignore[valid-type]
    episodes: List[Episode],  # type: ignore[valid-type]
    pipeline_metrics: Optional[metrics.Metrics] = None,
    speaker_detector: Optional[Any] = None,
) -> HostDetectionResult:
    """Detect hosts from feed metadata and analyze episode patterns.

    Args:
        cfg: Configuration object
        feed: Parsed RssFeed object
        episodes: List of Episode objects
        pipeline_metrics: Optional metrics collector
        speaker_detector: Optional pre-initialized speaker detector instance.
            If None and auto_speakers=True, will create one (for backward compatibility).

    Returns:
        HostDetectionResult with cached_hosts and heuristics
    """
    cached_hosts: set[str] = set()
    heuristics: Optional[Dict[str, Any]] = None
    from podcast_scraper.utils import correlation

    correlation.set_feed_title(_feed_title(feed))
    correlation.set_feed_description(getattr(feed, "description", None))

    # If auto_speakers is disabled, skip speaker detection entirely
    if not cfg.auto_speakers:
        return HostDetectionResult(cached_hosts, heuristics, None, _feed_title(feed))

    # In dry-run mode, still detect hosts from RSS author tags (no ML needed)
    if cfg.dry_run:
        return _handle_dry_run_host_detection(feed)

    # Use provided speaker detector, or create one if not provided (backward compatibility)
    speaker_detector = _create_speaker_detector_if_needed(cfg, speaker_detector)
    if speaker_detector is None:
        return HostDetectionResult(cached_hosts, heuristics, None, _feed_title(feed))

    # Detect hosts: prefer RSS author tags, fall back to NER
    feed_hosts = _detect_hosts_from_feed(feed, speaker_detector)
    # Strip network/publisher author tags the detector surfaces as hosts (e.g. "Colossus",
    # "Colossus | Investing & Business Podcasts"). For such shows the real host comes from the
    # transcript self-introduction at diarization time, not the feed metadata (#876). The
    # statement-first ordering inside _detect_hosts_from_feed means an org-authored feed whose
    # description names its hosts is already recovered before this strip runs (ADR-130 / audit F2).
    feed_hosts = {
        h for h in feed_hosts if not is_network_or_org_author(h, language=_feed_language(feed))
    }

    # Priority: Use known_hosts from config if provided (show-level override)
    if cfg.known_hosts:
        # Operator-supplied, but not exempt: a composite entry ("A, B and C") in a show config
        # is just as unmatchable against a diarized voice as one from a feed, and would mint the
        # same fake Person. Normalising here keeps every seeding path on one rule (#1652).
        known_hosts_set = normalize_host_names(cfg.known_hosts, language=_feed_language(feed))
        logger.info(
            "Using known_hosts from config: %s",
            ", ".join(sorted(known_hosts_set)),
        )
        # Merge with feed_hosts (known_hosts takes precedence)
        cached_hosts = known_hosts_set | feed_hosts
        if cached_hosts:
            logger.info(
                "DETECTED HOSTS (from config known_hosts + feed): %s",
                ", ".join(sorted(cached_hosts)),
            )
            # Skip validation since known_hosts are trusted
            _record_hosts_detected(cached_hosts, "config known_hosts + feed", feed_hosts, set())
            return HostDetectionResult(
                cached_hosts,
                heuristics,
                speaker_detector,
                _feed_title(feed),
                language=_feed_language(feed),
            )

    # Validate hosts with first episode: hosts should appear in first episode too
    cached_hosts = _validate_hosts_with_first_episode(
        feed_hosts, feed, episodes, speaker_detector, pipeline_metrics
    )

    # Track which branch actually produced the hosts, rather than re-deriving it later from
    # circumstantial evidence — that inference is what mislabelled a16z's episode-author hosts
    # as "RSS author tags" (#1652).
    host_source: Optional[str] = None

    # Fallback to episode-level authors if no feed-level hosts found (Issue #380)
    episode_authors: set[str] = set()
    if not cached_hosts:
        episode_authors = {
            a
            for a in _fallback_to_episode_authors(
                cfg, episodes, getattr(feed, "title", None), language=_feed_language(feed)
            )
            if not is_network_or_org_author(a, language=_feed_language(feed))
        }
        if episode_authors:
            cached_hosts = episode_authors
            host_source = "episode-level authors"
            logger.info(
                "DETECTED HOSTS (from episode-level authors): %s",
                ", ".join(sorted(cached_hosts)),
            )

    # Fallback to known_hosts from config if no hosts detected (show-level override)
    if not cached_hosts and cfg.known_hosts:
        cached_hosts = normalize_host_names(cfg.known_hosts, language=_feed_language(feed))
        host_source = "config known_hosts (fallback)"
        logger.info(
            "DETECTED HOSTS (from config known_hosts fallback): %s",
            ", ".join(sorted(cached_hosts)),
        )

    # RECURRENCE, from this feed's own transcripts. A guest appears once; a host appears every
    # week — the one property separating them that no per-episode rule can see, and the only host
    # signal at all on a show whose author tag is an organisation. Contributes nothing on a first
    # run (no transcripts yet), which is correct.
    #
    # UNION, never replace: a feed statement, an author tag or config `known_hosts` all outrank it.
    # This only ever ADDS a candidate that must still bind through an episode's own evidence.
    try:
        recurrent = _recurrent_hosts_from_disk(cfg.output_dir, feed)
    except Exception as exc:  # noqa: BLE001 - a host hint must never end a run
        logger.debug("recurrent-host scan skipped: %s", exc)
        recurrent = set()
    # FUZZY dedupe, not exact. The transcript spells the name the way the ASR heard it, and on the
    # production snapshot four of the 27 names this finds are variants of a host we already hold:
    # "Peter Atiyah" for Peter Attia, "Anita Arnon" for Anita Anand, "Adam Reichert" for Adam
    # Reichardt, "Chris Grayton" for Chris Gratien. An exact-lowercase check keeps all four, and
    # then the feed carries TWO host candidates for one human — the duplicate-person defect this
    # branch exists to remove, minted by the very fix meant to help.
    added = {n for n in recurrent if not any(same_person(n, h) for h in cached_hosts)}
    if added:
        cached_hosts = set(cached_hosts) | added
        if not host_source:
            host_source = "recurrent self-introduction across the feed"
        logger.info("RECURRENT HOSTS (self-introduced across episodes): %s", sorted(added))

    # ONE LAST FILTER OVER EVERY PATH THAT FED THIS SET. `cached_hosts` is assembled from the feed
    # statement, the author tag, config `known_hosts`, episode authors and the recurrence scan, and
    # each of those has its own idea of what an organisation looks like. Whatever got in, it leaves
    # here as PEOPLE: a name in `known_hosts` is a name the roster may paint on a voice, and 128 of
    # the 150 organisation names published as speakers on the snapshot arrived by exactly that
    # route — "Andreessen Horowitz" on 60 voices, "Conversations with Tyler" on 23.
    #
    # This is the only layer that knows the feed's own title, so it is the only one that can refuse
    # the SHOW; the publish gate catches publishers later but can never catch that.
    from podcast_scraper.speaker_detectors.entity_kind_votes import votes_for_cfg

    kind_votes = votes_for_cfg(cfg)
    _people = set(
        drop_non_person_names(
            sorted(cached_hosts), _feed_title(feed), kind_votes, language=_feed_language(feed)
        )
    )
    _dropped = set(cached_hosts) - _people
    if _people != cached_hosts:
        logger.info(
            "  → dropped non-person host candidate(s): %s",
            sorted(cached_hosts - _people),
        )
        cached_hosts = _people

    # Log detected hosts with their source
    _log_detected_hosts(cached_hosts, feed, episode_authors, cfg, source=host_source)
    _record_hosts_detected(
        cached_hosts,
        host_source
        or (
            _infer_host_source(cached_hosts, feed, episode_authors, cfg) if cached_hosts else "none"
        ),
        feed_hosts,
        _dropped,
    )

    # Analyze patterns from first few episodes to extract heuristics
    if cfg.auto_speakers and episodes:
        try:
            heuristics_dict = speaker_detector.analyze_patterns(
                episodes=episodes, known_hosts=cached_hosts
            )
        except Exception as exc:
            if not caused_by_missing_import(exc):
                raise
            # A missing optional PACKAGE degrades; it does not end the run. THIS is the site that
            # actually killed things: `analyze_patterns` -> `_initialize_spacy` -> `import spacy`
            # runs once per FEED, before any episode is processed, so a missing spaCy took down
            # the whole run here — several stages before per-episode speaker detection could even
            # be reached.
            #
            # Building the detector already tolerated this and logged "speaker detection will be
            # unavailable"; this call then contradicted it. Unlike ffmpeg (#26, deliberately FATAL
            # because preprocessing decides whether a transcript is correct at all), pattern
            # analysis only produces title/description HEURISTICS — the pipeline runs without them
            # and `auto_speakers=False` skips them entirely by design.
            #
            # Loud, not silent: heuristics stay None and the run continues with metadata-stated
            # names only.
            logger.error(
                "Speaker pattern analysis UNAVAILABLE — an optional dependency is not installed "
                "(%s: %s). Continuing without title/description heuristics; episodes keep the "
                "names their metadata states. Install the [ml] extra to enable it.",
                type(exc).__name__,
                exc,
            )
            heuristics_dict = None
        if heuristics_dict:
            heuristics = heuristics_dict
            if heuristics.get("title_position_preference"):
                logger.debug(
                    "Pattern analysis: guest names typically appear at %s of title",
                    heuristics["title_position_preference"],
                )

    # Return result with provider instance
    return HostDetectionResult(
        cached_hosts,
        heuristics,
        speaker_detector,
        _feed_title(feed),
        episode_author_hosts=frozenset(episode_authors & set(cached_hosts)),
        kind_votes=kind_votes,
        language=_feed_language(feed),
    )


def _get_speaker_detector(
    host_detection_result: HostDetectionResult, cfg: config.Config
) -> Optional[Any]:
    """Get speaker detector from result or create fallback."""
    detector = host_detection_result.speaker_detector
    if detector:
        return detector
    logger.warning("speaker_detector not found in host_detection_result, creating new instance")
    import sys

    workflow_pkg = sys.modules.get("podcast_scraper.workflow")
    if workflow_pkg and hasattr(workflow_pkg, "create_speaker_detector"):
        func = getattr(workflow_pkg, "create_speaker_detector")
        from unittest.mock import Mock

        detector = func(cfg) if isinstance(func, Mock) else create_speaker_detector(cfg)
    else:
        from podcast_scraper.speaker_detectors.factory import (
            create_speaker_detector as factory_create_speaker_detector,
        )

        detector = factory_create_speaker_detector(cfg)
    if detector:
        detector.initialize()
    return detector


class DetectedSpeakers(NamedTuple):
    """What the detector found, and what SURVIVED corroboration — they are not the same list.

    ``guests`` is what may be painted onto a voice. ``stated`` is every name the episode metadata
    put forward, *including* the ones corroboration rejected, and it names nobody on its own.

    Keeping the rejects is what stops us laundering our own failures. When corroboration threw a
    real guest away ("the episode text names them but never introduces them as speaking"), the name
    disappeared entirely — so the roster saw no name going spare and concluded that *nobody could
    have named* the voice, filing a 35%-of-the-episode guest as "Unidentified speaker". The name
    was right there in the show notes. We could not place it, and that is a defect, not innocence.
    """

    guests: List[str]
    stated: List[str]


def _record_naming_cost(pipeline_metrics: Any, cost_probe: Any, episode_idx: int) -> None:
    """Attribute the probe's captured naming cost to this episode.

    Only called once detection has actually run, so a recorded ``0.0`` means measured-and-free
    (the deterministic detector made no LLM call) — distinct from no entry at all, which means
    detection never ran and the cost is genuinely unknown. Best-effort: a metrics object without
    the recorder (older callers pass a plain object) must not fail the episode.
    """
    if pipeline_metrics is None or cost_probe is None:
        return
    recorder = getattr(pipeline_metrics, "record_speaker_detection_cost", None)
    if not callable(recorder):
        return
    try:
        recorder(float(cost_probe.speaker_detection_cost_usd), episode_idx)
    except Exception:  # pragma: no cover - metrics must never break processing
        logger.debug("[%s] could not record speaker detection cost", episode_idx)


def _detect_speakers_for_episode(
    episode: Episode,  # type: ignore[valid-type]
    cfg: config.Config,
    host_detection_result: HostDetectionResult,
    pipeline_metrics: metrics.Metrics,
    skip_speaker_detection: bool = False,
    skip_reason: Optional[str] = None,
    skip_detail: Optional[Dict[str, Any]] = None,
) -> Optional[DetectedSpeakers]:
    """Run speaker detection for one episode; return corroborated guests + every stated name.

    Every exit path records a stage outcome (#1647). Before that, three of these returns were
    silent and left no trace anywhere — no timing, no log, no error — so an episode whose
    speakers were never detected looked exactly like one that had none to detect. That is the
    ambiguity that let #1646 run unnoticed across 72 % of the corpus.
    """

    # The episode's detection record for the naming decision trace (#2276). Set first so every
    # exit below leaves one — a skipped detection is a fact the trace must show, not an absence.
    report: Dict[str, Any] = {}
    try:
        episode.speaker_detection_report = report
    except AttributeError:  # pragma: no cover - a stand-in episode without the field
        pass

    def _record(outcome: str, reason: Optional[str] = None, **kwargs: Any) -> None:
        report["outcome"] = outcome
        if reason:
            report["reason"] = reason
        if pipeline_metrics is not None:
            pipeline_metrics.record_stage_outcome(
                "speaker_detection", episode.idx, outcome, reason=reason, **kwargs
            )

    # An operator `guests` override on this episode is the answer, not an input (#2283): the
    # override REPLACES detection, so nothing detected can add a guest it left out.
    from podcast_scraper.workflow.apply_overrides import override_guests

    forced_guests = override_guests(episode)
    if forced_guests is not None:
        _record("skipped", "operator_override")
        return DetectedSpeakers(guests=forced_guests, stated=list(forced_guests))

    if not cfg.auto_speakers:
        if cfg.screenplay_speaker_names and len(cfg.screenplay_speaker_names) > 1:
            _record("skipped", "auto_speakers_disabled_using_configured_names")
            return DetectedSpeakers(guests=cfg.screenplay_speaker_names[1:], stated=[])
        _record("skipped", "auto_speakers_disabled")
        return None
    logger.debug("Episode %d: %s", episode.idx, episode.title)
    # One check, not two: this condition was tested again below with an identical body, so the
    # second test was unreachable. Collapsed while adding the ledger (#1647).
    if skip_speaker_detection:
        # WARNING, not silence: the caller decided to skip, and the consequence is that no
        # voice on this episode can be named. See #1646 for what that costs downstream.
        logger.warning(
            "[%s] Speaker detection SKIPPED (%s) — no voice on this episode can be named, "
            "so its insights will be marked unsurfaceable.",
            episode.idx,
            skip_reason or "reason not recorded",
        )
        _record("skipped", skip_reason or "skip_requested_by_caller", detail=skip_detail)
        return None
    if cfg.dry_run:
        episode_description = extract_episode_description(episode.item) or ""
        desc_preview = (
            episode_description[:50] + "..."
            if len(episode_description) > 50
            else episode_description
        )
        logger.info(
            "(dry-run) would detect speakers from: %s | %s",
            episode.title,
            desc_preview,
        )
        _record("skipped", "dry_run")
        return None
    episode_description = extract_episode_description(episode.item)
    extract_names_start = time.time()
    speaker_detector = _get_speaker_detector(host_detection_result, cfg)
    if not speaker_detector:
        logger.warning(
            "[%s] Speaker detection SKIPPED — no speaker detector could be constructed "
            "(provider=%s).",
            episode.idx,
            getattr(cfg, "speaker_detector_provider", None),
        )
        _record(
            "skipped",
            "no_speaker_detector_available",
            detail={"speaker_detector_provider": getattr(cfg, "speaker_detector_provider", None)},
        )
        return None
    # The feed's language: its reject vocabulary applies to these names as well as English's.
    _host_language = host_detection_result.language
    cached_hosts = host_detection_result.cached_hosts if cfg.cache_detected_hosts else set()
    # Per-episode seeding — a FIFTH path into known_hosts, found by the structural test rather
    # than by reading, and the reason that test exists. An un-normalised composite here reaches
    # the detector as the roster for every episode, which is where the fake Person is minted.
    combined_hosts = (
        normalize_host_names(cfg.known_hosts, language=_host_language) | cached_hosts
        if cfg.known_hosts
        else cached_hosts
    )
    import inspect

    # Isolate THIS episode's naming cost. Providers record speaker-detection cost onto a
    # run-level accumulator shared by parallel episodes, so a delta on it is racy and cannot be
    # attributed to one episode — which is why naming.cost_usd was absent from every manifest of
    # the acceptance run while the run total kept climbing. The probe forwards everything to the
    # real metrics object (run totals stay correct) and captures this episode's share on the side.
    from podcast_scraper.workflow.processing_manifest import EpisodeCostProbe

    cost_probe = EpisodeCostProbe(pipeline_metrics) if pipeline_metrics is not None else None
    detect_metrics = cost_probe if cost_probe is not None else pipeline_metrics

    sig = inspect.signature(speaker_detector.detect_speakers)
    report["detector"] = type(speaker_detector).__name__
    # Cleared first: a provider whose call fails before it parses leaves no raw answer, and the
    # previous episode's must not stand in for it.
    try:
        speaker_detector.last_speaker_detection_raw = None
        speaker_detector.last_speaker_detection_stated_only = None
    except AttributeError:  # pragma: no cover - a detector that refuses attributes
        pass
    # A raising detector previously recorded NOTHING — no ledger entry at all — so an episode
    # whose speaker detection blew up was indistinguishable from one where it never ran. That
    # silence is the #1646 shape, and it is also why ``failed`` ended up being misused for the
    # empty-result path below: there was no real failure path competing for the word.
    #
    # Control flow is deliberately unchanged: record, then re-raise.
    try:
        if "pipeline_metrics" in sig.parameters:
            detected_speakers, detected_hosts_set, detection_succeeded, _ = (
                speaker_detector.detect_speakers(
                    episode_title=episode.title,
                    episode_description=episode_description,
                    known_hosts=combined_hosts,
                    pipeline_metrics=detect_metrics,
                )
            )
        else:
            detected_speakers, detected_hosts_set, detection_succeeded, _ = (
                speaker_detector.detect_speakers(
                    episode_title=episode.title,
                    episode_description=episode_description,
                    known_hosts=combined_hosts,
                )
            )
    except Exception as exc:
        # Cost is recorded even here: a detector that raised after its LLM call still spent the
        # money, and a manifest that omits it under-reports the episode's true cost.
        _record_naming_cost(pipeline_metrics, cost_probe, episode.idx)
        if caused_by_missing_import(exc):
            # A missing optional PACKAGE degrades; it does not end the run.
            #
            # The pipeline already builds the detector defensively — a missing spaCy is caught
            # there and logged as "speaker detection will be unavailable" — and then this path
            # killed the run anyway, several stages later. The code announced a degrade it did
            # not perform. That was never a decision: unlike ffmpeg (#26, deliberately FATAL
            # because preprocessing decides whether a transcript is correct at all), speaker
            # detection is an enhancement over episode metadata and the pipeline already has a
            # no-detector path — ``auto_speakers=False`` returns immediately.
            #
            # LOUDLY, though. `degraded`, not `ran`, and its own reason slug: an episode whose
            # speakers were never detected must stay distinguishable from one where the detector
            # ran and honestly found nobody. That distinction is the whole point of #1647, and a
            # silent skip would destroy it as surely as a crash.
            #
            # ``caused_by_missing_import`` is the SAME walker preload uses (95be1ec1) — see
            # utils.optional_deps. Only a missing package qualifies; a missing model file, a
            # gated token, a timeout or a bug still raises below.
            logger.error(
                "[%s] Speaker detection UNAVAILABLE — an optional dependency is not installed "
                "(%s: %s). The episode keeps whatever names its metadata states; no speaker was "
                "inferred. Install the [ml] extra to enable it.",
                episode.idx,
                type(exc).__name__,
                exc,
            )
            _record(
                "degraded",
                "speaker_detector_package_missing",
                detail={
                    "exception": type(exc).__name__,
                    "speaker_detector_provider": getattr(cfg, "speaker_detector_provider", None),
                },
                duration_seconds=time.time() - extract_names_start,
            )
            return None
        _record(
            "failed",
            "detector_raised",
            detail={
                "exception": type(exc).__name__,
                "speaker_detector_provider": getattr(cfg, "speaker_detector_provider", None),
            },
            duration_seconds=time.time() - extract_names_start,
        )
        raise
    elapsed = time.time() - extract_names_start
    _raw = getattr(speaker_detector, "last_speaker_detection_raw", None)
    if isinstance(_raw, str):
        report["raw"] = _raw[:_DETECTION_RAW_MAX_CHARS]
        report["raw_chars"] = len(_raw)
    report["returned"] = {
        "speakers": [str(x) for x in (detected_speakers or [])],
        "hosts": sorted(str(x) for x in (detected_hosts_set or ())),
        "succeeded": bool(detection_succeeded),
    }
    _record_naming_cost(pipeline_metrics, cost_probe, episode.idx)
    if pipeline_metrics is not None:
        pipeline_metrics.record_extract_names_time(elapsed, episode.idx)
    if (
        not detection_succeeded
        and cfg.screenplay_speaker_names
        and len(cfg.screenplay_speaker_names) >= 2
    ):
        # The detector ran and came back empty; configured names are standing in for it.
        _record("degraded", "detection_failed_using_configured_names", duration_seconds=elapsed)
        return DetectedSpeakers(guests=cfg.screenplay_speaker_names[1:], stated=[])
    if not detection_succeeded:
        # RAN, not failed. ``detection_succeeded`` is ``bool(hosts or guests)``
        # (speaker_detectors/detection.py) — an EMPTINESS flag, not an error flag. Nothing
        # raised: the detector read the metadata and correctly found no names, which on a feed
        # that states no hosts is the designed outcome (#876 — NER on a description returns the
        # people an episode is ABOUT, so guessing is what put an advertiser's name on a show).
        #
        # Recording that as ``failed`` was a lie with two costs. A corpus report grouping by
        # outcome showed every host-less show as a permanent failure with nothing to fix; and
        # ``stage_did_run`` returns ``outcome in ("ran","degraded")``, so ``failed`` told the
        # roster the stage never ran — losing exactly the distinction between an UNMEASURED
        # voice and one measured as unnameable that #1647 exists to preserve.
        #
        # ``failed`` is reserved for a genuine exception, recorded by the caller's except path.
        _record("ran", "no_names_found_in_metadata", duration_seconds=elapsed)
    if detection_succeeded:
        flat_speakers: List[str] = []
        for entry in detected_speakers or []:
            flat_speakers.extend(_flatten_speaker_name_entries(entry))
        host_strings = _speaker_names_to_str_set(detected_hosts_set)

        # A PLACEHOLDER IS THE ABSENCE OF A NAME, AND IT WAS BEING PUBLISHED AS ONE.
        #
        # Every provider returns `DEFAULT_SPEAKER_NAMES` (`["Host", "unknown_guest_1"]`) as its
        # failure value, and from here they flowed on as if detection had succeeded: into
        # `detected_hosts`/`detected_guests`, into `metadata_named`, and so into the closed
        # candidate list the resolver may match a voice against. The resolver then did exactly what
        # it is built to do — matched a voice to a name the metadata stated. Measured on the
        # production snapshot: the literal string `Host` is published as a speaker on 59 episodes
        # and `unknown_guest_1` on 17, every one of them `source=llm_resolution`; on 29 episodes a
        # placeholder is the ONLY roster name, so the episode reads as attributed when nobody was
        # identified at all.
        #
        # Dropped here rather than at the publish gate because a placeholder must never become a
        # CANDIDATE: a name in the closed list is a name the roster is permitted to bind, and the
        # whole guard (#876) is that the list only contains people someone actually named.
        # Filtering leaves detection reporting honestly that it found nobody — which is what
        # happened.
        _before_speakers, _before_hosts = list(flat_speakers), set(host_strings)
        flat_speakers = filter_default_speaker_names(flat_speakers)
        host_strings = set(filter_default_speaker_names(sorted(host_strings)))
        report["dropped_placeholders"] = sorted(
            (set(_before_speakers) - set(flat_speakers)) | (_before_hosts - host_strings)
        )

        # ...and no organisations either. An LLM detector asked who is on the episode answers with
        # the network and the show as readily as with a person, and every name surviving here
        # reaches the closed candidate list the resolver may bind a voice to — which is how
        # "Machine Learning Street" came to be published as a speaker 26 times by that resolver.
        _feed_t = host_detection_result.feed_title
        _votes = host_detection_result.kind_votes
        _before_speakers, _before_hosts = list(flat_speakers), set(host_strings)
        flat_speakers = drop_non_person_names(
            flat_speakers, _feed_t, _votes, language=_host_language
        )
        host_strings = set(
            drop_non_person_names(sorted(host_strings), _feed_t, _votes, language=_host_language)
        )
        report["dropped_non_persons"] = sorted(
            (set(_before_speakers) - set(flat_speakers)) | (_before_hosts - host_strings)
        )

        # THE EPISODE'S OWN DESCRIPTION NAMES ITS HOST, on the shows where nothing else can.
        # "Elena Burger is joined by a16z's Andy McCall" — the guest cue reads what FOLLOWS the
        # verb; the name in front of it is the host, and that half was discarded. It matters most
        # where the feed cannot state a host at all: a16z rotates its host per episode and its
        # author tag is the firm, which the org filter correctly throws away, so this is the only
        # place a host name exists. Measured on the 136 production episodes that end with no named
        # speaker: 25 yield a host here, 22 of them a16z's.
        #
        # Union, never replace: a feed-stated or config host still outranks a parsed one, and this
        # only ever ADDS a candidate the roster may bind. The show's own name and any publisher are
        # refused inside the helper.
        _ep_hosts = hosts_from_episode_description(
            episode.title,
            episode_description,
            host_detection_result.feed_title,
            language=transcription_language(cfg) or "",
        )
        report["hosts_from_description"] = sorted(_ep_hosts or ())
        if _ep_hosts:
            host_strings = set(host_strings) | _ep_hosts
            logger.info("  → Host from the episode description: %s", sorted(_ep_hosts))

        proposed = [name for name in flat_speakers if name not in host_strings]

        # An LLM detector's name list is a PROPOSAL, not a result — it returns success=True whatever
        # it emits. Corroborate every proposed guest against the description before their name is
        # painted onto a diarized voice cluster.
        #
        # But keep the PROPOSAL as well. A rejected name is still a name the metadata stated, and
        # the roster needs to know it existed — otherwise a guest we could not place is filed as a
        # person nobody could have named.
        _refused: List[Dict[str, str]] = []
        corroborated = corroborate_guests(
            proposed,
            episode_title=episode.title,
            episode_description=episode_description,
            known_hosts=host_strings | combined_hosts,
            rejected_out=_refused,
            language=transcription_language(cfg),
        )
        report["hosts"] = sorted(host_strings)
        report["proposed_guests"] = list(proposed)
        report["corroborated_guests"] = list(corroborated)
        report["corroboration_rejected"] = _refused
        # People the detector named but labelled neither host nor guest: STATED (the resolver may
        # match a voice to them, and an unplaced one is counted) but never corroborated, so no
        # count-based placement paints them on a voice (#2276; see the OpenAI-compatible parser).
        _stated_only = [
            n
            for n in drop_non_person_names(
                filter_default_speaker_names(
                    list(
                        getattr(speaker_detector, "last_speaker_detection_stated_only", None) or []
                    )
                ),
                _feed_t,
                _votes,
                language=_host_language,
            )
            if n not in proposed and n not in host_strings
        ]
        report["stated_only"] = _stated_only
        # Counts, not names: the ledger is a health signal, and a name list would make every
        # episode's record unbounded. `proposed` vs `corroborated` is the useful delta — a
        # detector proposing names that never survive corroboration is a distinct failure
        # from one that proposes nothing.
        _record(
            "ran",
            duration_seconds=elapsed,
            detail={
                "proposed_count": len(proposed),
                "corroborated_count": len(corroborated),
                "known_host_count": len(host_strings | combined_hosts),
            },
        )
        # NO SEAT CAP HERE, AND THAT IS DELIBERATE — it was tried and it was wrong (#2095/#2078).
        #
        # The worry is real: `guests` becomes `detected_guests`, then the roster's `guest_names`,
        # then `spare`, and finally the FORCED "one spare name, one spare voice" binding, which
        # publishes a name on nothing but a count. Uncapping detection therefore lets a guest
        # reach that path on exactly the two-host shows this fix targets, and a previous attempt
        # at uncapping was reverted for producing wrong forced names (7d083c4b).
        #
        # Re-imposing `screenplay_num_speakers - len(hosts)` here fails
        # `test_the_pipeline_still_returns_the_real_guest`, and that test is right: on a two-host
        # show the bound is ZERO, so `Dr. Adam Rodman` — corroborated, genuinely in the room —
        # is deleted from `guests` altogether. And `guests` is not only the arithmetic pool; it
        # also feeds `corroborated_persons`, which gates the EVIDENCE-based report-verb binding in
        # the roster. Capping it starves the very path that is supposed to name her.
        #
        # What actually bounds the risk is `corroborate_guests` above: a name reaches `guests`
        # only with textual evidence that the person SPOKE, which is the gate the reverted attempt
        # bypassed by drawing on `metadata_named` (topic-people) instead.
        #
        # NOT VERIFIED: the forced-path effect of the wider `guests` list is not measured. The
        # offline replay feeds stored `detected_guests`, which are pre-fix and capped, so no tier
        # available here can observe it. It needs a real run.
        return DetectedSpeakers(guests=corroborated, stated=[*proposed, *_stated_only])
    return None
