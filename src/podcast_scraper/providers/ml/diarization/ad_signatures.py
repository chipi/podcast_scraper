"""Cross-show ad signatures — what the CORPUS knows about ads that one episode, or one feed, cannot.

#1188 indexes the script a feed repeats across its own episodes. An advertisement is wider than
that: the same ad is inserted into many different shows. Measured on prod 2026-10-02 over 4,101
short voices: a German IONOS ad in 15+ episodes of 6 shows, the NYT house ads across The Daily and
Hard Fork, Goalhanger and Vox cross-promos. Text a voice shares with OTHER shows is an ad.

Two signals, both learned from the corpus, neither naming a language or a place:

* **Recurrence across shows.** A 6-word sequence seen in at least 3 episodes of at least 2 feeds
  is ad script. "At least 3 episodes" is what keeps a CROSS-POSTED episode (one show republished in
  another feed) from making every one of its voices "recur": a twin is only 2 episodes.
* **Ad languages.** Publishers geo-target ads by the IP that fetches the audio, so ads arrive in the
  language of wherever production runs. Among voices whose language differs from their episode's,
  a language whose voices mostly RECUR is an ad language; a one-off voice in it is an ad too
  (37 German ads on 2026-10-02 aired only once). A language whose foreign voices do not recur is
  content: the Spanish and Portuguese clips on Latin America in Focus recur 0 times and stay.

Only SHORT voices (15-600 words) are judged or indexed: a host is far longer, so a co-host reading
a cross-show house ad is never swept up, and the index stays small.

Built over the corpus by the batch finalize and written to ``search/ad_signatures.json``; the
diarization pipeline loads it and hands it to ``classify_voices``. Absent file = no opinion.
"""

from __future__ import annotations

import json
import logging
import re
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
FILENAME = "ad_signatures.json"
SHINGLE_WORDS = 6
VOICE_MIN_WORDS = 15
VOICE_MAX_WORDS = 600
RECUR_MIN_EPISODES = 3
RECUR_MIN_FEEDS = 2
#: A voice whose 6-word sequences are at least this much ad script is an ad.
VOICE_RECUR_FRACTION = 0.3
AD_LANGUAGE_MIN_VOICES = 10
AD_LANGUAGE_MIN_RECUR_SHARE = 0.5
#: Rebuild at most this often; the corpus grows by a few episodes per run.
MAX_AGE_S = 6 * 3600

_WORD = re.compile(r"[^\W\d_]+")

# Function words per language, kept only where UNIQUE to that language: shared short words ("a",
# "de", "no", "as") read English and German as Portuguese and taught the corpus a false ad language.
_FUNCTION_WORDS_RAW: Dict[str, List[str]] = {
    "en": "the and is that you this with was are have they what but not".split(),
    "de": (
        "der die das und ist nicht für mit sie auf dem ein eine wir sich auch noch ich du".split()
    ),
    "es": "el los las por para con del una pero muy está porque nosotros también y".split(),
    "pt": "os não uma você também do da na isso muito ele com".split(),
    "fr": "le les des et une pour avec pas qui est du au je nous vous ce".split(),
    "it": "il che di della gli sono anche per è non".split(),
    "nl": "het een van dat op te voor niet zijn ook ik je".split(),
}
_ALL_FUNCTION_WORDS = Counter(w for ws in _FUNCTION_WORDS_RAW.values() for w in ws)
FUNCTION_WORDS: Dict[str, FrozenSet[str]] = {
    lang: frozenset(w for w in ws if _ALL_FUNCTION_WORDS[w] == 1)
    for lang, ws in _FUNCTION_WORDS_RAW.items()
}


def words(text: str) -> List[str]:
    """Lowercased word tokens of ``text``."""
    return _WORD.findall((text or "").lower())


def shingles(ws: List[str]) -> Set[int]:
    """Stable 32-bit ids of every ``SHINGLE_WORDS``-word run (stable across processes)."""
    return {
        zlib.crc32(" ".join(ws[i : i + SHINGLE_WORDS]).encode("utf-8"))
        for i in range(len(ws) - SHINGLE_WORDS + 1)
    }


def detect_language(ws: List[str]) -> Optional[str]:
    """Best-guess language from function words, or ``None`` when the text does not say clearly."""
    if len(ws) < VOICE_MIN_WORDS:
        return None
    scored = sorted(
        ((sum(w in fw for w in ws) / len(ws), lang) for lang, fw in FUNCTION_WORDS.items()),
        reverse=True,
    )
    (best, lang), (second, _) = scored[0], scored[1]
    return lang if best >= 0.06 and best >= 2 * second else None


def _judged(ws: List[str]) -> bool:
    return VOICE_MIN_WORDS <= len(ws) <= VOICE_MAX_WORDS


@dataclass(frozen=True)
class AdSignatures:
    """Cross-show ad evidence: text runs many shows repeat, and languages ads arrive in."""

    recurring: FrozenSet[int]
    ad_languages: FrozenSet[str]

    def recurring_fraction(self, ws: List[str]) -> float:
        """Share of the voice's shingles that other shows carry too."""
        sh = shingles(ws)
        return sum(h in self.recurring for h in sh) / len(sh) if sh else 0.0

    def is_ad_voice(self, text: str, episode_language: Optional[str]) -> bool:
        """An ad: mostly cross-show recurring text, or a known ad language unlike the episode's."""
        ws = words(text)
        if not _judged(ws):
            return False
        if self.recurring_fraction(ws) >= VOICE_RECUR_FRACTION:
            return True
        lang = detect_language(ws)
        return bool(
            lang and episode_language and lang != episode_language and lang in self.ad_languages
        )


def episode_language(voice_texts: Dict[str, str]) -> Optional[str]:
    """The episode's language, judged over all its voices together."""
    return detect_language([w for t in voice_texts.values() for w in words(t)])


def build(episodes: Iterable[tuple]) -> Dict[str, Any]:
    """Signatures from ``(feed_id, episode_id, {voice: text})`` triples."""
    sh_episodes: Dict[int, Set[str]] = defaultdict(set)
    sh_feeds: Dict[int, Set[str]] = defaultdict(set)
    judged: List[tuple] = []  # (shingles, voice language, episode language)
    n_episodes = 0
    for feed_id, episode_id, voice_texts in episodes:
        n_episodes += 1
        ep_lang = episode_language(voice_texts)
        for text in voice_texts.values():
            ws = words(text)
            if not _judged(ws):
                continue
            sh = shingles(ws)
            for h in sh:
                sh_episodes[h].add(episode_id)
                sh_feeds[h].add(feed_id)
            judged.append((sh, detect_language(ws), ep_lang))
    recurring = {
        h
        for h, eps in sh_episodes.items()
        if len(eps) >= RECUR_MIN_EPISODES and len(sh_feeds[h]) >= RECUR_MIN_FEEDS
    }
    foreign: Counter = Counter()
    foreign_recurring: Counter = Counter()
    for sh, lang, ep_lang in judged:
        if lang and ep_lang and lang != ep_lang:
            foreign[lang] += 1
            if sh and sum(h in recurring for h in sh) / len(sh) >= VOICE_RECUR_FRACTION:
                foreign_recurring[lang] += 1
    ad_languages = sorted(
        lang
        for lang, n in foreign.items()
        if n >= AD_LANGUAGE_MIN_VOICES
        and foreign_recurring[lang] / n >= AD_LANGUAGE_MIN_RECUR_SHARE
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "shingle_words": SHINGLE_WORDS,
        "episodes": n_episodes,
        "voices_judged": len(judged),
        "foreign_voices_by_language": {
            k: [foreign[k], foreign_recurring[k]] for k in sorted(foreign)
        },
        "ad_languages": ad_languages,
        "recurring": sorted(recurring),
    }


def _corpus_episodes(corpus_root: Path) -> Iterable[tuple]:
    from ....search.corpus_scope import (
        dedupe_metadata_paths_newest_run_per_episode,
        discover_metadata_files,
    )

    metas = dedupe_metadata_paths_newest_run_per_episode(
        corpus_root, discover_metadata_files(corpus_root)
    )
    for meta in metas:
        meta = Path(meta)
        try:
            doc = json.loads(meta.read_text(encoding="utf-8"))
            rel = str((doc.get("content") or {}).get("transcript_file_path") or "")
            if not rel:
                continue
            seg_path = meta.parent.parent / rel.replace(".txt", ".segments.json")
            segs = json.loads(seg_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        segs = segs if isinstance(segs, list) else (segs or {}).get("segments") or []
        texts: Dict[str, List[str]] = defaultdict(list)
        for s in segs:
            if isinstance(s, dict) and s.get("speaker"):
                texts[str(s["speaker"])].append(str(s.get("text") or ""))
        if texts:
            yield (
                meta.parent.parent.parent.name,
                str(meta),
                {v: " ".join(t) for v, t in texts.items()},
            )


def write_for_corpus(corpus_root: Path, *, force: bool = False) -> Optional[Path]:
    """Rebuild ``search/ad_signatures.json`` for a corpus when missing or stale. Never raises."""
    dest = Path(corpus_root) / "search" / FILENAME
    try:
        if not force and dest.is_file() and time.time() - dest.stat().st_mtime < MAX_AGE_S:
            return dest
        t0 = time.perf_counter()
        doc = build(_corpus_episodes(Path(corpus_root)))
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc), encoding="utf-8")
        tmp.replace(dest)
        logger.info(
            "ad signatures: %d recurring passages, ad languages %s, from %d voices in %d episodes "
            "(%.1fs)",
            len(doc["recurring"]),
            doc["ad_languages"] or "none",
            doc["voices_judged"],
            doc["episodes"],
            time.perf_counter() - t0,
        )
        return dest
    except Exception as exc:  # noqa: BLE001 — a missing index must never fail a finished run
        logger.warning("ad signatures: could not build (%s); classification abstains", exc)
        return None


_cache: Dict[str, tuple] = {}


def load_near(output_dir: str) -> Optional[AdSignatures]:
    """The corpus's signatures for anything written under ``output_dir``, or ``None``.

    A feed writes under ``<corpus>/feeds/<feed>/run_*``; the file lives at ``<corpus>/search``, so
    this walks up a few levels. Cached by path and mtime.
    """
    if not output_dir:
        return None
    here = Path(output_dir).resolve()
    for d in [here, *list(here.parents)[:4]]:
        path = d / "search" / FILENAME
        if not path.is_file():
            continue
        try:
            mtime = path.stat().st_mtime
            hit = _cache.get(str(path))
            if hit and hit[0] == mtime:
                return hit[1]  # type: ignore[no-any-return]
            doc = json.loads(path.read_text(encoding="utf-8"))
            if doc.get("schema_version") != SCHEMA_VERSION:
                return None
            sig = AdSignatures(
                recurring=frozenset(int(h) for h in doc.get("recurring") or []),
                ad_languages=frozenset(doc.get("ad_languages") or []),
            )
            _cache[str(path)] = (mtime, sig)
            return sig
        except (OSError, ValueError) as exc:
            logger.warning("ad signatures at %s unreadable (%s); abstaining", path, exc)
            return None
    return None
