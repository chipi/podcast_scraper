"""Per-user mutable state as plain files (#1065, RFC-098 §3): playback, queue, library.

Builds on the per-user directory from #1063 (``<data_dir>/users/<id>/``). Each kind is one
JSON file; reads return a default when absent, writes are atomic. No DB — the personal
overlay only; shared corpus artifacts are never touched here.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from filelock import FileLock

from podcast_scraper.server.atomic_write import atomic_write_text

logger = logging.getLogger(__name__)

# Read-modify-write mutations on one user's file must not interleave (a second writer reading the
# pre-write state would lose the first's append). Each mutator holds a per-(user, file) lock over
# its read+write; the timeout makes a stuck lock fail loudly rather than deadlock.
_LOCK_TIMEOUT_S = 15.0


def _state_path(data_dir: Path, user_id: str, name: str) -> Path:
    return data_dir / "users" / user_id / f"{name}.json"


def _user_lock(data_dir: Path, user_id: str, name: str) -> FileLock:
    """A per-(user, file) write lock; serialises concurrent read-modify-write on that file."""
    path = _state_path(data_dir, user_id, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(path.with_name(f".{name}.lock")), timeout=_LOCK_TIMEOUT_S)


class UserStateUnreadable(RuntimeError):
    """A state file exists but could not be read or parsed.

    Raised only on the read-modify-WRITE path. Readers stay lenient: rendering an empty library
    because a file is briefly unreadable is recoverable and destroys nothing. Overwriting it is
    not — every mutator here persists what it just read, so answering a read error with the empty
    default meant one bad read plus one new capture replaced the user's entire history with a
    single row. "Absent" and "unreadable" must not be the same answer to a writer.
    """


def _read(data_dir: Path, user_id: str, name: str, default: Any, *, strict: bool = False) -> Any:
    """Parsed state, or ``default``. With ``strict``, an unreadable EXISTING file raises."""
    path = _state_path(data_dir, user_id, name)
    if not path.is_file():
        return default  # genuinely absent — the empty default is the truth
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        if strict:
            raise UserStateUnreadable(f"{name}.json is unreadable for user {user_id}") from exc
        return default


def _rows_for_update(data_dir: Path, user_id: str, name: str) -> list[dict[str, Any]]:
    """The RAW rows of a list-shaped state file, for read-modify-write.

    Two differences from the public getters, both deliberate:

    * unreadable raises (see :class:`UserStateUnreadable`) instead of silently yielding ``[]``;
    * rows the getters filter out — missing a field a response model requires, or written by a
      different schema version — are KEPT. The getters drop them so the API can still render; a
      mutator that wrote that filtered list back would delete them permanently, as a side effect of
      changing an unrelated row. Dropping on read is a display decision; persisting the drop is
      data loss.

    The invariant every caller must preserve: mutating row X must not rewrite any other row.

    A payload that parses but is the WRONG SHAPE (a dict where the rows should be) raises too. It
    is either hand-corruption or a schema version this build does not understand, and in both cases
    "reset it to empty and write" destroys exactly as much as answering a parse error did. Absent,
    unreadable and unrecognised must not be the same answer to a writer — only the first is safe.
    """
    data = _read(data_dir, user_id, name, [], strict=True)
    if not isinstance(data, list):
        raise UserStateUnreadable(f"{name}.json is not a list for user {user_id}")
    return [x for x in data if isinstance(x, dict)]


def _strings_for_update(data_dir: Path, user_id: str, name: str) -> list[str]:
    """The RAW entries of a list-of-strings state file (``interests``), for read-modify-write.

    :func:`_rows_for_update`'s sibling for the one state file whose rows are bare strings rather
    than objects, with the same two guarantees: unreadable or wrong-shaped raises instead of
    yielding ``[]``, and nothing is dropped on the way to the writer.
    """
    data = _read(data_dir, user_id, name, [], strict=True)
    if not isinstance(data, list):
        raise UserStateUnreadable(f"{name}.json is not a list for user {user_id}")
    return [str(x) for x in data]


def _mapping_for_update(data_dir: Path, user_id: str, name: str) -> dict[str, Any]:
    """The RAW mapping of a dict-shaped state file, for read-modify-write.

    The dict-shaped sibling of :func:`_rows_for_update`, and it exists for the same reason: an
    unreadable ``resurfacing.json`` used to read as ``{}``, so marking ONE highlight surfaced
    persisted a single-key mapping and erased every other highlight's ``{count, last_surfaced}``.
    Losing that is not cosmetic — every count resets to 0 and ``last_seen`` falls back to
    ``created_at``, so the user's whole history floods back in as due at once and the schedule
    they built is gone.

    Wrong-shaped-but-parseable raises for the same reason it does in :func:`_rows_for_update`.
    """
    data = _read(data_dir, user_id, name, {}, strict=True)
    if not isinstance(data, dict):
        raise UserStateUnreadable(f"{name}.json is not a mapping for user {user_id}")
    return data


def _write(data_dir: Path, user_id: str, name: str, obj: Any) -> None:
    path = _state_path(data_dir, user_id, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2))


# A client timestamp is advisory. A device with a wrong clock — or a hostile one — must not be
# able to write events into the distant past (poisoning windowed stats) or the future (parking a
# record where nothing later can beat it). #1924.
CLIENT_TS_MAX_AGE_SECONDS = 30 * 24 * 3600
CLIENT_TS_MAX_SKEW_SECONDS = 5 * 60


def clamp_client_ts(client_ts: int | None, now: int) -> int:
    """The timestamp to store: the client's when plausible, otherwise ``now``.

    Both ends CLAMP rather than fall back, and that asymmetry mattered: a too-old stamp used to be
    pulled to the floor while a too-future one collapsed all the way to ``now``. For the listen log
    that is advisory either way, but ``playback.updated_at`` is what the client's cross-device
    conflict resolution compares, so the bound has to be predictable in both directions.
    """
    if client_ts is None:
        return now
    if client_ts > now + CLIENT_TS_MAX_SKEW_SECONDS:
        return now + CLIENT_TS_MAX_SKEW_SECONDS
    if client_ts < now - CLIENT_TS_MAX_AGE_SECONDS:
        return now - CLIENT_TS_MAX_AGE_SECONDS
    return client_ts


# --- playback positions (slug -> {position_seconds, updated_at}) ---


def get_playback(data_dir: Path, user_id: str, slug: str) -> dict[str, Any] | None:
    """Return the saved playback record for an episode, or ``None`` when unset."""
    data = _read(data_dir, user_id, "playback", {})
    rec = data.get(slug) if isinstance(data, dict) else None
    return rec if isinstance(rec, dict) else None


def set_playback(
    data_dir: Path,
    user_id: str,
    slug: str,
    position_seconds: float,
    updated_at: int,
    finished: bool = False,
    tz_offset_minutes: int | None = None,
) -> dict[str, Any]:
    """Save the playback position for an episode; return the stored record.

    Reads strictly (:func:`_mapping_for_update`): this is the highest-frequency writer in the whole
    subsystem — clients save position continuously while audio plays — so it has the shortest window
    between "playback.json goes bad" and "playback.json gets overwritten". Resetting to ``{}`` here
    would drop every other episode's resume point on the next position save.
    """
    with _user_lock(data_dir, user_id, "playback"):
        data = _mapping_for_update(data_dir, user_id, "playback")
        prior = data.get(slug)
        previous_seconds = (
            float(prior.get("position_seconds", 0.0)) if isinstance(prior, dict) else None
        )
        rec = {
            "position_seconds": position_seconds,
            "updated_at": updated_at,
            "finished": bool(finished),
        }
        data[slug] = rec
        _write(data_dir, user_id, "playback", data)
        # Accrue listening time in the SAME lock (#1914 Phase 0). Inside, because this is the
        # highest-frequency writer in the subsystem and a second lock would double the contention
        # on the hot path; and after the position write, so a failure here cannot cost a resume
        # point. `_record_listening_unlocked` swallows its own errors for the same reason.
        _record_listening_unlocked(
            data_dir,
            user_id,
            slug,
            previous_seconds,
            position_seconds,
            updated_at,
            bool(finished),
            tz_offset_minutes,
        )
        return rec


# --- listening time (#1914 Phase 0) ---
#
# "Hours listened" does not exist yet. ``app_stats`` computes it as ``sum(position_seconds)`` — a
# lifetime snapshot of FURTHEST POSITION REACHED, which cannot be windowed, does not grow on a
# re-listen, and inflates when you seek forward. A recap led by that number would be fabricated.
#
# So record the real thing from now on: time actually accrued, bucketed by day. Recording is
# cheap and starts the clock; nothing reads it yet. A day recorded is a day we can recap later,
# and every day we do not record is gone for good — which is why this ships ahead of the feature.

MAX_LISTEN_DELTA_SECONDS = 30
"""Ceiling on one save's contribution.

The client saves position every 10s (``stores/player.ts`` ``SAVE_INTERVAL_MS``), so a legitimate
delta is about that. The ceiling is deliberately looser than the cadence — a busy device, a
backgrounded tab or a slow network genuinely delay a save — but far tighter than a seek, which is
what it exists to reject: skipping forward 20 minutes must not book 20 minutes of listening.
"""


#: Widest real UTC offset (UTC-12 .. UTC+14), in minutes. Anything outside is a broken or hostile
#: client and is ignored rather than trusted — see ``clamp_tz_offset``.
MAX_TZ_OFFSET_MINUTES = 14 * 60
MIN_TZ_OFFSET_MINUTES = -12 * 60


def clamp_tz_offset(offset_minutes: int | None) -> int:
    """The offset to bucket by: the client's when plausible, otherwise UTC."""
    if offset_minutes is None:
        return 0
    try:
        value = int(offset_minutes)
    except (TypeError, ValueError):
        return 0
    if value < MIN_TZ_OFFSET_MINUTES or value > MAX_TZ_OFFSET_MINUTES:
        return 0
    return value


def _day_key(ts: int, tz_offset_minutes: int = 0) -> str:
    """The listener's LOCAL calendar day for a timestamp.

    Local, not UTC, because a recap is about the listener's days: "your Tuesday", "your year". In
    UTC a year boundary puts New Year's Eve in the wrong year for most of the planet, and an
    evening listen west of Greenwich books to tomorrow.

    The offset travels WITH each save rather than being stored once per account, which is also the
    correct answer for DST and for travel: a save is bucketed by the offset in effect at the moment
    it happened, which is exactly what "the day you listened" means.

    Absent (older clients, and anything recorded before this shipped) means UTC.
    """
    local = int(ts) + clamp_tz_offset(tz_offset_minutes) * 60
    return datetime.fromtimestamp(local, timezone.utc).date().isoformat()


def accrue_listening(
    state: dict[str, Any],
    slug: str,
    previous_seconds: float | None,
    position_seconds: float,
    at_ts: int,
    finished: bool = False,
    tz_offset_minutes: int | None = None,
) -> dict[str, Any]:
    """Fold one position save into a listening record. PURE — no files, no clock.

    The delta is clamped to ``[0, MAX_LISTEN_DELTA_SECONDS]``:

    * **Never negative.** Rewinding is listening too, and subtracting it would let someone scrub
      backwards into negative time. It simply accrues nothing.
    * **Never more than the ceiling.** A forward seek moves the position without anyone hearing
      it. This is the whole reason ``sum(position_seconds)`` is unusable, and clamping is what
      makes the new number honest rather than merely different.

    A first save for an episode (``previous_seconds is None``) accrues nothing: we know where the
    listener is, not how they got there. Resuming at 12:00 is not twelve minutes of listening.
    """
    days = state.setdefault("days", {})
    delta = 0.0
    if previous_seconds is not None:
        moved = float(position_seconds) - float(previous_seconds)
        delta = max(0.0, min(moved, float(MAX_LISTEN_DELTA_SECONDS)))
    if delta > 0:
        key = _day_key(at_ts, clamp_tz_offset(tz_offset_minutes))
        days[key] = round(float(days.get(key, 0.0)) + delta, 3)
    # The anchor a recap needs when there is no account creation date to lean on ("since you
    # started listening" rather than "since you joined").
    first = state.get("first_listened_at")
    if first is None or int(at_ts) < int(first):
        state["first_listened_at"] = int(at_ts)
    if finished:
        # WHEN something was finished, which `finished: bool` cannot answer — so a recap can say
        # "you finished 14 episodes in March" rather than "you have finished 200 episodes ever".
        state.setdefault("finished_at", {}).setdefault(str(slug), int(at_ts))
    return state


def get_listening(data_dir: Path, user_id: str) -> dict[str, Any]:
    """The user's listening record; a well-shaped empty one when absent or corrupt."""
    data = _read(data_dir, user_id, "listening_daily", {})
    if not isinstance(data, dict):
        return {"days": {}, "first_listened_at": None, "finished_at": {}}
    days = data.get("days")
    finished_at = data.get("finished_at")
    return {
        "days": days if isinstance(days, dict) else {},
        "first_listened_at": data.get("first_listened_at"),
        "finished_at": finished_at if isinstance(finished_at, dict) else {},
    }


def _record_listening_unlocked(
    data_dir: Path,
    user_id: str,
    slug: str,
    previous_seconds: float | None,
    position_seconds: float,
    at_ts: int,
    finished: bool,
    tz_offset_minutes: int | None = None,
) -> None:
    """Accrue and persist. Call INSIDE the playback lock — see ``set_playback``.

    Never raises: this rides along with the position save, and losing a recap statistic must not
    cost the listener their resume point.
    """
    try:
        before = get_listening(data_dir, user_id)
        # DEEP copy. `dict(before)` shares the nested `days` mapping, so accruing into the copy
        # also mutates the original and the "did anything change?" test below is always false —
        # every delta was silently dropped while `first_listened_at` (a scalar) still updated, so
        # the file looked alive and recorded nothing.
        after = accrue_listening(
            deepcopy(before),
            slug,
            previous_seconds,
            position_seconds,
            at_ts,
            finished,
            tz_offset_minutes,
        )
        # Skip the write when nothing moved: this rides the highest-frequency writer in the
        # subsystem, and a paused player still saves position.
        if after != before:
            _write(data_dir, user_id, "listening_daily", after)
    except Exception:  # noqa: BLE001 — a statistic must never break playback persistence.
        logger.debug("listening accrual failed for %s/%s", user_id, slug, exc_info=True)


def list_playback(data_dir: Path, user_id: str) -> list[dict[str, Any]]:
    """All saved playback positions, newest-updated first (for the Home 'Continue' rail)."""
    data = _read(data_dir, user_id, "playback", {})
    if not isinstance(data, dict):
        return []
    out: list[dict[str, Any]] = []
    for slug, rec in data.items():
        if isinstance(rec, dict):
            out.append(
                {
                    "slug": str(slug),
                    "position_seconds": float(rec.get("position_seconds", 0.0)),
                    "updated_at": rec.get("updated_at"),
                    # Absent on records written before the flag existed — an old record is
                    # unfinished, which is what it always behaved as.
                    "finished": bool(rec.get("finished", False)),
                }
            )
    out.sort(key=lambda r: (r.get("updated_at") or 0), reverse=True)
    return out


# --- listen events (append-only log of episode opens, for analytics ) ---
#
# One line of JSON per "open", in <data_dir>/users/<id>/listen_events.jsonl. Append-only so the
# series is cheap to write and never rewrites history; aggregation (streaks, sparklines, cross-user
# listener counts) reads the whole small log. This is the ONLY per-listen history we keep — playback
# stays last-position-only.


def _events_path(data_dir: Path, user_id: str) -> Path:
    return data_dir / "users" / user_id / "listen_events.jsonl"


# How far back a duplicate is looked for. The client replays its queue oldest-first in one pass,
# so a redelivered event is within a few lines of its original — this is a cheap tail scan, not an
# index, and it deliberately does not promise global uniqueness.
LISTEN_DEDUPE_TAIL_LINES = 500


def _is_duplicate_listen(path: Path, slug: str, iso_ts: str) -> bool:
    """Has this exact (slug, timestamp) already been appended recently?

    The offline queue replays an event that never got a RESPONSE, not one that never arrived — the
    server may well have appended it already. Both attempts carry the same ``client_ts``, so the
    pair is a natural idempotency key and a lost 204 stops inflating the user's own stats (#1925
    review). Events without a client timestamp are stamped on arrival, so two genuine opens a
    second apart still record as two.
    """
    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8") as fh:
            tail = deque(fh, maxlen=LISTEN_DEDUPE_TAIL_LINES)
    except OSError:
        return False
    for line in tail:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.get("slug") == slug and rec.get("ts") == iso_ts:
            return True
    return False


def append_listen_event(
    data_dir: Path, user_id: str, slug: str, feed_id: str | None, ts: int
) -> None:
    """Append one 'opened this episode' event to the user's listen log.

    Canonical event (ADR-119): the shared ``{ts, schema, event_type}`` envelope with
    an ISO-8601 ``ts``. The epoch ``ts`` arg is converted to ISO; the consumers
    (``app_stats._ts_to_date`` / ``app_engagement_series._week_of_ts``) accept BOTH
    epoch and ISO, so pre-existing epoch-ts logs still bucket correctly.

    A redelivery of the same (slug, ts) is dropped — see ``_is_duplicate_listen``.
    """
    from ..obs.events import emit_event

    iso_ts = datetime.fromtimestamp(int(ts), timezone.utc).isoformat()
    # Dedupe only inside the window where the stamp is the client's own. Everything older than
    # CLIENT_TS_MAX_AGE_SECONDS clamps to the SAME floor timestamp, so `(slug, ts)` stops being a
    # distinguishing key there: two genuinely separate listens of one episode during a long
    # offline stretch would collapse into one (advisor 2.5). A redelivery that old is far less
    # likely than a real second listen, so the tie breaks toward keeping the data.
    floor = int(time.time()) - CLIENT_TS_MAX_AGE_SECONDS
    if int(ts) > floor and _is_duplicate_listen(_events_path(data_dir, user_id), str(slug), iso_ts):
        return

    emit_event(
        "listen",
        sink="file",
        path=_events_path(data_dir, user_id),
        ts=iso_ts,
        slug=str(slug),
        feed_id=feed_id,
    )


# --- topic exposure (#1923) ---
#
# What the listener was EXPOSED to, recorded when it happened. Topic interest is otherwise derived
# on READ from the episode set and time-decayed, which is right for "what are you into now" and
# wrong for two things it can never answer:
#
#   1. **What changed.** A decayed score depends on when it was computed, so the same history
#      yields a different answer tomorrow. Drift needs a fixed record.
#   2. **Co-occurrence across users.** Same reason, plus deriving it means loading a KG artifact
#      per user per episode, for every read.
#
# One row per (episode, entity), which is the shape every other log here uses, and the one that
# keeps "which episode caused this exposure" — the question co-occurrence needs later and the one
# a daily rollup would throw away irreversibly.


def _exposure_path(data_dir: Path, user_id: str) -> Path:
    return data_dir / "users" / user_id / "topic_exposure.jsonl"


def append_topic_exposure(
    data_dir: Path, user_id: str, slug: str, entities: list[tuple[str, str, str]], ts: int
) -> int:
    """Record the topics and people one episode exposed the listener to. Returns rows written.

    Append-only and NOT deduplicated: hearing the same topic again next month is a second
    exposure, and that recurrence is the signal. Deduplication belongs to whoever aggregates.

    Never raises — this rides the listen path, and losing a statistic must not cost a listen.
    """
    if not entities:
        return 0
    path = _exposure_path(data_dir, user_id)
    iso = datetime.fromtimestamp(int(ts), timezone.utc).isoformat()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Same lock the trim takes: an append landing while the file is being REPLACED writes to
        # the old inode and is lost.
        with (
            _user_lock(data_dir, user_id, "topic_exposure"),
            path.open("a", encoding="utf-8") as fh,
        ):
            for kind, ent_id, label in entities:
                fh.write(
                    json.dumps(
                        {
                            "ts": iso,
                            "slug": str(slug),
                            "kind": str(kind),
                            "id": str(ent_id),
                            "label": str(label or ent_id),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        _trim_exposure(data_dir, user_id, path)
        return len(entities)
    except OSError:
        logger.debug("topic exposure write failed for %s/%s", user_id, slug, exc_info=True)
        return 0


def _trim_exposure(data_dir: Path, user_id: str, path: Path) -> None:
    """Keep the newest ``MAX_EXPOSURE_ROWS`` lines. Best-effort; never raises.

    Takes the exposure lock, like every other read-modify-write in this module: this REPLACES the
    file, and a concurrent append in another worker holds an fd on the old inode, so its rows would
    land on a file that no longer exists (advisor-2 #5).
    """
    try:
        if path.stat().st_size < MAX_EXPOSURE_ROWS * _EXPOSURE_MIN_BYTES_PER_ROW:
            return
        with _user_lock(data_dir, user_id, "topic_exposure"):
            lines = path.read_text(encoding="utf-8").splitlines()
            if len(lines) <= EXPOSURE_TRIM_TO_ROWS:
                return
            atomic_write_text(path, "\n".join(lines[-EXPOSURE_TRIM_TO_ROWS:]) + "\n")
    except OSError:
        logger.debug("topic exposure trim failed for %s", path, exc_info=True)


#: How many exposure rows are kept. One listen writes one row per topic/person in the episode —
#: dozens — so this file grows far faster than the listen log beside it, and every recap read
#: parses it twice (the window and the window before). Unbounded growth on a per-request read is
#: how a heavy listener's Profile gets slow and stays slow (advisor 2.6).
#:
#: Trimmed from the FRONT, so the newest rows survive: a recap only ever looks back a year, and the
#: oldest rows are the least useful for both drift and co-occurrence.
MAX_EXPOSURE_ROWS = 20_000

#: LOWER bound on a row's size, used for the cheap "worth counting lines?" check. It must be an
#: under-estimate so the check fires before the cap rather than after it.
#:
#: Two failure modes were hit getting this right, and both are why the low-water mark below exists
#: (advisor-2 #5):
#:   * too LOW an estimate (120) and a freshly trimmed file still sits above the trigger while
#:     being under the row cap — so every later append reads the whole file and rewrites nothing.
#:     The per-request full read this exists to prevent returns as a per-WRITE full read, for ever.
#:   * too HIGH an estimate (400) and the trigger fires long after the cap is passed.
#: A cheap size check cannot be both an exact ceiling and self-clearing, so the trim cuts to a LOW
#: WATER MARK well under the cap; the trimmed file is then unambiguously below the trigger.
_EXPOSURE_MIN_BYTES_PER_ROW = 80

#: Rows kept when a trim runs. Half the cap, so a trimmed file cannot immediately re-trigger.
EXPOSURE_TRIM_TO_ROWS = MAX_EXPOSURE_ROWS // 2


def list_topic_exposure(data_dir: Path, user_id: str) -> list[dict[str, Any]]:
    """One user's exposure rows, as written; skips blank and corrupt lines."""
    path = _exposure_path(data_dir, user_id)
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.get("id") and rec.get("ts") is not None:
            out.append(rec)
    return out


def list_listen_events(data_dir: Path, user_id: str) -> list[dict[str, Any]]:
    """All of one user's listen events (chronological as written); skips corrupt lines."""
    path = _events_path(data_dir, user_id)
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.get("slug") and rec.get("ts") is not None:
            out.append(rec)
    return out


def iter_user_ids(data_dir: Path) -> list[str]:
    """Every user id with a per-user directory (for cross-user aggregation)."""
    users_dir = data_dir / "users"
    if not users_dir.is_dir():
        return []
    return [p.name for p in users_dir.iterdir() if p.is_dir()]


# --- queue (ordered list of slugs) ---


def get_queue(data_dir: Path, user_id: str) -> list[str]:
    """Return the user's play queue (ordered slugs); empty when unset."""
    data = _read(data_dir, user_id, "queue", [])
    return [str(x) for x in data] if isinstance(data, list) else []


def set_queue(data_dir: Path, user_id: str, items: list[str]) -> list[str]:
    """Replace the user's play queue; return the stored list.

    Locks even though it does not READ: the file now has read-modify-write writers beside it
    (``add_queue_item`` / ``remove_queue_item``), and an unlocked replace can land between their
    read and their write — the same reasoning ``set_interests`` records.
    """
    clean = [str(x) for x in items]
    with _user_lock(data_dir, user_id, "queue"):
        _write(data_dir, user_id, "queue", clean)
    return clean


def _queue_for_update(data_dir: Path, user_id: str) -> list[str]:
    """The queue, read STRICTLY, for a read-modify-write. CALLER MUST HOLD THE LOCK.

    Strict (``_read(..., strict=True)``) for the reason this module's header gives at length: a
    lenient read answers an unreadable file with ``[]``, and a mutator that then persists what it
    read replaces the user's whole queue with whatever it is adding. One transient bad read plus
    one "add to queue" and the rest is gone, permanently (advisor 1.3).
    """
    data = _read(data_dir, user_id, "queue", [], strict=True)
    if not isinstance(data, list):
        raise UserStateUnreadable(f"queue for {user_id} is not a list")
    return [str(x) for x in data]


def add_queue_item(data_dir: Path, user_id: str, slug: str, after: str | None = None) -> list[str]:
    """Queue one episode; return the stored list.

    Idempotent in the sense that matters for a replayed offline write: the episode ends up queued
    exactly once. A repeat of a plain append is a no-op rather than a duplicate, and a repeat of a
    "play next" re-anchors it — the user's most recent intent for that slug wins, which is what a
    second identical request means.

    Holds the queue lock across read AND write. These were the only unlocked read-modify-write
    mutators in the module, which is a lost update under multi-worker uvicorn: a boot flush
    replaying "add ep-1" from one device while another adds ep-2 keeps one and silently drops the
    other (advisor 1.3).
    """
    with _user_lock(data_dir, user_id, "queue"):
        items = _queue_for_update(data_dir, user_id)
        if after is None and slug in items:
            return items
        items = [x for x in items if x != slug]
        if after is None:
            items.append(slug)
        else:
            idx = items.index(after) if after in items else -1
            items.insert(idx + 1, slug)
        _write(data_dir, user_id, "queue", items)
        return items


def remove_queue_item(data_dir: Path, user_id: str, slug: str) -> list[str]:
    """Drop one episode from the queue; return the stored list. A no-op when it is not there."""
    with _user_lock(data_dir, user_id, "queue"):
        items = _queue_for_update(data_dir, user_id)
        if slug not in items:
            return items
        items = [x for x in items if x != slug]
        _write(data_dir, user_id, "queue", items)
        return items


def _upsert_in_place(
    rows: list[dict[str, Any]], item: dict[str, Any], matches: "Callable[[dict[str, Any]], bool]"
) -> None:
    """Replace the first row ``matches`` selects, KEEPING its slot and its ``added_at``.

    "Idempotent on kind+ref" was only half-true: re-saving removed the row and appended the new one,
    so the item jumped to the end of the list and got a fresh ``added_at`` — the route always stamps
    ``time.time()``. Two consequences, both wrong: re-saving something reordered the user's list for
    no reason they took, and (RFC-103) each re-save read as a fresh save in the weekly momentum
    series, inflating engagement with an action that changed nothing.

    ``added_at`` is when the thing was FIRST saved. A re-save is the same save. Contrast
    :func:`add_interest`, which has always got this right — stable position, event only on a new
    follow.
    """
    for i, row in enumerate(rows):
        if matches(row):
            rows[i] = {**item, "added_at": row.get("added_at", item.get("added_at"))}
            return
    rows.append(item)


# --- favorites (polymorphic "saved things": episodes, insights, … keyed by kind+ref) ---


def get_favorites(data_dir: Path, user_id: str) -> list[dict[str, Any]]:
    """Return the user's saved favorites (newest-last as stored); empty when unset."""
    data = _read(data_dir, user_id, "favorites", [])
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict) and x.get("kind") and x.get("ref")]


def add_favorite(data_dir: Path, user_id: str, item: dict[str, Any]) -> list[dict[str, Any]]:
    """Add/replace a favorite (idempotent on ``kind``+``ref``); appended newest-last.

    Raw rows, not :func:`get_favorites`: that getter both swallows an unreadable file AND drops
    rows missing ``kind``/``ref``, so reading through it meant one bad read wiped the list and one
    ordinary save silently purged any row a different schema version wrote. Returns the getter
    view, so the API still sees only rows it can render.
    """
    kind, ref = item.get("kind"), item.get("ref")
    with _user_lock(data_dir, user_id, "favorites"):
        favorites = _rows_for_update(data_dir, user_id, "favorites")
        _upsert_in_place(favorites, item, lambda x: (x.get("kind"), x.get("ref")) == (kind, ref))
        _write(data_dir, user_id, "favorites", favorites)
        return get_favorites(data_dir, user_id)


def remove_favorite(data_dir: Path, user_id: str, kind: str, ref: str) -> list[dict[str, Any]]:
    """Remove a favorite by ``kind``+``ref`` (no-op if absent); return the remaining list."""
    with _user_lock(data_dir, user_id, "favorites"):
        favorites = [
            x
            for x in _rows_for_update(data_dir, user_id, "favorites")
            if (x.get("kind"), x.get("ref")) != (kind, ref)
        ]
        _write(data_dir, user_id, "favorites", favorites)
        return get_favorites(data_dir, user_id)


# --- interests (personalized discovery; ordered list of cluster ids) ---


def get_interests(data_dir: Path, user_id: str) -> list[str]:
    """Return the user's interest cluster ids (graph_compound_parent_id); empty when unset."""
    data = _read(data_dir, user_id, "interests", [])
    return [str(x) for x in data] if isinstance(data, list) else []


def _write_interests(data_dir: Path, user_id: str, cluster_ids: list[str]) -> list[str]:
    """De-duplicate (order preserved, blanks dropped) and write. CALLER MUST HOLD THE LOCK.

    Split out so the lock is taken in exactly one place per call path. ``_user_lock`` mints a fresh
    ``FileLock`` each call and is not a singleton, so a nested acquire from a mutator that already
    holds the lock would block until timeout rather than re-enter.
    """
    seen: set[str] = set()
    clean: list[str] = []
    for x in cluster_ids:
        s = str(x)
        if s and s not in seen:
            seen.add(s)
            clean.append(s)
    _write(data_dir, user_id, "interests", clean)
    return clean


def set_interests(data_dir: Path, user_id: str, cluster_ids: list[str]) -> list[str]:
    """Replace the user's interests; return the stored list (de-duplicated, order preserved).

    Locked, unlike the queue's equally-replacing ``set_queue``, and the difference matters: this
    file also has read-modify-write writers. ``add_interest`` reads under the lock, so an unlocked
    PUT /interests landing between that read and its write was not last-write-wins —
    ``add_interest`` then persisted a list derived from the PRE-PUT state and the replacement
    vanished silently.
    Symptom: saving the interest picker in one tab while following a person from an entity card in
    another, and watching the picker save revert.
    """
    with _user_lock(data_dir, user_id, "interests"):
        return _write_interests(data_dir, user_id, cluster_ids)


def add_interest(data_dir: Path, user_id: str, token: str) -> list[str]:
    """Follow one interest token (cluster ``tc:``, topic ``topic:`` or person ``person:``).

    Reads strictly: ``get_interests`` answers an unreadable file with ``[]``, so following one topic
    over a bad ``interests.json`` replaced the user's whole profile with that single token — and per
    RFC-103 the interest list is the source of truth the momentum history is derived against.
    """
    with _user_lock(data_dir, user_id, "interests"):
        return _write_interests(
            data_dir, user_id, [*_strings_for_update(data_dir, user_id, "interests"), token]
        )


def remove_interest(data_dir: Path, user_id: str, token: str) -> list[str]:
    """Unfollow one interest token (no-op if absent); return the remaining list."""
    with _user_lock(data_dir, user_id, "interests"):
        return _write_interests(
            data_dir,
            user_id,
            [x for x in _strings_for_update(data_dir, user_id, "interests") if x != token],
        )


# Follow events — an append-only, timestamped log of interest follows (momentum engagement source,
# RFC-103). The interest LIST above stays the source of truth for "what's followed"; this log is the
# timestamped history a weekly-momentum series needs (the list carries no per-token time). One line
# of JSON ({token, ts}) per newly-followed token, in <data_dir>/users/<id>/interest_events.jsonl.


def record_interest_follow(data_dir: Path, user_id: str, token: str, ts: int) -> None:
    """Append a timestamped follow event for *token* (call only when it was newly followed)."""
    path = data_dir / "users" / user_id / "interest_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"token": str(token), "ts": int(ts)}, ensure_ascii=False) + "\n")


# --- library (subscriptions; list of {feed_id, feed_url?, title?, added_at?}) ---


def get_library(data_dir: Path, user_id: str) -> list[dict[str, Any]]:
    """Return the user's subscriptions; empty when unset."""
    data = _read(data_dir, user_id, "library", [])
    return [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []


def add_subscription(data_dir: Path, user_id: str, item: dict[str, Any]) -> list[dict[str, Any]]:
    """Add/replace a subscription by ``feed_id`` (idempotent on feed_id).

    Raw rows for the same reason as :func:`add_favorite` — an unreadable ``library.json`` read as
    ``[]``, so one follow replaced every subscription the user had with that single show.
    """
    feed_id = item.get("feed_id")
    with _user_lock(data_dir, user_id, "library"):
        library = _rows_for_update(data_dir, user_id, "library")
        _upsert_in_place(library, item, lambda x: x.get("feed_id") == feed_id)
        _write(data_dir, user_id, "library", library)
        return get_library(data_dir, user_id)


def remove_subscription(data_dir: Path, user_id: str, feed_id: str) -> list[dict[str, Any]]:
    """Remove a subscription by ``feed_id`` (no-op if absent); return the remaining list."""
    with _user_lock(data_dir, user_id, "library"):
        library = [
            x for x in _rows_for_update(data_dir, user_id, "library") if x.get("feed_id") != feed_id
        ]
        _write(data_dir, user_id, "library", library)
        return get_library(data_dir, user_id)


# --- highlights (P2 Capture, PRD-040 / RFC-098 §7: "mark this moment" + transcript spans) ---
#
# A highlight is a captured moment in an episode the user wants to keep: a transcript ``span``
# selection, a one-tap ``moment`` (a single timestamp), or a saved ``insight`` (grounded GIL claim).
# Stored as one ``highlights.json`` list (newest-last), keyed by an opaque ``id`` the route mints.
#
# **The timestamp is the stable anchor.** Char offsets and segment ids are positional and drift when
# an episode is re-scraped (transcript text shifts); ``start_ms``/``end_ms`` survive.
# ``reanchor_highlight`` recomputes the positional fields against a fresh transcript and NEVER
# drops a highlight — a span that no longer resolves is marked ``anchor_status="drifted"`` (§7).

# ``kind`` (span|moment|insight) + ``target`` (highlight|insight|episode) are validated at the API
# boundary by the route's Pydantic ``Literal`` fields; the store stays permissive and only protects
# the immutable identity fields below from being overwritten on update.
_IMMUTABLE_HIGHLIGHT_FIELDS = frozenset({"id", "episode_slug", "created_at"})


def get_highlights(
    data_dir: Path, user_id: str, episode_slug: str | None = None
) -> list[dict[str, Any]]:
    """Return saved highlights (newest-last), optionally scoped to one episode."""
    data = _read(data_dir, user_id, "highlights", [])
    if not isinstance(data, list):
        return []
    out = [
        x
        for x in data
        if isinstance(x, dict)
        and x.get("id")
        and x.get("episode_slug")
        and x.get("kind")
        and x.get(
            "created_at"
        )  # required by the Highlight response model; drop hand-corrupted rows
    ]
    if episode_slug is not None:
        out = [x for x in out if x.get("episode_slug") == episode_slug]
    return out


def add_highlight(data_dir: Path, user_id: str, item: dict[str, Any]) -> list[dict[str, Any]]:
    """Add/replace a highlight (idempotent on ``id``); appended newest-last."""
    hid = item.get("id")
    with _user_lock(data_dir, user_id, "highlights"):
        # Raw rows, not get_highlights(): that filters for the response model, and writing the
        # filtered list back would delete a row of a different schema version as a side effect of
        # adding an unrelated one.
        rows = [x for x in _rows_for_update(data_dir, user_id, "highlights") if x.get("id") != hid]
        rows.append(item)
        _write(data_dir, user_id, "highlights", rows)
        return get_highlights(data_dir, user_id)


def update_highlight(
    data_dir: Path, user_id: str, highlight_id: str, fields: dict[str, Any]
) -> dict[str, Any] | None:
    """Merge ``fields`` into a highlight by ``id`` (no-op if absent); return the updated record.

    Used for in-place edits (``color``, ``quote_text``) and persisting a re-anchor. ``id``,
    ``episode_slug`` and ``created_at`` are immutable and cannot be overwritten via ``fields``.
    """
    with _user_lock(data_dir, user_id, "highlights"):
        rows = _rows_for_update(data_dir, user_id, "highlights")
        updated: dict[str, Any] | None = None
        for rec in rows:
            if rec.get("id") == highlight_id:
                rec.update(
                    {k: v for k, v in fields.items() if k not in _IMMUTABLE_HIGHLIGHT_FIELDS}
                )
                updated = rec
                break
        if updated is not None:
            _write(data_dir, user_id, "highlights", rows)
        return updated


def remove_highlight(data_dir: Path, user_id: str, highlight_id: str) -> list[dict[str, Any]]:
    """Remove a highlight by ``id`` (no-op if absent); return the remaining list."""
    with _user_lock(data_dir, user_id, "highlights"):
        rows = [
            x
            for x in _rows_for_update(data_dir, user_id, "highlights")
            if x.get("id") != highlight_id
        ]
        _write(data_dir, user_id, "highlights", rows)
        return get_highlights(data_dir, user_id)


def reanchor_highlight(highlight: dict[str, Any], segments: list[dict[str, Any]]) -> dict[str, Any]:
    """Re-resolve a highlight's positional fields against a fresh transcript.

    ``segments`` are the player's transcript contract (``segments_view.to_contract_segments``):
    ``{id, start, end, text}`` with **start/end in SECONDS**. That is the only segment shape this
    codebase produces, and naming the units here matters — an earlier signature asked for
    ``{segment_id, start_ms, end_ms, char_start, char_end}``, which nothing produced: the pipeline
    carries seconds and TRANSCRIPT-GLOBAL char offsets, while the client stores offsets relative to
    the first touched segment. Feeding one into the other silently rewrites offsets between two
    coordinate systems that share a field name.

    How an anchor is re-established, and why in this order:

    1. **Time selects the candidates.** ``start_ms``/``end_ms`` survive a re-scrape; segment ids do
       not (``to_contract_segments`` mints ``seg_{index}`` from list position, so inserting one
       segment renumbers every later id).
    2. **``quote_text`` decides whether they are RIGHT.** Time alone is not enough: if the audio
       timeline moved — an ad removed, an adfree segment file substituted whose own docstring warns
       it runs "minutes shorter" — the overlapping segment is simply the wrong passage, and
       stamping it ``anchored`` would be a confident lie. So the quote must actually be found in
       the candidates' text.
    3. **Finding the quote also recomputes the offsets**, in the client's coordinate system
       (relative to the first candidate segment), which is what makes them correct rather than
       merely present.

    A highlight is NEVER dropped. Unresolvable → ``anchor_status="drifted"`` with the stored
    positional fields left untouched, so a later re-anchor can recover it if the text returns.
    ``insight`` highlights are anchored by ``source_insight_id`` rather than time and pass through,
    so they receive no ``anchor_status`` at all — a GI re-run that retires an insight leaves the id
    dangling and nothing flags it.

    That is a deliberate gap, and VERIFIED as low-consequence rather than assumed (#34.7): the only
    consumers of ``source_insight_id`` are the client's save TOGGLE — ``savedInsightIds`` and the
    lookup that finds an existing highlight for an insight (``stores/capture.ts``). Nothing renders
    content through the id, because the quote is stored on the highlight at capture. So a dangling
    id degrades a toggle (the insight stops showing as already-saved); it never loses the capture.

    Revisit if any surface ever dereferences the id to READ the insight — at that point a stale id
    becomes a blank or wrong render, and these highlights need a status like the others.

    Returns a NEW dict; the input is not mutated.
    """
    result = dict(highlight)
    if highlight.get("kind") == "insight":
        return result
    start_ms = highlight.get("start_ms")
    if start_ms is None:
        result["anchor_status"] = "drifted"
        return result
    end_ms = highlight.get("end_ms")
    # A moment is a point; a span is a window. Treat end as start for point overlap.
    lo = int(start_ms)
    hi = int(end_ms) if end_ms is not None else lo
    if hi < lo:  # an inverted window would make the overlap test near-vacuous
        lo, hi = hi, lo

    # Overlap is STRICT for a span, matching the client's own selection maths
    # (transcriptCapture.ts: `r.start < sub.char_end && r.end > sub.char_start`). A `<=`/`>=` test
    # pulls in every segment the window merely TOUCHES: a 5s-9s span against 0-5s / 5-9s / 9-20s
    # matched all three, so a highlight of one sentence claimed the whole neighbourhood.
    #
    # A moment is a point, where strict comparison would match nothing at a boundary, so it uses
    # the half-open convention instead — the segment that CONTAINS the instant.
    is_point = lo == hi
    overlapping: list[dict[str, Any]] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        start = seg.get("start")
        end = seg.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        seg_lo = int(start * 1000)
        seg_hi = int(end * 1000)
        hit = (seg_lo <= lo < seg_hi) if is_point else (seg_lo < hi and seg_hi > lo)
        if hit:
            overlapping.append(seg)
    if not overlapping:
        result["anchor_status"] = "drifted"
        return result

    quote = str(highlight.get("quote_text") or "").strip()
    if not quote:
        # A `moment` carries no quote — there is nothing to verify against, so time is all we have.
        # Say so honestly in the status rather than claiming a verified anchor.
        result["segment_ids"] = [str(s.get("id")) for s in overlapping if s.get("id")]
        result["anchor_status"] = "time_only"
        return result

    # Search the candidates' concatenated text, joined exactly as the client concatenates the
    # rendered transcript, so a recovered offset means the same thing on both sides.
    joined = "".join(str(s.get("text") or "") for s in overlapping)
    found = joined.find(quote)
    if found < 0:
        # The window exists but no longer contains the quote — the timeline moved under it.
        result["anchor_status"] = "drifted"
        return result

    result["segment_ids"] = [str(s.get("id")) for s in overlapping if s.get("id")]
    result["char_start"] = found
    result["char_end"] = found + len(quote)
    result["anchor_status"] = "anchored"
    return result


# --- notes (P2 Capture: free-text notes attached to a highlight, insight or whole episode) ---
#
# A note is plain user text targeting one of three things (``target`` = highlight|insight|episode,
# ``target_id`` = its id/slug). Stored as one ``notes.json`` list, keyed by an opaque ``id``. A
# separate file from highlights so a note can attach independently (e.g. an episode-level note with
# no highlight). The route mints ``id``/``created_at``/``updated_at``.


def get_notes(
    data_dir: Path,
    user_id: str,
    target: str | None = None,
    target_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return saved notes (newest-last), optionally scoped to one ``target``/``target_id``."""
    data = _read(data_dir, user_id, "notes", [])
    if not isinstance(data, list):
        return []
    out = [
        x
        for x in data
        if isinstance(x, dict) and x.get("id") and x.get("target") and x.get("target_id")
    ]
    if target is not None:
        out = [x for x in out if x.get("target") == target]
    if target_id is not None:
        out = [x for x in out if x.get("target_id") == target_id]
    return out


def add_highlight_if_absent(
    data_dir: Path, user_id: str, item: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    """Add a highlight only if its id is free; return ``(record, created)``.

    ``add_highlight`` REPLACES on a matching id, which is right for an edit and wrong for a replay:
    a re-delivered offline capture would overwrite ``created_at`` (and the graph refs resolved at
    capture) with the moment the network came back. Under a client-minted id the first write wins
    and the replay is a no-op — a replay of the same create is the same create (#1925).

    The existence check and the append share ONE lock, so two concurrent replays of the same id
    cannot both see it absent.
    """
    hid = item.get("id")
    with _user_lock(data_dir, user_id, "highlights"):
        rows = _rows_for_update(data_dir, user_id, "highlights")
        for row in rows:
            if row.get("id") == hid:
                return dict(row), False
        rows.append(item)
        _write(data_dir, user_id, "highlights", rows)
        return dict(item), True


def add_note(data_dir: Path, user_id: str, item: dict[str, Any]) -> list[dict[str, Any]]:
    """Add/replace a note (idempotent on ``id``); appended newest-last."""
    nid = item.get("id")
    with _user_lock(data_dir, user_id, "notes"):
        # Raw rows, not get_notes() — same reason as add_highlight: the getter filters for the
        # response model, and persisting that filtered list deletes rows it merely could not render.
        rows = [x for x in _rows_for_update(data_dir, user_id, "notes") if x.get("id") != nid]
        rows.append(item)
        _write(data_dir, user_id, "notes", rows)
        return get_notes(data_dir, user_id)


def add_note_if_absent(
    data_dir: Path, user_id: str, item: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    """Add a note only if its id is free; return ``(record, created)``.

    The if-absent half of client-minted ids (#1925) — see ``add_highlight_if_absent``.
    """
    nid = item.get("id")
    with _user_lock(data_dir, user_id, "notes"):
        rows = _rows_for_update(data_dir, user_id, "notes")
        for row in rows:
            if row.get("id") == nid:
                return dict(row), False
        rows.append(item)
        _write(data_dir, user_id, "notes", rows)
        return dict(item), True


def update_note(
    data_dir: Path, user_id: str, note_id: str, text: str, updated_at: int
) -> dict[str, Any] | None:
    """Edit a note's ``text`` by ``id`` (no-op if absent); return the updated record."""
    with _user_lock(data_dir, user_id, "notes"):
        rows = _rows_for_update(data_dir, user_id, "notes")
        updated: dict[str, Any] | None = None
        for rec in rows:
            if rec.get("id") == note_id:
                rec["text"] = text
                rec["updated_at"] = updated_at
                updated = rec
                break
        if updated is not None:
            _write(data_dir, user_id, "notes", rows)
        return updated


def remove_note(data_dir: Path, user_id: str, note_id: str) -> list[dict[str, Any]]:
    """Remove a note by ``id`` (no-op if absent); return the remaining list."""
    with _user_lock(data_dir, user_id, "notes"):
        rows = [x for x in _rows_for_update(data_dir, user_id, "notes") if x.get("id") != note_id]
        _write(data_dir, user_id, "notes", rows)
        return get_notes(data_dir, user_id)


def remove_notes_for_target(data_dir: Path, user_id: str, target: str, target_id: str) -> int:
    """Drop every note attached to one target; return how many went.

    Notes on a deleted highlight used to survive it server-side. The client pruned them LOCALLY
    (``capture.ts``), so they looked deleted and then RESURRECTED on the next full load — the worst
    of both: the user is told the note is gone, and it is not. The client's own filter is the intent
    the server was failing to implement, so the server implements it.

    Separate from :func:`remove_note` so the sweep takes the notes lock once, not once per note.
    """
    with _user_lock(data_dir, user_id, "notes"):
        rows = _rows_for_update(data_dir, user_id, "notes")
        keep = [
            x
            for x in rows
            if not (x.get("target") == target and str(x.get("target_id")) == str(target_id))
        ]
        if len(keep) == len(rows):
            return 0
        _write(data_dir, user_id, "notes", keep)
        return len(rows) - len(keep)


# --- resurfacing state (P3 #1123): per-highlight {last_surfaced, count} + pacing settings ---
#
# Read-time spaced resurfacing (RFC-101 §5) needs only to remember, per highlight, when it was last
# shown and how many times — the due ladder is computed on read (``app_resurfacing.select_due``).
# ``resurfacing.json`` = {highlight_id: {last_surfaced, count}}; ``resurfacing_settings.json`` =
# {paused}. No scheduler.


def get_resurfacing_state(data_dir: Path, user_id: str) -> dict[str, Any]:
    """Per-highlight resurfacing bookkeeping ({highlight_id: {last_surfaced, count}})."""
    data = _read(data_dir, user_id, "resurfacing", {})
    return data if isinstance(data, dict) else {}


def mark_surfaced(data_dir: Path, user_id: str, highlight_id: str, ts: int) -> dict[str, Any]:
    """Record that a highlight was just surfaced (bumps ``count``, sets ``last_surfaced``)."""
    with _user_lock(data_dir, user_id, "resurfacing"):
        # Strict read: an unreadable file must not be answered with {} and then persisted as a
        # single-key mapping — that erases every other highlight's schedule.
        data = _mapping_for_update(data_dir, user_id, "resurfacing")
        prev = data.get(highlight_id)
        # A hand-edited or corrupt count must not crash the route or, worse, index the ladder
        # negatively and silently pick the wrong rung.
        try:
            previous_count = int(prev.get("count", 0)) if isinstance(prev, dict) else 0
        except (TypeError, ValueError):
            previous_count = 0
        rec = {"last_surfaced": int(ts), "count": max(previous_count, 0) + 1}
        data[highlight_id] = rec
        _write(data_dir, user_id, "resurfacing", data)
        return rec


def remove_resurfacing_state(data_dir: Path, user_id: str, highlight_id: str) -> None:
    """Drop a highlight's schedule entry (no-op if absent) — the delete cascade for #39.

    ``select_due`` iterates HIGHLIGHTS and looks their state up, so an orphaned entry is never
    read and nothing misbehaves; the file simply grows for ever, one dead key per deleted capture.
    It is also the only per-user file that had no cascade at all, so it was the one place a
    "deleted" thing left a trace — the same shape of promise-vs-reality gap the note cascade fixed.
    """
    with _user_lock(data_dir, user_id, "resurfacing"):
        data = _mapping_for_update(data_dir, user_id, "resurfacing")
        if highlight_id not in data:
            return  # nothing to write — do not rewrite the file for a no-op delete
        data.pop(highlight_id, None)
        _write(data_dir, user_id, "resurfacing", data)


def get_resurfacing_settings(data_dir: Path, user_id: str) -> dict[str, Any]:
    """Pacing settings ({paused}); defaults to not-paused when unset."""
    data = _read(data_dir, user_id, "resurfacing_settings", {})
    paused = bool(data.get("paused")) if isinstance(data, dict) else False
    return {"paused": paused}


def set_resurfacing_settings(data_dir: Path, user_id: str, *, paused: bool) -> dict[str, Any]:
    """Replace the pacing settings; return the stored record."""
    settings = {"paused": bool(paused)}
    _write(data_dir, user_id, "resurfacing_settings", settings)
    return settings
