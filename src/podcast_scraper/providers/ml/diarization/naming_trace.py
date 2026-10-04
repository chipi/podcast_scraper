"""Per-voice decision trace of the speaker-naming ladder (#2276).

The diagnostics sidecar recorded what went INTO the roster and what came OUT, not the ladder in
between, so "how did this voice get this name" could only be re-derived by hand. ``NamingTrace``
is a pure observer handed to ``resolve_speaker_roster``: each stage reports what it proposed,
accepted, refused or overrode, per voice and per name, and the trace is written to the sidecar as
``decision_trace``. It never influences a decision: a roster built with a trace is identical to one
built without (pinned by tests and by replaying the corpus).

Without a trace the roster records into ``NullTrace``, which records nothing, so the ladder's code
carries plain calls and no branches.

A recording error must never cost an episode its roster: every recorder swallows its own exception
and marks the trace ``degraded`` instead.

Because the roster is deterministic, ``scripts/measure/roster_replay.py --trace-out`` rebuilds the
same trace for any stored episode offline.
"""

from __future__ import annotations

import functools
import json
import logging
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TypeVar

logger = logging.getLogger(__name__)

TRACE_VERSION = 1

_F = TypeVar("_F", bound=Callable[..., None])


def _recorder(fn: _F) -> _F:
    """Skip when the trace is off; on any error, mark the trace degraded instead of raising."""

    @functools.wraps(fn)
    def wrapper(self: "NamingTrace", *args: Any, **kwargs: Any) -> None:
        if not self.enabled:
            return None
        try:
            fn(self, *args, **kwargs)
        except Exception:  # noqa: BLE001 — an observer must never break what it observes
            self.degraded = True
            logger.debug("naming trace: %s failed", fn.__name__, exc_info=True)
        return None

    return wrapper  # type: ignore[return-value]


def _role_view(role: Any) -> Dict[str, Any]:
    """The traced fields of a ``SpeakerRole`` (duck-typed: the roster module imports this one)."""
    return {
        "name": getattr(role, "name", None),
        "role": getattr(role, "role", None),
        "named": bool(getattr(role, "named", False)),
        "source": getattr(role, "source", None),
        "voice_type": getattr(role, "voice_type", None),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    return str(value)


class NamingTrace:
    """Collects ``{rung, decision, ...}`` steps per voice, per name, and for the episode."""

    enabled = True

    def __init__(self) -> None:
        self.inputs: Dict[str, Any] = {}
        self.voices: Dict[str, List[Dict[str, Any]]] = {}
        self.names: Dict[str, List[Dict[str, Any]]] = {}
        self.episode: List[Dict[str, Any]] = []
        self.degraded = False

    # --- primitives -------------------------------------------------------------------------

    @_recorder
    def input(self, key: str, value: Any) -> None:
        """Record what a stage started from (pools, classifications, the LLM's answers)."""
        self.inputs[key] = value

    @_recorder
    def voice(self, voice: str, rung: str, decision: str, **detail: Any) -> None:
        """One step for one voice.

        ``decision`` is named / renamed / removed / set / changed / dropped / accepted / added /
        seated / restored (the voice's name or entry was set), stated (a source states this name),
        or skipped / refused / refused_spelling (it was not). ``name`` is always the name the step
        SET or refused; a proposal the step did not take is ``proposed``.
        """
        step = {"rung": rung, "decision": decision}
        step.update({k: v for k, v in detail.items() if v is not None})
        self.voices.setdefault(voice, []).append(step)

    @_recorder
    def name(self, name: str, rung: str, decision: str, **detail: Any) -> None:
        """One step for one candidate NAME (a pool exclusion, a publish-gate refusal)."""
        step = {"rung": rung, "decision": decision}
        step.update({k: v for k, v in detail.items() if v is not None})
        self.names.setdefault(name, []).append(step)

    @_recorder
    def note(self, rung: str, **detail: Any) -> None:
        """An episode-level step (e.g. which voices took host seats, in order)."""
        step: Dict[str, Any] = {"rung": rung}
        step.update(detail)
        self.episode.append(step)

    @_recorder
    def diff_names(
        self, rung: str, before: Mapping[str, str], after: Mapping[str, str], **detail: Any
    ) -> None:
        """Every voice whose NAME this stage set, changed or removed (``voice -> name`` maps)."""
        for v in sorted(set(before) | set(after)):
            old, new = before.get(v), after.get(v)
            if old == new:
                continue
            if old is None:
                self.voice(v, rung, "named", name=new, **detail)
            elif new is None:
                self.voice(v, rung, "removed", name=old, **detail)
            else:
                self.voice(v, rung, "renamed", name=new, previous=old, **detail)

    @_recorder
    def diff_roles(
        self, rung: str, before: Mapping[str, Any], after: Mapping[str, Any], **detail: Any
    ) -> None:
        """Every voice whose roster entry (name / role / named / source / type) changed.

        ``previous`` holds the old value of every changed field, including one that became None.
        """
        for v in sorted(set(before) | set(after)):
            old = _role_view(before[v]) if v in before else None
            new = _role_view(after[v]) if v in after else None
            if old == new:
                continue
            if old is None:
                self.voice(v, rung, "set", **new, **detail)  # type: ignore[arg-type]
            elif new is None:
                self.voice(v, rung, "dropped", **old, **detail)  # type: ignore[arg-type]
            else:
                changed = [k for k in new if new[k] != old[k]]
                self.voice(
                    v,
                    rung,
                    "changed",
                    **{k: new[k] for k in changed},
                    cleared=[k for k in changed if new[k] is None] or None,
                    previous={k: old[k] for k in changed},
                    **detail,
                )

    # --- stage helpers (keep the roster's own code free of loops and branches) ----------------

    @_recorder
    def roster_inputs(self, **inputs: Any) -> None:
        """The roster's inputs, as received."""
        for key, value in inputs.items():
            self.input(key, value)

    @_recorder
    def added_to_pool(self, rung: str, before: Sequence[str], after: Sequence[str]) -> None:
        """Names a stage added to a pool (e.g. a guest host the description states)."""
        added = [n for n in after if n not in before]
        if added:
            self.note(rung, added=added)

    @_recorder
    def appended(
        self, rung: str, seq: Sequence[str], start: int, decision: str, **detail: Any
    ) -> None:
        """Every voice a step appended to ``seq`` since index ``start`` (e.g. a host-seat step)."""
        for v in list(seq)[start:]:
            self.voice(v, rung, decision, **detail)

    @_recorder
    def stated(self, rung: str, names_by_voice: Mapping[str, str]) -> None:
        """Names a source states per voice (the publisher's own label), whether or not they change
        what the voice already had — the name diff that follows records only the changes."""
        for v in sorted(names_by_voice):
            self.voice(v, rung, "stated", name=names_by_voice[v])

    @_recorder
    def refused_spellings(
        self,
        voices: Iterable[str],
        proposed: Mapping[str, str],
        stated: Sequence[str],
        resembles: Optional[Callable[[str, Sequence[str]], Optional[str]]] = None,
    ) -> None:
        """Voices whose introduced spelling was refused: it names nobody the episode states."""
        for v in sorted(voices):
            name = proposed.get(v)
            self.voice(
                v,
                "intro_reader",
                "refused_spelling",
                name=name,
                resembles=resembles(name, stated) if resembles and name else None,
                reason="introduced_name_not_stated",
            )

    @_recorder
    def host_pool(self, pool: Sequence[Tuple[str, str]], named: Sequence[Tuple[str, str]]) -> None:
        """The host pool with each name's source; entries that may seat but never name."""
        self.input("host_pool", [[n, s] for n, s in pool])
        for n, s in pool:
            if (n, s) not in named:
                self.name(n, "host_pool", "never_names", reason="names_the_show", source=s)

    @_recorder
    def host_seats(self, seats: Sequence[str], cohost_said_present: Any) -> None:
        """The voices that took host seats, in seat order."""
        self.note("host_seats", voices=list(seats), cohost_said_present=cohost_said_present)
        for i, v in enumerate(seats):
            self.voice(v, "host_seat", "seated", order=i)

    @_recorder
    def guest_pool(
        self,
        declared: Sequence[str],
        guest_names: Sequence[str],
        host_names_lower: Iterable[str],
        intro_names_lower: Iterable[str],
    ) -> None:
        """The guest pool, and why each declared name that did not make it was excluded."""
        hosts, intros = set(host_names_lower), set(intro_names_lower)
        for g in declared:
            if g in guest_names:
                continue
            if g.lower() in hosts:
                why = "host_pool_name"
            elif g.lower() in intros:
                why = "already_named_on_a_voice"
            else:
                why = "same_person_as_a_named_voice"
            self.name(g, "guest_pool", "excluded", reason=why)
        self.note("guest_pool", declared=list(declared), guest_names=list(guest_names))

    @_recorder
    def publish_refused(self, voice: str, name: str, reason: str) -> None:
        """The publish gate demoted this voice's name back to the raw label."""
        self.voice(voice, "publish_gate", "refused", name=name, reason=reason)
        self.name(name, "publish_gate", "refused", reason=reason, voice=voice)

    def to_dict(self) -> Dict[str, Any]:
        """The ``decision_trace`` block for the diagnostics sidecar: a JSON-safe COPY.

        Never raises. A value JSON cannot hold (a set, an enum) is converted; a trace that still
        cannot be serialised (e.g. NaN) is replaced by a stub saying so, so the sidecar is written.
        """
        raw = {
            "version": TRACE_VERSION,
            "degraded": self.degraded,
            "inputs": self.inputs,
            "voices": self.voices,
            "names": self.names,
            "episode": self.episode,
        }
        try:
            out: Dict[str, Any] = json.loads(json.dumps(raw, allow_nan=False, default=_jsonable))
            return out
        except Exception as exc:  # noqa: BLE001 — the sidecar must still be written
            logger.debug("naming trace: not serialisable", exc_info=True)
            return {"version": TRACE_VERSION, "degraded": True, "error": type(exc).__name__}


class NullTrace(NamingTrace):
    """Records nothing: the roster's default when no trace is requested."""

    enabled = False


def or_null(trace: "NamingTrace | None") -> NamingTrace:
    """The trace a helper records into: the caller's, or a ``NullTrace`` when none was passed."""
    return trace if trace is not None else NullTrace()
