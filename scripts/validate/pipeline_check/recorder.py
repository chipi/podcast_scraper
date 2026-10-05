"""Record every keyed decision the pipeline makes, without changing its code.

Runs INSIDE the pipeline's process (the worker imports it before driving any stage). Two kinds of
decision point are discovered — not listed by hand, so a map or a function added next month is
traced without touching this tool:

* **keyed maps** — module-level dicts whose keys are values of the varied dimension (for the
  locale dimension: two or more language codes). Each is replaced, in place on its module, by a
  :class:`RecordingMap` with identical content that logs every lookup: the key asked for and
  whether it was there.
* **dimension functions** — module-level functions with a parameter named after the dimension
  (``language``, ``lang``, ``feed_language`` …). Each is wrapped, and every module attribute that
  referred to the original is repointed, so callers that did ``from x import f`` are traced too.
  The wrapper logs the parameter values and the return value.

WHAT IT CANNOT SEE: a value copied out of a map when its module was first imported (e.g.
``_THIS_IS_INTRO = _THIS_IS_INTRO_BY_LANGUAGE["en"]``) happened before this ran. Those are listed
by :func:`import_time_copies` so the report can name them as "checked statically, not traced"
rather than silently counting them as covered.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import pkgutil
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set

#: Parameter names that carry the locale dimension.
LOCALE_PARAMS = ("language", "lang", "feed_language", "source_language", "language_tag", "tag")

#: A fallback code set, used only when the code under test has no registry of its own (``main``
#: before the multilingual work has none). Enough to recognise a map keyed by language.
_FALLBACK_CODES = frozenset(
    "en es it fr de pt nl ca sv no nb nn ru ro bg sr ja ko zh ar pl tr uk cs el he hi hu fi da sk "
    "sl hr id vi th fa gl eu is ga cy".split()
)


@dataclass
class Event:
    """One decision: a lookup in a keyed map, or a call to a dimension function."""

    kind: str  # "map" | "call"
    site: str  # module.NAME (map) or module.function (call)
    key: Any  # the key asked for (map) or the dimension argument(s) (call)
    hit: Optional[bool] = None  # map: was the key present
    result: Any = None  # call: summary of the return value
    caller: str = ""  # file:line of the first frame outside this tool and the site's module
    variant: str = ""
    stage: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "site": self.site,
            "key": _jsonable(self.key),
            "hit": self.hit,
            "result": _jsonable(self.result),
            "caller": self.caller,
            "variant": self.variant,
            "stage": self.stage,
        }


@dataclass
class Recorder:
    """Holds the events, and the current variant/stage labels the worker sets."""

    events: List[Event] = field(default_factory=list)
    variant: str = ""
    stage: str = ""
    maps: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    skipped_modules: Dict[str, str] = field(default_factory=dict)

    def log(self, event: Event) -> None:
        event.variant = self.variant
        event.stage = self.stage
        self.events.append(event)


RECORDER = Recorder()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return f"<{type(value).__name__}>"


def _caller(skip_files: Set[str]) -> str:
    """``file:line`` of the pipeline code that made the decision.

    For a map lookup that is the line doing the lookup — usually inside the map's own module, so
    only this tool's frames (and *skip_files*, e.g. a wrapped function's own body) are skipped.
    """
    frame = sys._getframe(2)
    while frame is not None:
        fname = frame.f_code.co_filename
        if fname not in skip_files and fname != __file__ and "pipeline_check" not in fname:
            return f"{_short(fname)}:{frame.f_lineno}"
        frame = frame.f_back  # type: ignore[assignment]
    return ""


def _short(path: str) -> str:
    marker = "/podcast_scraper/"
    return path[path.rfind(marker) + 1 :] if marker in path else path


class RecordingMap(dict):
    """A dict with the original's content that logs every lookup of a key.

    Iteration, ``len`` and ``items()`` are NOT logged: they read the whole map (building a regex
    over every language, say), which is not a choice between languages.
    """

    def __init__(self, original: Dict[Any, Any], site: str, module_file: str) -> None:
        super().__init__(original)
        self._site = site
        self._skip = {__file__}  # the lookup line is IN the map's module: do not skip it

    def _note(self, key: Any) -> None:
        RECORDER.log(
            Event(
                kind="map",
                site=self._site,
                key=key,
                hit=dict.__contains__(self, key),
                caller=_caller(self._skip),
            )
        )

    def __getitem__(self, key: Any) -> Any:
        self._note(key)
        return dict.__getitem__(self, key)

    def get(self, key: Any, default: Any = None) -> Any:
        self._note(key)
        return dict.get(self, key, default)

    def __contains__(self, key: Any) -> bool:
        self._note(key)
        return dict.__contains__(self, key)


def _summary(value: Any) -> Any:
    """A comparable summary of a return value: scalars, container contents, pattern text."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "pattern") and hasattr(value, "flags"):
        return {"re": value.pattern, "flags": int(value.flags)}
    if isinstance(value, (set, frozenset, list, tuple)):
        return _jsonable(value)
    if isinstance(value, dict):
        return {"dict_keys": sorted(str(k) for k in value)}
    if isinstance(value, tuple):
        return _jsonable(value)
    return f"<{type(value).__name__}>"


def _wrap(fn: Callable[..., Any], site: str, params: Sequence[str]) -> Callable[..., Any]:
    sig = inspect.signature(fn)
    names = [p for p in params if p in sig.parameters]
    skip = {getattr(getattr(fn, "__code__", None), "co_filename", ""), __file__}

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            key = {n: bound.arguments.get(n) for n in names}
        except TypeError:
            key = {n: kwargs.get(n) for n in names}
        result = fn(*args, **kwargs)
        RECORDER.log(
            Event(kind="call", site=site, key=key, result=_summary(result), caller=_caller(skip))
        )
        return result

    wrapper.__pipeline_check_wrapped__ = True  # type: ignore[attr-defined]
    return wrapper


def dimension_codes() -> frozenset:
    """The language codes the code under test knows; the fallback set when it has no registry."""
    try:
        languages = importlib.import_module("podcast_scraper.languages")
        codes = getattr(languages, "ISO_639_1", None)
        if codes:
            return frozenset(codes)
    except Exception:  # noqa: BLE001 - absent on a base ref that predates it
        pass
    return _FALLBACK_CODES


def _is_keyed_map(value: Any, codes: frozenset) -> bool:
    if not isinstance(value, dict) or isinstance(value, RecordingMap) or len(value) < 2:
        return False
    keys = list(value)
    if not all(isinstance(k, str) for k in keys):
        return False
    return len({k for k in keys if k in codes}) >= 2 and all(k in codes for k in keys)


def iter_modules(packages: Iterable[str]) -> List[str]:
    """Every importable module name under *packages* (a package and its submodules)."""
    out: List[str] = []
    for name in packages:
        out.append(name)
        try:
            pkg = importlib.import_module(name)
        except Exception:  # noqa: BLE001 - reported by install()
            continue
        if hasattr(pkg, "__path__"):
            out.extend(m.name for m in pkgutil.walk_packages(pkg.__path__, prefix=name + "."))
    return out


def install(packages: Sequence[str], params: Sequence[str] = LOCALE_PARAMS) -> Recorder:
    """Discover and wrap every keyed map and dimension function in *packages*' modules.

    A module that fails to import (a missing optional dependency on this machine) is skipped and
    recorded in ``RECORDER.skipped_modules`` — the report names them, because a decision point in
    a module that could not load was not traced.
    """
    codes = dimension_codes()
    modules: Dict[str, types.ModuleType] = {}
    for name in iter_modules(packages):
        try:
            modules[name] = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            RECORDER.skipped_modules[name] = f"{type(exc).__name__}: {exc}"[:160]

    replaced: Dict[int, Callable[..., Any]] = {}
    for name, mod in modules.items():
        mod_file = getattr(mod, "__file__", "") or ""
        for attr, value in list(vars(mod).items()):
            if _is_keyed_map(value, codes):
                setattr(mod, attr, RecordingMap(value, f"{name}.{attr}", mod_file))
                RECORDER.maps.append(f"{name}.{attr}")
            elif (
                inspect.isfunction(value)
                and value.__module__ == name
                and not getattr(value, "__pipeline_check_wrapped__", False)
                and any(p in inspect.signature(value).parameters for p in params)
            ):
                wrapped = _wrap(value, f"{name}.{value.__name__}", params)
                replaced[id(value)] = wrapped
                setattr(mod, attr, wrapped)
                RECORDER.functions.append(f"{name}.{value.__name__}")

    # Repoint every other module's reference to a wrapped function (`from x import f` copies).
    for mod in list(sys.modules.values()):
        if mod is None or not getattr(mod, "__name__", "").startswith(tuple(packages)):
            continue
        for attr, value in list(vars(mod).items()):
            if inspect.isfunction(value) and id(value) in replaced:
                setattr(mod, attr, replaced[id(value)])
    return RECORDER


def import_time_copies(packages: Sequence[str]) -> List[str]:
    """Module-level values that are a copy of one row of a keyed map, taken at import.

    Detected as: a module attribute that IS (by identity) the value stored under some key of a
    keyed map in the same module. These were chosen before the recorder existed.
    """
    out: List[str] = []
    for name, mod in list(sys.modules.items()):
        if mod is None or not name.startswith(tuple(packages)):
            continue
        maps = [(a, v) for a, v in vars(mod).items() if isinstance(v, RecordingMap)]
        if not maps:
            continue
        rows = {id(row): f"{name}.{a}[{k!r}]" for a, m in maps for k, row in dict.items(m)}
        for attr, value in vars(mod).items():
            if (
                not isinstance(value, RecordingMap)
                and id(value) in rows
                and not isinstance(value, (str, int))
            ):
                out.append(f"{name}.{attr} = {rows[id(value)]}")
    return sorted(out)
