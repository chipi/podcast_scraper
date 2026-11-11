"""Methods a provider class gains from the stage that owns them (E04 split line, S1).

A provider class keeps the generic calls public uses (summary, translation, the basic KG pass).
The methods a moving stage adds (insight extraction, grounding, speaker and host detection) are a
separate class the stage owns. :class:`ExtensibleProvider` adds them back when a provider is built:
``OpenAIProvider(cfg)`` returns an instance of ``OpenAIProvider`` combined with every methods class
an owner registers for it or for one of its bases (``PipelineHooks.provider_methods``). Lookup
order is the one a direct base gave: the provider's own methods first, then the stage's.

Registrations are dotted import paths, so building one provider never imports another's SDK.
"""

from __future__ import annotations

import importlib
import threading
from typing import Any, Dict, List, Tuple

_CACHE: Dict[Tuple[type, Tuple[type, ...]], type] = {}
_LOCK = threading.Lock()


def _class_path(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _load(path: str) -> type:
    module, _, name = path.rpartition(".")
    loaded = getattr(importlib.import_module(module), name)
    if not isinstance(loaded, type):
        raise TypeError(f"provider methods {path!r} is not a class")
    return loaded


def methods_for(cls: type) -> Tuple[type, ...]:
    """The methods classes registered for ``cls`` or any of its bases, nearest class first."""
    from .. import stage_hooks

    registered = stage_hooks.provider_methods()
    out: List[type] = []
    for klass in cls.__mro__:
        for path in registered.get(_class_path(klass), ()):
            methods = _load(path)
            if methods not in out:
                out.append(methods)
    return tuple(out)


def composed_class(cls: type) -> type:
    """``cls`` with its registered methods classes after it, or ``cls`` when none is registered."""
    methods = methods_for(cls)
    if not methods:
        return cls
    key = (cls, methods)
    with _LOCK:
        composed = _CACHE.get(key)
        if composed is None:
            composed = type(
                cls.__name__,
                (cls, *methods),
                {
                    "__module__": cls.__module__,
                    "__qualname__": cls.__qualname__,
                    "_stage_methods_composed": True,
                },
            )
            _CACHE[key] = composed
    return composed


class _StageMethodsMeta(type):
    """Class-level access sees the registered methods too, as a direct base showed them:
    ``OpenAIProvider.generate_insights``, ``patch.object(OpenAIProvider, ...)`` (the patch lands
    on the provider class, which the composed class finds first) and ``dir()``."""

    def __getattr__(cls, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        for methods in methods_for(cls):
            if hasattr(methods, name):
                return getattr(methods, name)
        raise AttributeError(f"type object {cls.__name__!r} has no attribute {name!r}")

    def __dir__(cls) -> List[str]:
        names = set(super().__dir__())
        for methods in methods_for(cls):
            names.update(dir(methods))
        return sorted(names)


class ExtensibleProvider(metaclass=_StageMethodsMeta):
    """Base of a provider whose stage methods are registered rather than inherited."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        # A composed class (a copy, an unpickle) is built as it is, never composed again.
        if cls.__dict__.get("_stage_methods_composed"):
            return object.__new__(cls)
        return object.__new__(composed_class(cls))
