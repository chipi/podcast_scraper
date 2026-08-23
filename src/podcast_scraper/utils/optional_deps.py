"""Is this exception a missing OPTIONAL dependency, or a real failure?

WHY THIS IS SHARED AND NOT INLINE
The pipeline has to answer that question in more than one place, and answering it differently in
each is how you end up with a run that degrades in one stage and dies in the next for the very
same cause. That state existed: ``preload_ml_models_if_needed`` correctly treated a missing
package as non-fatal and logged "models will be loaded on-demand", while speaker detection hit the
same missing spaCy at its point of use and killed the run. One mechanism, one rule, both callers.

The rule itself is deliberately narrow: a missing PACKAGE degrades, everything else still fails.
A missing model FILE, a gated token, a timeout and a bug are all real failures that must stop the
run — that is what "fail fast for required models" was protecting, and it stays protected.
"""

from __future__ import annotations


def caused_by_missing_import(exc: BaseException, *, max_depth: int = 10) -> bool:
    """True when an ``ImportError`` sits anywhere behind ``exc``.

    Two things this has to get right, both learned by MEASURING the real exception rather than
    reading the raise sites:

    * **Walk the whole chain, not one link.** The providers re-wrap, so a missing
      ``openai-whisper`` arrives as ``ProviderDependencyError("Failed to preload Whisper model:
      …")`` wrapping ``ProviderDependencyError("openai-whisper library not installed")`` wrapping
      ``ImportError``. Checking only the first link calls that a hard failure.
    * **Follow ``__context__`` as well as ``__cause__``.** The raise sites are NOT consistent:
      ``MLProvider.preload`` uses ``raise … from e``, but ``_initialize_whisper`` uses a bare
      ``raise`` inside ``except ImportError``, which records the ImportError as ``__context__``
      only — ``__cause__`` is ``None`` there. Implicit chaining is still the cause; a check that
      trusts explicit chaining alone silently mis-classifies whichever sites happen to omit
      ``from``.

    * **Check ``exc`` ITSELF, not only what is behind it.** The version of this that lived in
      ``setup.py`` inspected the chain alone, which was sufficient there because the preload path
      always arrives pre-wrapped in ``ProviderDependencyError``. It is NOT sufficient in general:
      ``analyze_patterns`` -> ``_initialize_spacy`` -> ``import spacy`` raises a bare
      ``ModuleNotFoundError`` with ``__cause__`` and ``__context__`` both ``None``, so a
      chain-only walk reported "not a missing import" for the most literal missing import there
      is. Found by measuring 2026-08-17: the degrade was added, the run still died, and the
      traceback showed the same site untouched.

    ``max_depth`` and the seen-set guard against cycles; a real chain is two or three links.

    Note ``ModuleNotFoundError`` is a subclass of ``ImportError``, so it is covered.
    """
    if isinstance(exc, ImportError):
        return True
    seen: set[int] = set()
    queue: list[BaseException | None] = [exc.__cause__, exc.__context__]
    for _ in range(max_depth):
        nxt: list[BaseException | None] = []
        for cur in queue:
            if cur is None or id(cur) in seen:
                continue
            if isinstance(cur, ImportError):
                return True
            seen.add(id(cur))
            nxt.extend((cur.__cause__, cur.__context__))
        if not nxt:
            return False
        queue = nxt
    return False
