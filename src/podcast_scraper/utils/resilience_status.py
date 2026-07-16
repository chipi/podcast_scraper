"""One place to ASK "what is resilience doing right now?" and to RESET it — the operator surface.

ADR-113 shipped the fuses and per-model resilience but left a gap: when a breaker opens or a fuse
blows, an operator had no way to SEE it or ACT on it short of grepping logs. This module closes that
gap on the read/act side; the server API, the o11y MCP tool, and the operator UI are all thin
wrappers over the two functions here.

State lives in-process (the LLM breaker's ``_provider_state``, the RSS ``http_policy`` globals), so
this reports and resets THIS process's resilience — which for the long-running pipeline server is
exactly the state that matters. Ephemeral CLI/eval runs carry their own fuse per run and reset on
exit, so there is nothing to observe there.

FUSES vs BREAKERS, on the reset question:
* **breakers** self-heal on a cooldown; reset only FORCE-CLOSES them early (e.g. you know the outage
  is over and don't want to wait out the cooldown).
* **fuses** are per-run hard stops — there is no persistent state to reset; the record here is
  informational (the configured budgets), and recovery is fix-the-cause-and-rerun.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# The providers whose LLM breaker we surface. Kept explicit so the status view lists every one, even
# those that have not failed yet (absence would read as "no breaker" rather than "closed").
_LLM_PROVIDERS = ("openai", "anthropic", "gemini", "mistral", "deepseek", "grok", "ollama")


def resilience_snapshot(cfg: Optional[Any] = None) -> Dict[str, Any]:
    """A single read of the whole resilience posture: LLM breakers, RSS breaker, and fuse budgets.

    Shape is stable and JSON-safe so the server API, the MCP tool, and the UI can all render it.
    """
    from .llm_circuit_breaker import stats as _llm_stats

    llm: Dict[str, Any] = {}
    open_llm: List[str] = []
    for provider in _LLM_PROVIDERS:
        try:
            s = dict(_llm_stats(provider))
        except Exception:
            continue
        # "open" for this wait-and-resume breaker == currently in its cooldown window.
        is_open = bool(s.get("in_cooldown"))
        entry = {
            "open": is_open,
            "recent_failures": int(s.get("recent_failures_in_window", 0)),
            "cooldown_remaining_seconds": round(float(s.get("cooldown_remaining_seconds", 0.0)), 1),
            "trips_total": int(s.get("trips_total", 0)),
        }
        llm[provider] = entry
        if is_open:
            open_llm.append(provider)

    try:
        from ..rss.http_policy import get_http_policy_metrics_snapshot

        rss = get_http_policy_metrics_snapshot()
    except Exception:
        rss = {}

    fuses = {
        "llm_max_calls_per_episode": getattr(cfg, "llm_max_calls_per_episode", None),
        "llm_max_calls_per_run": getattr(cfg, "llm_max_calls_per_run", None),
        "note": (
            "Fuses are per-run hard stops (money/access). They have no persistent state to reset — "
            "recovery is fix-the-cause-and-rerun. The breakers above self-heal on a cooldown; "
            "resetting them force-closes early."
        ),
    }

    return {
        "llm_breakers": llm,
        "llm_breakers_open": open_llm,
        "rss": rss,
        "fuses": fuses,
        "any_open": bool(open_llm) or bool(rss.get("circuit_breaker_open_feeds")),
    }


def reset_resilience(scope: str = "all") -> Dict[str, Any]:
    """Force-close breakers early (the operator "plug it back in" control).

    ``scope``: ``"llm"`` resets the per-provider LLM breakers; ``"rss"`` resets the RSS host
    breaker; ``"all"`` does both. Returns what was reset. Fuses are untouched (nothing to reset).
    """
    did: List[str] = []
    if scope in ("all", "llm"):
        try:
            from . import llm_circuit_breaker

            llm_circuit_breaker.reset_all()  # force-closes every provider breaker
            did.append("llm_breakers")
        except Exception:
            pass
    if scope in ("all", "rss"):
        try:
            from ..rss import http_policy

            http_policy.reset_http_policy_metrics()
            did.append("rss_breaker")
        except Exception:
            pass
    return {"reset": did, "scope": scope}
