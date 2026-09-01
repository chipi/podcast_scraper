"""A profile routing a REASONING model through LiteLLM must turn reasoning off.

Reasoning tokens are billed against ``max_tokens`` but never appear in ``content``. The insight
budget is ``max_insights * GI_INSIGHT_TOKENS_EACH`` — 50 * 50 = 2500 for a typical episode — and
a DeepSeek-v4 model spends that entirely on ``reasoning_content`` before writing a single
insight. Measured live against the homelab gateway on 2026-08-31, production's own alias
``podcast-flash-0731`` on a real 52k-char transcript:

    extra_body                 budget   reasoning_tokens   insights   finish
    profile (reasoning off)      2500                  0         35   stop
    profile (reasoning off)      7500                  0         41   stop
    (omitted)                    2500               2500          0   length
    (omitted)                    7500               2765         17   stop

So the directive is load-bearing: without it the episode silently gets ZERO insights, the
guardrail raises on finish_reason=length, salvage finds nothing recoverable, and the run still
exits 0. Every current profile sets it — this test is here so that stays true.

Why a test and not a comment: the two providers that already knew about this
(``deepseek_provider._REASONING_TOKEN_HEADROOM``, ``groq_provider``) each carry a headroom
FALLBACK for a run that leaves reasoning on. ``LiteLLMProvider`` inherits ``generate_insights``
unchanged and its ``_token_kwarg`` is a bare ``{"max_tokens": n}`` — no headroom, no warning.
On the LiteLLM path the profile directive is the ONLY thing standing between a reasoning model
and a silently empty insight set.

Adding the headroom fallback to LiteLLMProvider is the deeper fix and is tracked separately;
this guard is the cheap half that removes the trap today.

2026-09-01 — the SAME trap on the NATIVE DeepSeek path (#1892). ``cloud_with_dgx_primary``
pinned ``deepseek-v4-flash`` as its RFC-106 emergency tier without disabling thinking, so the
fallback could not answer at all. Reproduced against the real ``extract_quotes_bundled`` prompt
at every budget that stage uses:

    budget 4096 -> reasoning 4096, content 0
    budget 5888 -> reasoning 5885, content 0     (old 384/insight + headroom)
    budget 7168 -> reasoning 8661, content 0     (new 640/insight + headroom)

Reasoning alone EXCEEDED the whole budget on the last row, and varied 5698 -> 8661 across two
identical requests. ``deepseek_provider._REASONING_TOKEN_HEADROOM = 2048`` is the intended
safety net and is an order of magnitude short. That is why the guard below covers the deepseek
path too rather than trusting the headroom: a fallback tier that can never produce content is
worse than no fallback, because the ladder reports it as a handled event.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_PROFILES = Path(__file__).resolve().parents[3] / "config" / "profiles"

# Only the LiteLLM path is exposed. ``deepseek_provider`` and ``groq_provider`` carry their own
# reasoning-token headroom; ``LiteLLMProvider`` inherits generate_insights unchanged and its
# ``_token_kwarg`` is a bare ``{"max_tokens": n}``, so there the profile directive is the only
# protection.
_EXPOSED_PROVIDERS = {"litellm"}

# The rule is INVERTED from the obvious one, deliberately.
#
# The first cut matched gateway ALIAS NAMES against a marker list ("deepseek", "flash-0731",
# ...). That was wrong twice over. Too broad first: it matched "qwen3" and failed six
# vLLM/ollama profiles, including prod_dgx_full, whose Qwen3-30B-A3B-**Instruct** does not
# reason at all (72 episodes of the 2026-08-31 batch prove it). Then too NARROW once corrected:
# it selected 5 profiles out of the 12 that actually route a stage through LiteLLM, so the
# other 7 — every ``bakeoff_litellm_*`` — were vacuously green and would have said nothing if
# their directive were deleted. It also read only insight/summary model keys and so missed
# ``gi_value_gate_provider``, whose budget (``GI_VALUE_GATE_TOKENS_EACH = 24`` per insight) is
# the tightest in the pipeline and the one where a reasoning block is most lethal.
#
# A name allowlist is structurally unable to be right: it has to be updated every time an alias
# is renamed or a model is swapped behind one, and it fails SILENTLY when it isn't. So the rule
# is: any profile routing ANY stage through LiteLLM must state a reasoning posture. That is
# checkable without knowing what the alias points at today.


def _reasoning_is_disabled(extra_body: object) -> bool:
    """True if the body carries any directive meaning "do not think".

    Accepts every shape a profile might reasonably use, matching
    ``deepseek_provider._extra_body_disables_thinking``: the point is to detect INTENT, since
    which shape a given gateway honours depends on how it routes.
    """
    if not isinstance(extra_body, dict):
        return False
    if str(extra_body.get("reasoning_effort", "")).strip().lower() == "none":
        return True
    thinking = extra_body.get("thinking")
    if isinstance(thinking, dict) and str(thinking.get("type", "")).lower() == "disabled":
        return True
    reasoning = extra_body.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("enabled") is False:
        return True
    if extra_body.get("enable_thinking") is False:
        return True
    ctk = extra_body.get("chat_template_kwargs")
    if isinstance(ctk, dict) and ctk.get("enable_thinking") is False:
        return True
    return False


def _litellm_profiles():
    """Every profile routing ANY stage through LiteLLM, with the stages it routes.

    Scans all ``*_provider`` keys rather than a hand-listed few, so a stage added later (or one
    I did not think of, as ``gi_value_gate_provider`` was) is covered without an edit here.
    """
    out = []
    for path in sorted(_PROFILES.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:  # pragma: no cover - a broken profile fails its own test
            continue
        if not isinstance(data, dict):
            continue
        routed = sorted(
            k
            for k in data
            if k.endswith("_provider") and str(data.get(k) or "").lower() in _EXPOSED_PROVIDERS
        )
        if routed:
            out.append((path.name, data, routed))
    return out


def test_the_scan_finds_profiles():
    """Guards the guard — an empty or shrunken scan would make the assertion below vacuous.

    The floor is deliberately concrete: 12 profiles route a stage through LiteLLM today. A
    refactor that quietly stops matching most of them is the failure mode this catches, and it
    is exactly what the earlier alias-marker version did (it saw 5 of the 12).
    """
    found = _litellm_profiles()
    assert len(found) >= 12, (
        f"only {len(found)} litellm-routed profile(s) matched; the scan has narrowed and is "
        "silently guarding less than it claims"
    )
    names = {n for n, _, _ in found}
    assert "cloud_balanced.yaml" in names
    assert any(
        n.startswith("bakeoff_litellm_") for n in names
    ), "the bakeoff profiles route through LiteLLM too and were missed by the first version"


@pytest.mark.parametrize(
    "name,data,routed", _litellm_profiles(), ids=[p[0] for p in _litellm_profiles()]
)
def test_every_litellm_profile_declares_a_reasoning_posture(name, data, routed):
    assert _reasoning_is_disabled(data.get("litellm_extra_body")), (
        f"{name} routes {routed} through LiteLLM without stating a reasoning posture. "
        "LiteLLMProvider adds no reasoning-token headroom (unlike deepseek_provider and "
        "groq_provider), so if the alias resolves to a reasoning model the whole max_tokens "
        "budget is spent on reasoning_content: measured ZERO insights with "
        "finish_reason=length while the run exits 0. The tightest budget is the value gate at "
        "GI_VALUE_GATE_TOKENS_EACH=24 per insight. "
        "Set litellm_extra_body: {reasoning: {enabled: false}}."
    )


class TestTheDetectorItself:
    @pytest.mark.parametrize(
        "body",
        [
            {"reasoning": {"enabled": False}},
            {"reasoning_effort": "none"},
            {"thinking": {"type": "disabled"}},
            {"enable_thinking": False},
            {"chat_template_kwargs": {"enable_thinking": False}},
        ],
    )
    def test_accepts_every_disable_shape(self, body):
        assert _reasoning_is_disabled(body)

    @pytest.mark.parametrize(
        "body",
        [
            None,
            {},
            {"reasoning": {"enabled": True}},
            {"reasoning_effort": "low"},
            {"thinking": {"type": "enabled"}},
            {"provider": {"order": ["novita"]}},  # unrelated key must not count as a disable
        ],
    )
    def test_rejects_bodies_that_leave_reasoning_on(self, body):
        assert not _reasoning_is_disabled(body)


# --- the same trap on the NATIVE DeepSeek path (#1892) ---------------------------------------

#: Model-name markers for DeepSeek families that emit a reasoning block before the answer.
#: Mirrors ``deepseek_provider._REASONING_MODEL_MARKERS`` — kept in sync by the test below
#: rather than by hand, so a family added there cannot be silently missed here.
_DEEPSEEK_REASONING_MARKERS = ("v4", "-r1", "reasoner", "reasoning")

#: Every config key that pins a wire model on the deepseek tier. Scanned as a set, not
#: individually, because the governance audit has already caught a half-fix here once: only
#: ``deepseek_summary_model`` was pinned while cleaning/speaker kept leaking the old default.
_DEEPSEEK_MODEL_KEYS = (
    "deepseek_summary_model",
    "deepseek_cleaning_model",
    "deepseek_speaker_model",
    "deepseek_insight_model",
)


def _deepseek_reasoning_profiles():
    """Profiles that route a REASONING deepseek model on any tier."""
    out = []
    for path in sorted(_PROFILES.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:  # pragma: no cover
            continue
        if not isinstance(data, dict):
            continue
        models = [str(data.get(k) or "") for k in _DEEPSEEK_MODEL_KEYS]
        hit = [m for m in models if m and any(x in m.lower() for x in _DEEPSEEK_REASONING_MARKERS)]
        if hit:
            out.append((path.name, data, sorted(set(hit))))
    return out


def test_the_deepseek_scan_finds_profiles():
    """Guards the guard — an empty scan makes the assertion below vacuous."""
    found = _deepseek_reasoning_profiles()
    assert found, "no profile matched a reasoning deepseek model; the scan is broken"
    assert any(
        n == "cloud_with_dgx_primary.yaml" for n, _, _ in found
    ), "the profile that produced #1892 must be in scope"


@pytest.mark.parametrize(
    "name,data,models",
    _deepseek_reasoning_profiles(),
    ids=[p[0] for p in _deepseek_reasoning_profiles()],
)
def test_reasoning_deepseek_profiles_disable_thinking(name, data, models):
    assert _reasoning_is_disabled(data.get("deepseek_extra_body")), (
        f"{name} pins reasoning deepseek model(s) {models} without disabling thinking. The "
        "reasoning block consumes the whole output budget before any content, so the tier "
        "returns EMPTY content with finish_reason=length — measured reasoning 4096-8661 "
        "against budgets of 4096-7168, i.e. it cannot answer at ANY budget this pipeline "
        "uses. _REASONING_TOKEN_HEADROOM=2048 does not cover it. "
        "Set deepseek_extra_body: {reasoning_effort: none}."
    )


def test_the_markers_stay_in_sync_with_the_provider():
    """A family added to the provider but not here would silently escape the guard."""
    from podcast_scraper.providers.deepseek.deepseek_provider import _REASONING_MODEL_MARKERS

    assert set(_REASONING_MODEL_MARKERS) <= set(
        _DEEPSEEK_REASONING_MARKERS
    ), "deepseek_provider knows about a reasoning family this guard does not scan for"
