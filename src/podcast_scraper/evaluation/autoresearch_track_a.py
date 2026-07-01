"""Track A autoresearch helpers: env keys, ROUGE extraction, dual LLM judges.

Logic here is imported by ``autoresearch/prompt_tuning/eval/score.py`` and unit-tested
without running the full experiment subprocess.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from podcast_scraper.utils.log_redaction import format_exception_for_log

logger = logging.getLogger(__name__)


from podcast_scraper.utils.runtime_env import is_pytest_run as _is_pytest_run  # noqa: E402


def load_local_dotenv_files(repo_root: Path) -> None:
    """Load ``.env`` then optional ``.env.autoresearch`` from the project root.

    Used by ``autoresearch/prompt_tuning/eval/score.py`` because that CLI does not
    import ``podcast_scraper.config`` (which normally loads ``.env``).

    - Skips entirely in test environments (same policy as ``config``).
    - ``.env``: ``override=False`` (shell-exported vars win).
    - ``.env.autoresearch``: ``override=True`` so autoresearch-only keys can override
      values from ``.env`` when both define the same variable.

    You can put ``AUTORESEARCH_*`` (and optional ``AUTORESEARCH_ALLOW_PRODUCTION_KEYS``)
    in the main ``.env`` or split them into ``.env.autoresearch``; both are gitignored.
    """
    if _is_pytest_run():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.debug("python-dotenv not installed; skip .env loading for autoresearch score")
        return

    env_main = repo_root / ".env"
    if env_main.is_file():
        try:
            load_dotenv(env_main, override=False)
        except (OSError, PermissionError) as e:
            logger.warning(
                "Could not load %s: %s",
                env_main,
                format_exception_for_log(e),
            )

    env_ar = repo_root / ".env.autoresearch"
    if env_ar.is_file():
        try:
            load_dotenv(env_ar, override=True)
        except (OSError, PermissionError) as e:
            logger.warning(
                "Could not load %s: %s",
                env_ar,
                format_exception_for_log(e),
            )


ALLOW_PRODUCTION_KEYS_ENV = "AUTORESEARCH_ALLOW_PRODUCTION_KEYS"
EXPERIMENT_OPENAI_KEY_ENV = "AUTORESEARCH_EXPERIMENT_OPENAI_API_KEY"
JUDGE_OPENAI_KEY_ENV = "AUTORESEARCH_JUDGE_OPENAI_API_KEY"
JUDGE_ANTHROPIC_KEY_ENV = "AUTORESEARCH_JUDGE_ANTHROPIC_API_KEY"

# Experiment keys per provider: type → (autoresearch env var, production fallback env var)
_PROVIDER_EXPERIMENT_KEY_MAP: Dict[str, Tuple[str, str]] = {
    "openai": ("AUTORESEARCH_EXPERIMENT_OPENAI_API_KEY", "OPENAI_API_KEY"),
    "anthropic": ("AUTORESEARCH_EXPERIMENT_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    "gemini": ("AUTORESEARCH_EXPERIMENT_GEMINI_API_KEY", "GEMINI_API_KEY"),
    "grok": ("AUTORESEARCH_EXPERIMENT_GROK_API_KEY", "GROK_API_KEY"),
    "deepseek": ("AUTORESEARCH_EXPERIMENT_DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY"),
    "mistral": ("AUTORESEARCH_EXPERIMENT_MISTRAL_API_KEY", "MISTRAL_API_KEY"),
}
# Env var name that run_experiment.py reads for each provider
_PROVIDER_RUNTIME_KEY_ENV: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "grok": "GROK_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}
EVAL_N_ENV = "AUTORESEARCH_EVAL_N"
ROUGE_WEIGHT_ENV = "AUTORESEARCH_SCORE_ROUGE_WEIGHT"
DEFAULT_ROUGE_WEIGHT = 0.4
MAX_TRANSCRIPT_CHARS = 28_000
DIVERGENCE_THRESHOLD = 0.25
# Fraction of episodes that must individually contest before the run is considered contested.
# Binary OR (any single episode) was too brittle at small dataset scales.
CONTEST_FRACTION_THRESHOLD = 0.40


class AutoresearchConfigError(RuntimeError):
    """Missing or invalid autoresearch environment configuration."""


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def resolve_experiment_openai_key() -> str:
    """API key for OpenAI summarization inside ``run_experiment`` subprocess."""
    direct = os.environ.get(EXPERIMENT_OPENAI_KEY_ENV, "").strip()
    if direct:
        return direct
    if _truthy_env(ALLOW_PRODUCTION_KEYS_ENV):
        prod = os.environ.get("OPENAI_API_KEY", "").strip()
        if prod:
            return prod
    raise AutoresearchConfigError(
        f"Set {EXPERIMENT_OPENAI_KEY_ENV} or set {ALLOW_PRODUCTION_KEYS_ENV}=1 with "
        "OPENAI_API_KEY for local development."
    )


def resolve_experiment_provider_key(provider: str) -> str:
    """Return the experiment API key for *provider*, injected into the subprocess env.

    Checks ``AUTORESEARCH_EXPERIMENT_{PROVIDER}_API_KEY`` first.  Falls back to the
    production key (e.g. ``ANTHROPIC_API_KEY``) only when
    ``AUTORESEARCH_ALLOW_PRODUCTION_KEYS=1``.

    Args:
        provider: Backend type string, e.g. ``"anthropic"``, ``"gemini"``, ``"grok"``.

    Returns:
        API key string to inject as the provider's runtime env var.

    Raises:
        AutoresearchConfigError: Key not found and production fallback not allowed.
    """
    entry = _PROVIDER_EXPERIMENT_KEY_MAP.get(provider)
    if entry is None:
        raise AutoresearchConfigError(
            f"No experiment key mapping for provider '{provider}'. "
            f"Supported: {list(_PROVIDER_EXPERIMENT_KEY_MAP)}"
        )
    autoresearch_env, prod_env = entry
    direct = os.environ.get(autoresearch_env, "").strip()
    if direct:
        return direct
    if _truthy_env(ALLOW_PRODUCTION_KEYS_ENV):
        prod = os.environ.get(prod_env, "").strip()
        if prod:
            return prod
    raise AutoresearchConfigError(
        f"Set {autoresearch_env} or set {ALLOW_PRODUCTION_KEYS_ENV}=1 with {prod_env}."
    )


def provider_runtime_key_env(provider: str) -> str:
    """Return the env var name that ``run_experiment.py`` reads for *provider*.

    Used by ``score.py`` to know which env var to override in the subprocess.
    E.g. ``"anthropic"`` → ``"ANTHROPIC_API_KEY"``.
    """
    key = _PROVIDER_RUNTIME_KEY_ENV.get(provider)
    if key is None:
        raise AutoresearchConfigError(
            f"No runtime key env for provider '{provider}'. "
            f"Supported: {list(_PROVIDER_RUNTIME_KEY_ENV)}"
        )
    return key


def resolve_judge_openai_key() -> str:
    """Return OpenAI API key for the autoresearch judge (dedicated env or production fallback)."""
    key = os.environ.get(JUDGE_OPENAI_KEY_ENV, "").strip()
    if key:
        return key
    if _truthy_env(ALLOW_PRODUCTION_KEYS_ENV):
        prod = os.environ.get("OPENAI_API_KEY", "").strip()
        if prod:
            return prod
    raise AutoresearchConfigError(
        f"Set {JUDGE_OPENAI_KEY_ENV} or {ALLOW_PRODUCTION_KEYS_ENV}=1 with OPENAI_API_KEY."
    )


def resolve_judge_anthropic_key() -> str:
    """Return Anthropic API key for the judge (dedicated env or production fallback)."""
    key = os.environ.get(JUDGE_ANTHROPIC_KEY_ENV, "").strip()
    if key:
        return key
    if _truthy_env(ALLOW_PRODUCTION_KEYS_ENV):
        prod = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if prod:
            return prod
    raise AutoresearchConfigError(
        f"Set {JUDGE_ANTHROPIC_KEY_ENV} or {ALLOW_PRODUCTION_KEYS_ENV}=1 with " "ANTHROPIC_API_KEY."
    )


def eval_n_from_env(default: int = 5) -> int:
    """Parse episode sample size for eval from ``EVAL_N_ENV`` (integer >= 1)."""
    raw = os.environ.get(EVAL_N_ENV, "").strip()
    if not raw:
        return default
    try:
        n = int(raw)
    except ValueError as e:
        raise AutoresearchConfigError(f"{EVAL_N_ENV} must be an integer, got {raw!r}") from e
    if n < 1:
        raise AutoresearchConfigError(f"{EVAL_N_ENV} must be >= 1")
    return n


def rouge_weight_from_env() -> float:
    """Parse scalar weight in [0, 1] for ROUGE vs judge blend from ``ROUGE_WEIGHT_ENV``."""
    raw = os.environ.get(ROUGE_WEIGHT_ENV, "").strip()
    if not raw:
        return DEFAULT_ROUGE_WEIGHT
    try:
        w = float(raw)
    except ValueError as e:
        raise AutoresearchConfigError(f"{ROUGE_WEIGHT_ENV} must be a float") from e
    if not 0.0 <= w <= 1.0:
        raise AutoresearchConfigError(f"{ROUGE_WEIGHT_ENV} must be in [0, 1]")
    return w


def extract_mean_rouge_l_f1(metrics: Dict[str, Any]) -> Optional[float]:
    """Return mean ROUGE-L F1 from ``score_run`` metrics (first successful vs_reference)."""
    vs = metrics.get("vs_reference")
    if not isinstance(vs, dict):
        return None
    for _ref_id, blob in vs.items():
        if not isinstance(blob, dict) or "error" in blob:
            continue
        val = blob.get("rougeL_f1")
        if val is not None:
            return float(val)
    return None


def load_judge_config(path: Path) -> Dict[str, Any]:
    """Load YAML judge configuration from ``path``; must deserialize to a mapping."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Judge config must be a mapping: {path}")
    return raw


def parse_judge_score_json(text: str) -> float:
    """Parse assistant text containing judge JSON; supports both legacy and per-dimension formats.

    Legacy format: ``{"score": 0.85, "notes": "..."}``
    Per-dimension format: ``{"coverage": 0.9, "accuracy": 1.0, "efficiency": 0.8,
                             "score": 0.9, "notes": "..."}``

    When per-dimension fields are present they are logged for visibility. ``score`` is always the
    authoritative value; if absent it is computed as the mean of the three dimension scores.
    """
    import logging as _logging

    _log = _logging.getLogger(__name__)
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("Judge JSON must be an object")

    # Log per-dimension breakdown when available.
    dims = {k: data.get(k) for k in ("coverage", "accuracy", "efficiency")}
    if any(v is not None for v in dims.values()):
        _log.debug(
            "Judge dimensions — coverage=%.3f accuracy=%.3f efficiency=%.3f",
            float(dims["coverage"] or 0),
            float(dims["accuracy"] or 0),
            float(dims["efficiency"] or 0),
        )

    score = data.get("score")
    if score is None:
        # Fall back to mean of dimensions if all three are present.
        if all(dims[k] is not None for k in ("coverage", "accuracy", "efficiency")):
            dim_vals = [float(dims[k] or 0) for k in ("coverage", "accuracy", "efficiency")]
            score = sum(dim_vals) / 3.0  # type: ignore[assignment]
    if score is None:
        raise ValueError("Judge JSON missing 'score' and not all dimension scores present")
    f = float(score)
    if f < 0.0 or f > 1.0:
        raise ValueError(f"score out of range [0,1]: {f}")
    return f


def _extract_summary_prose(summary: str) -> str:
    """Return human-readable prose from ``summary``.

    Bundled-mode predictions store ``summary_final`` as a JSON string
    ``{"title": "...", "summary": "...", "bullets": [...]}``.  Passing raw JSON
    to a judge model is ambiguous — one model may treat length of the JSON blob
    as a conciseness signal while another focuses on the prose inside.  Extract
    and format the prose fields so both judges evaluate the same content.

    Falls back to the original string if it is not parseable JSON or lacks the
    expected shape.
    """
    try:
        data = json.loads(summary)
        if not isinstance(data, dict):
            return summary
        parts: List[str] = []
        title = data.get("title")
        if title:
            parts.append(f"Title: {title}")
        prose = data.get("summary") or data.get("text")
        if prose:
            parts.append(f"Summary:\n{prose}")
        bullets = data.get("bullets")
        if bullets and isinstance(bullets, list):
            bullet_lines = "\n".join(f"- {b}" for b in bullets if b)
            parts.append(f"Key takeaways:\n{bullet_lines}")
        if parts:
            return "\n\n".join(parts)
    except (json.JSONDecodeError, TypeError):
        pass
    return summary


def _judge_user_message(*, rubric: str, transcript: str, summary: str) -> str:
    t = transcript if len(transcript) <= MAX_TRANSCRIPT_CHARS else transcript[:MAX_TRANSCRIPT_CHARS]
    formatted_summary = _extract_summary_prose(summary)
    return (
        "You evaluate podcast episode summaries against the transcript.\n\n"
        "### Rubric\n"
        f"{rubric}\n\n"
        "### Transcript (may be truncated)\n"
        f"{t}\n\n"
        "### Candidate summary\n"
        f"{formatted_summary}\n\n"
        "Reply with a single JSON object only, no markdown:\n"
        '{"coverage": <float 0.0-1.0>, "accuracy": <float 0.0-1.0>, '
        '"efficiency": <float 0.0-1.0>, "score": <mean of the three>, '
        '"notes": "<one short sentence>"}\n'
        "score 1.0 = fully satisfies all rubric dimensions; 0.0 = fails badly."
    )


def call_openai_judge(*, api_key: str, model: str, user_content: str) -> float:
    """Run one OpenAI chat completion and parse ``score`` from the assistant JSON reply."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[{"role": "user", "content": user_content}],
    )
    content = (resp.choices[0].message.content or "").strip()
    return parse_judge_score_json(content)


def call_anthropic_judge(*, api_key: str, model: str, user_content: str) -> float:
    """Run one Anthropic messages call and parse ``score`` from the assistant JSON reply."""
    import logging

    import anthropic

    logger = logging.getLogger(__name__)

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0.0,
        messages=[{"role": "user", "content": user_content}],
    )
    # Handle both Message object and string responses
    if isinstance(msg, str):
        content = msg.strip()
    else:
        logger.debug(
            "Message type: %s, stop_reason: %s, content blocks: %d",
            type(msg),
            getattr(msg, "stop_reason", "N/A"),
            len(msg.content),
        )
        parts: List[str] = []
        for i, block in enumerate(msg.content):
            logger.debug(
                "  Block %d: type=%s, has_text=%s",
                i,
                type(block).__name__,
                hasattr(block, "text"),
            )
            if hasattr(block, "text"):
                parts.append(block.text)
        content = "".join(parts).strip()
    return parse_judge_score_json(content)


@dataclass(frozen=True)
class JudgeOutcome:
    """Per-episode judge scores."""

    judge_a: float
    judge_b: float
    contested: bool


def _call_judge(
    provider: str,
    *,
    model: str,
    user_content: str,
    openai_key: str,
    anthropic_key: str,
) -> float:
    """Dispatch a single judge call by provider name.

    Supported providers:
    - ``openai`` — cloud API, needs ``openai_key``
    - ``anthropic`` — cloud API, needs ``anthropic_key``
    - ``ollama`` — local Ollama via ``OLLAMA_API_BASE`` / ``DGX_TAILNET_FQDN``,
      no key required (the runner reaches Ollama over the tailnet)
    - ``vllm`` — local vLLM on DGX autoresearch container via ``VLLM_API_BASE`` /
      ``DGX_TAILNET_FQDN`` (port 8003 — the autoresearch slot, NOT the
      ``coder-next`` IDE slot which is operator-only per
      [[project_dgx_vllm_distinction]]). No key required.
    """
    if provider == "openai":
        return call_openai_judge(api_key=openai_key, model=model, user_content=user_content)
    if provider == "anthropic":
        return call_anthropic_judge(api_key=anthropic_key, model=model, user_content=user_content)
    if provider == "ollama":
        # Imported lazily so test_unit-only envs without httpx don't need it.
        from podcast_scraper.evaluation.judges.ollama_chat import OllamaChatJudge

        return OllamaChatJudge(model=model).score(user_content=user_content)
    if provider == "vllm":
        from podcast_scraper.evaluation.judges.vllm_chat import VllmChatJudge

        return VllmChatJudge(model=model).score(user_content=user_content)
    raise ValueError(
        f"unsupported judge provider: {provider!r} " f"(supported: openai, anthropic, ollama, vllm)"
    )


def judge_one_episode(
    *,
    rubric: str,
    transcript: str,
    summary: str,
    judge_a_provider: str,
    judge_a_model: str,
    judge_b_provider: str,
    judge_b_model: str,
    openai_key: str,
    anthropic_key: str,
) -> JudgeOutcome:
    """Score one summary with two judges; mark contested if scores diverge.

    Providers per judge are configurable via ``judge_config*.yaml``. Cross-
    vendor diversity (see ``feedback_silver_judge_vendor_bias`` in operator
    memory) requires the two providers to come from different model families
    than any single candidate. For Ollama-only smokes that means picking
    judges from different Ollama model families than the candidate.
    """
    user_msg = _judge_user_message(rubric=rubric, transcript=transcript, summary=summary)
    s_a = _call_judge(
        judge_a_provider,
        model=judge_a_model,
        user_content=user_msg,
        openai_key=openai_key,
        anthropic_key=anthropic_key,
    )
    s_b = _call_judge(
        judge_b_provider,
        model=judge_b_model,
        user_content=user_msg,
        openai_key=openai_key,
        anthropic_key=anthropic_key,
    )
    contested = abs(s_a - s_b) > DIVERGENCE_THRESHOLD
    return JudgeOutcome(judge_a=s_a, judge_b=s_b, contested=contested)


def summary_text_from_prediction(pred: Dict[str, Any]) -> str:
    """Extract final summary string from a prediction dict (``output`` string or nested fields)."""
    out = pred.get("output") or {}
    if isinstance(out, str):
        return out
    text = out.get("summary_final") or out.get("summary_long")
    if isinstance(text, str):
        return text
    return ""


def transcripts_by_episode_id(
    *, dataset_id: str, episode_ids: List[str], eval_root: Path
) -> Dict[str, str]:
    """Load materialized transcript text files keyed by episode id under ``eval_root``."""
    base = eval_root / "materialized" / dataset_id
    out: Dict[str, str] = {}
    for eid in episode_ids:
        path = base / f"{eid}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Materialized transcript missing: {path}")
        out[eid] = path.read_text(encoding="utf-8")
    return out


def mean_judge_scores(
    *,
    predictions: List[Dict[str, Any]],
    rubric: str,
    judge_cfg: Dict[str, Any],
    dataset_id: str,
    eval_root: Path,
    openai_key: str,
    anthropic_key: str,
) -> Tuple[float, bool, List[JudgeOutcome]]:
    """Return (mean of per-episode judge midpoints, any_contested, per_episode outcomes)."""
    ja = judge_cfg.get("judge_a") or {}
    jb = judge_cfg.get("judge_b") or {}
    j_a_prov = str(ja.get("provider", "openai"))
    j_a_model = str(ja["model"])
    j_b_prov = str(jb.get("provider", "anthropic"))
    j_b_model = str(jb["model"])

    eids = [str(p.get("episode_id")) for p in predictions if p.get("episode_id")]
    transcripts = transcripts_by_episode_id(
        dataset_id=dataset_id, episode_ids=eids, eval_root=eval_root
    )
    mids: List[float] = []
    outcomes: List[JudgeOutcome] = []
    contested_count = 0
    for pred in predictions:
        eid = pred.get("episode_id")
        if not eid:
            continue
        summary = summary_text_from_prediction(pred)
        if not summary.strip():
            logger.warning("Skipping judge for episode %s: empty summary", eid)
            continue
        tr = transcripts.get(str(eid), "")
        o = judge_one_episode(
            rubric=rubric,
            transcript=tr,
            summary=summary,
            judge_a_provider=j_a_prov,
            judge_a_model=j_a_model,
            judge_b_provider=j_b_prov,
            judge_b_model=j_b_model,
            openai_key=openai_key,
            anthropic_key=anthropic_key,
        )
        outcomes.append(o)
        mids.append((o.judge_a + o.judge_b) / 2.0)
        if o.contested:
            contested_count += 1
    if not mids:
        raise RuntimeError("No episodes scored by judges")
    # Contested when a meaningful fraction of episodes diverge — not just one.
    # A single outlier episode should not collapse the entire run to ROUGE-only.
    any_contested = (contested_count / len(mids)) > CONTEST_FRACTION_THRESHOLD
    logger.info(
        "Judge contestation: %d/%d episodes contested (threshold %.0f%%) → contested=%s",
        contested_count,
        len(mids),
        CONTEST_FRACTION_THRESHOLD * 100,
        any_contested,
    )
    return sum(mids) / len(mids), any_contested, outcomes


def _silver_text_from_record(rec: Dict[str, Any]) -> str:
    """Extract the silver summary string from one predictions.jsonl record.

    Handles both dict-shaped ``output`` (with ``summary_final`` /
    ``summary_long`` keys) and plain-string ``output`` shapes. Returns an
    empty string when neither shape yields text — the caller filters those.
    """
    out = rec.get("output") or {}
    if isinstance(out, str):
        return out
    if isinstance(out, dict):
        return out.get("summary_final") or out.get("summary_long") or ""
    return ""


def _load_silver_summaries(silver_reference_path: Path) -> Dict[str, str]:
    """Load per-episode silver summary strings from ``predictions.jsonl``.

    The pairwise judge compares each candidate summary against the same
    silver reference the scalar judge uses via ROUGE-L. Extracting the
    prose here mirrors :func:`_extract_summary_prose` so the judge sees
    the same shape it would for the candidate.
    """
    predictions = silver_reference_path / "predictions.jsonl"
    if not predictions.is_file():
        raise FileNotFoundError(
            f"Pairwise scoring requires a silver predictions.jsonl at "
            f"{predictions}; add the silver reference or run scalar mode."
        )
    by_id: Dict[str, str] = {}
    for line in predictions.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        eid = rec.get("episode_id")
        text = _silver_text_from_record(rec)
        if eid and text:
            by_id[str(eid)] = _extract_summary_prose(text)
    return by_id


def _expand_env_in_url(raw: Optional[str]) -> Optional[str]:
    """Expand ``${VAR}`` placeholders in a URL against ``os.environ``.

    Judge configs write ``api_base: http://${DGX_TAILNET_FQDN}:8004/v1``
    so the hostname isn't hardcoded across dev / CI / homelab. Returns
    None for None input; passes plain URLs through unchanged.
    """
    if raw is None:
        return None
    return os.path.expandvars(str(raw))


def _call_pairwise_judge(
    provider: str,
    *,
    model: str,
    user_content: str,
    candidate_slot: "CandidateSlot",  # type: ignore[name-defined]  # noqa: F821
    api_base: Optional[str] = None,
) -> "PairwiseVerdict":  # type: ignore[name-defined]  # noqa: F821
    """Dispatch one pairwise judge call by provider name.

    Only ``ollama`` and ``vllm`` are supported — DGX judging is the
    intended surface for pairwise. ``openai`` / ``anthropic`` pairwise
    aren't implemented (the DGX judge_config never uses them); if you
    need them, add ``call_openai_judge_raw`` + ``call_anthropic_judge_raw``
    sibling functions and route here.

    ``api_base`` (optional): override the transport's base URL. Used by
    the multi-phase sweep to route each phase's judge to its own
    vLLM port (judge-a on :8004, judge-b on :8005). When None the
    transport falls back to its env-var precedence.
    """
    from podcast_scraper.evaluation.pairwise import parse_pairwise_verdict

    if provider == "ollama":
        from podcast_scraper.evaluation.judges.ollama_chat import OllamaChatJudge

        text = OllamaChatJudge(model=model, api_base=api_base).raw(user_content=user_content)
        return parse_pairwise_verdict(text, candidate_slot=candidate_slot)
    if provider == "vllm":
        from podcast_scraper.evaluation.judges.vllm_chat import VllmChatJudge

        text = VllmChatJudge(model=model, api_base=api_base).raw(user_content=user_content)
        return parse_pairwise_verdict(text, candidate_slot=candidate_slot)
    raise NotImplementedError(
        f"Pairwise dispatch not implemented for provider={provider!r}. "
        "Currently supported: ollama, vllm. Route through a scalar judge or "
        "add raw()-returning openai/anthropic callers."
    )


def judge_one_episode_pairwise(
    *,
    transcript: str,
    candidate_summary: str,
    silver_summary: str,
    episode_id: str,
    judge_a_provider: str,
    judge_a_model: str,
    judge_a_api_base: Optional[str] = None,
    judge_b_provider: Optional[str] = None,
    judge_b_model: Optional[str] = None,
    judge_b_api_base: Optional[str] = None,
) -> "PairwiseOutcome":  # type: ignore[name-defined]  # noqa: F821
    """Score one (candidate, silver) pair through both pairwise judges.

    Position-randomizes the candidate into slot A or B once via
    :func:`prepare_slots`, then routes the SAME slotted message through
    both judges — no double-shuffling. Contest fires only on directional
    disagreement (see :func:`podcast_scraper.evaluation.pairwise.is_contested`).

    Single-judge mode: when ``judge_b_provider`` (and model) are None,
    only judge_a is called. The outcome mirrors judge_a's verdict into
    judge_b for shape compatibility with dual-judge callers; contest
    never fires (a judge trivially agrees with itself). Used by the
    ``judging a`` / ``judging b`` phases in the multi-phase sweep where
    each phase runs one vLLM as the sole judge.
    """
    from podcast_scraper.evaluation.pairwise import (
        build_pairwise_user_message,
        is_contested,
        PAIRWISE_RUBRIC,
        PairwiseOutcome,
        prepare_slots,
    )

    slot, slot_a, slot_b = prepare_slots(
        episode_id=episode_id,
        candidate_summary=candidate_summary,
        silver_summary=silver_summary,
    )
    user_msg = build_pairwise_user_message(
        rubric=PAIRWISE_RUBRIC,
        transcript=transcript,
        slot_a_summary=slot_a,
        slot_b_summary=slot_b,
    )
    v_a = _call_pairwise_judge(
        judge_a_provider,
        model=judge_a_model,
        user_content=user_msg,
        candidate_slot=slot,
        api_base=judge_a_api_base,
    )
    if judge_b_provider is None or judge_b_model is None:
        # Single-judge phase — mirror v_a into judge_b for shape compat.
        return PairwiseOutcome(judge_a=v_a, judge_b=v_a, contested=False)
    v_b = _call_pairwise_judge(
        judge_b_provider,
        model=judge_b_model,
        user_content=user_msg,
        candidate_slot=slot,
        api_base=judge_b_api_base,
    )
    return PairwiseOutcome(judge_a=v_a, judge_b=v_b, contested=is_contested(v_a, v_b))


def mean_pairwise_scores(
    *,
    predictions: List[Dict[str, Any]],
    judge_cfg: Dict[str, Any],
    dataset_id: str,
    eval_root: Path,
    silver_reference_path: Path,
) -> Tuple[float, bool, Dict[str, Any]]:
    """Pairwise counterpart to :func:`mean_judge_scores`.

    Returns:
        mean_score: candidate quality proxy in [0, 1] — mean of
            :func:`pairwise_verdict_to_score` across both judges and all
            episodes. Aggregating both judges' scores here (rather than
            reporting them separately) mirrors ``mean_judge_scores``'s
            midpoint aggregation so the scoring formula
            ``rouge_weight * rougeL + (1 - rouge_weight) * mean_score``
            keeps its shape.
        any_contested: contested if a meaningful fraction of episodes had
            directional judge disagreement (same
            ``CONTEST_FRACTION_THRESHOLD`` as scalar).
        summary: dict with per-judge win_rate / tie_rate / decisive_rate
            for the leaderboard audit column.
    """
    from podcast_scraper.evaluation.pairwise import (
        pairwise_verdict_to_score,
        summarize_pairwise_run,
    )

    ja = judge_cfg.get("judge_a") or {}
    jb = judge_cfg.get("judge_b") or {}
    j_a_prov = str(ja.get("provider", "ollama"))
    j_a_model = str(ja["model"])
    j_a_api_base = _expand_env_in_url(ja.get("api_base"))
    # Single-judge phase support — when judge_b is absent from the config,
    # skip its call. Used by the per-phase vLLM judges in the multi-phase
    # sweep (``judging a`` and ``judging b`` each run one vLLM as sole judge).
    if jb:
        j_b_prov: Optional[str] = str(jb.get("provider", "ollama"))
        j_b_model: Optional[str] = str(jb["model"])
        j_b_api_base: Optional[str] = _expand_env_in_url(jb.get("api_base"))
    else:
        j_b_prov = None
        j_b_model = None
        j_b_api_base = None

    eids = [str(p.get("episode_id")) for p in predictions if p.get("episode_id")]
    transcripts = transcripts_by_episode_id(
        dataset_id=dataset_id, episode_ids=eids, eval_root=eval_root
    )
    silvers = _load_silver_summaries(silver_reference_path)

    judge_a_verdicts = []
    judge_b_verdicts = []
    per_episode_scores: List[float] = []
    contested_count = 0

    for pred in predictions:
        eid = pred.get("episode_id")
        if not eid:
            continue
        candidate = summary_text_from_prediction(pred)
        if not candidate.strip():
            logger.warning("Skipping pairwise judge for episode %s: empty summary", eid)
            continue
        silver = silvers.get(str(eid), "")
        if not silver.strip():
            logger.warning("Skipping pairwise judge for episode %s: no silver summary", eid)
            continue
        tr = transcripts.get(str(eid), "")

        outcome = judge_one_episode_pairwise(
            transcript=tr,
            candidate_summary=candidate,
            silver_summary=silver,
            episode_id=str(eid),
            judge_a_provider=j_a_prov,
            judge_a_model=j_a_model,
            judge_a_api_base=j_a_api_base,
            judge_b_provider=j_b_prov,
            judge_b_model=j_b_model,
            judge_b_api_base=j_b_api_base,
        )
        judge_a_verdicts.append(outcome.judge_a)
        judge_b_verdicts.append(outcome.judge_b)
        per_episode_scores.append(
            (
                pairwise_verdict_to_score(outcome.judge_a)
                + pairwise_verdict_to_score(outcome.judge_b)
            )
            / 2.0
        )
        if outcome.contested:
            contested_count += 1

    if not per_episode_scores:
        raise RuntimeError("No episodes scored by pairwise judges")

    any_contested = (contested_count / len(per_episode_scores)) > CONTEST_FRACTION_THRESHOLD
    logger.info(
        "Pairwise judge contestation: %d/%d episodes directionally contested "
        "(threshold %.0f%%) → contested=%s",
        contested_count,
        len(per_episode_scores),
        CONTEST_FRACTION_THRESHOLD * 100,
        any_contested,
    )

    summary = {
        "judge_a": summarize_pairwise_run(judge_a_verdicts),
        "judge_b": summarize_pairwise_run(judge_b_verdicts),
        "contested_count": contested_count,
        "episodes": len(per_episode_scores),
    }
    return sum(per_episode_scores) / len(per_episode_scores), any_contested, summary


def combine_track_a_scalar(
    *,
    rouge_l_f1: Optional[float],
    judge_mean: Optional[float],
    contested: bool,
    rouge_weight: float,
) -> float:
    """Single scalar for autoresearch ratchet (higher is better)."""
    r = 0.0 if rouge_l_f1 is None else float(rouge_l_f1)
    if contested or judge_mean is None:
        return r
    jw = 1.0 - rouge_weight
    return rouge_weight * r + jw * float(judge_mean)


def merge_max_episodes_into_config_yaml(source: Path, dest: Path, max_episodes: int) -> None:
    """Write experiment YAML copy with ``data.max_episodes`` set."""
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid experiment yaml: {source}")
    data = raw.get("data")
    if not isinstance(data, dict):
        raise ValueError("Experiment yaml missing data: mapping")
    data["max_episodes"] = max_episodes
    raw["data"] = data
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.dump(raw, default_flow_style=False, sort_keys=False), encoding="utf-8")
