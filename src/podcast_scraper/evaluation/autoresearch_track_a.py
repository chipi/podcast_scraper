"""Track A autoresearch helpers: env keys, ROUGE extraction, dual LLM judges.

Logic here is imported by ``autoresearch/prompt_tuning/eval/score.py`` and unit-tested
without running the full experiment subprocess.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from podcast_scraper.utils.log_redaction import format_exception_for_log

logger = logging.getLogger(__name__)


def _is_test_environment() -> bool:
    if "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ:
        return True
    if "unittest" in sys.modules:
        return True
    if os.environ.get("TESTING", "").lower() in ("1", "true", "yes"):
        return True
    return False


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
    if _is_test_environment():
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
    """Score one summary with OpenAI + Anthropic judges; mark contested if scores diverge."""
    user_msg = _judge_user_message(rubric=rubric, transcript=transcript, summary=summary)
    if judge_a_provider != "openai":
        raise ValueError(f"judge_a unsupported provider: {judge_a_provider}")
    if judge_b_provider != "anthropic":
        raise ValueError(f"judge_b unsupported provider: {judge_b_provider}")
    s_a = call_openai_judge(api_key=openai_key, model=judge_a_model, user_content=user_msg)
    s_b = call_anthropic_judge(api_key=anthropic_key, model=judge_b_model, user_content=user_msg)
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
