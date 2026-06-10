"""G-Eval core for the autoresearch finale tier (#932).

We score a candidate summary against a transcript on four dimensions
(faithfulness / coverage / coherence / fluency), each in the integer range
``[1, 5]``. The structure follows the G-Eval methodology (Liu et al. 2023):

1. Each dimension has its own rubric prompt with clear anchors (1=fail, 5=excellent).
2. The judge replies in a strict JSON schema we can parse deterministically.
3. We score one dimension per call (parallelizable, isolates one rubric in
   the judge's context window) — this is cheaper *and* more reliable than a
   single mega-call asking for all four at once because:
   - smaller context → less prompt-tax,
   - one rubric in attention → less score-leakage between dimensions,
   - per-call retry on a single transient parse failure.

Public surface:

- :class:`DIMENSIONS`             — the 4 dimension names
- :func:`build_dimension_prompt`  — render the user prompt for a (dim, transcript, summary)
- :func:`parse_dimension_response` — parse the judge's JSON reply → ``DimensionScore``
- :func:`score_summary`           — orchestrate all 4 dimensions through a judge
- :class:`SummaryScore`           — per-summary aggregate
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from podcast_scraper.evaluation.judges.base import JudgeResult, JudgeUnavailableError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dimension catalogue


DIMENSIONS: Tuple[str, ...] = ("faithfulness", "coverage", "coherence", "fluency")

# Per-dimension rubric prompts. Each prompt is *exactly* what the judge sees
# beyond the shared header / transcript / summary frame in
# ``build_dimension_prompt``. Anchors are integers 1-5 with descriptors aligned
# to the G-Eval paper's tone (specific, behavior-grounded, not vibes-based).
_RUBRICS: Dict[str, str] = {
    "faithfulness": (
        "Score the summary's FAITHFULNESS to the transcript.\n"
        "A faithful summary makes no claims that contradict the transcript and\n"
        "introduces no facts, names, numbers, or quotes that the transcript does\n"
        "not support.\n\n"
        "Anchors:\n"
        "  5 — Every factual claim is grounded in the transcript. No fabricated\n"
        "      names, numbers, or quotes. No misattributed statements.\n"
        "  4 — Mostly faithful; at most one minor ambiguity (e.g. a paraphrase\n"
        "      that drifts slightly from the source but does not change the meaning).\n"
        "  3 — One clear minor hallucination (a number, name, or claim not in the\n"
        "      transcript) OR multiple ambiguities.\n"
        "  2 — A major hallucination that changes meaning (a fabricated argument,\n"
        "      a misattributed quote, or several minor hallucinations).\n"
        "  1 — Multiple major hallucinations OR systematic misrepresentation of\n"
        "      what the speakers said."
    ),
    "coverage": (
        "Score the summary's COVERAGE of the transcript.\n"
        "Good coverage captures the central thread: key decisions, arguments,\n"
        "lessons, and concrete examples that drive the episode. It does NOT\n"
        "mean repeating every detail — selectivity is good — but the substantive\n"
        "spine must be there.\n\n"
        "Anchors:\n"
        "  5 — Captures all major themes, key decisions/arguments, and the\n"
        "      concrete examples that anchor them. A reader of the summary\n"
        "      would know what the episode is about and what to take away.\n"
        "  4 — Captures most major themes; misses one secondary thread or one\n"
        "      concrete example that would have strengthened a point.\n"
        "  3 — Captures roughly half the substantive material; misses one\n"
        "      central decision/argument or several supporting examples.\n"
        "  2 — Misses multiple central threads; reads as a partial summary of\n"
        "      one section rather than the whole episode.\n"
        "  1 — Captures almost none of the substantive material; reads like a\n"
        "      surface-level paraphrase of the title and intro."
    ),
    "coherence": (
        "Score the summary's COHERENCE as a piece of writing.\n"
        "A coherent summary flows as a single argument — sentences connect to\n"
        "one another, paragraphs have a clear logical role, transitions feel\n"
        "intentional, and the reader is never lost about why one idea follows\n"
        "the next.\n\n"
        "Anchors:\n"
        "  5 — Reads as a single, well-structured piece. Clear paragraph roles,\n"
        "      smooth transitions, ideas build on each other.\n"
        "  4 — Generally coherent; one abrupt transition or one paragraph that\n"
        "      could be tightened.\n"
        "  3 — Several abrupt transitions OR a paragraph that doesn't connect\n"
        "      clearly to its neighbors. Reads as 'a list of true statements'\n"
        "      rather than a unified piece.\n"
        "  2 — Frequent jumps and structural problems; reader has to work hard\n"
        "      to assemble the throughline.\n"
        "  1 — Effectively incoherent; sentences/paragraphs feel randomly ordered."
    ),
    "fluency": (
        "Score the summary's FLUENCY at the sentence/grammar level.\n"
        "A fluent summary uses correct grammar, idiomatic English, and natural\n"
        "phrasing. Awkward translations, run-on sentences, redundant phrases,\n"
        "and word-choice errors hurt fluency.\n\n"
        "Anchors:\n"
        "  5 — Clean prose; reads as if a careful editor passed over it. No\n"
        "      grammar mistakes; no awkward phrasings.\n"
        "  4 — Mostly clean; one minor grammar slip or one slightly awkward\n"
        "      sentence.\n"
        "  3 — Several minor grammar/style issues OR one major awkwardness that\n"
        "      forces a re-read.\n"
        "  2 — Frequent grammar errors or awkward constructions; readable but\n"
        "      consistently bumpy.\n"
        "  1 — Pervasive grammar/style problems; reads as machine-translated\n"
        "      or unedited draft."
    ),
}

# Soft cap on transcript size sent to the judge.  Mirrors track_a.py's
# ``MAX_TRANSCRIPT_CHARS`` so the same evidence window is presented across
# qualifier and finale tiers (apples-to-apples).
MAX_TRANSCRIPT_CHARS = 28_000


# ---------------------------------------------------------------------------
# Prompt construction


def build_dimension_prompt(
    *,
    dimension: str,
    transcript: str,
    summary: str,
    max_transcript_chars: int = MAX_TRANSCRIPT_CHARS,
) -> str:
    """Render the user prompt sent to a judge for one (dimension, summary) pair.

    Args:
        dimension: One of :data:`DIMENSIONS`. Other values raise ``ValueError``.
        transcript: Episode transcript text — truncated to ``max_transcript_chars``.
        summary: The candidate summary to score.
        max_transcript_chars: Soft truncation cap (default 28k).

    Returns:
        A single string the judge consumes as a user message.

    Raises:
        ValueError: ``dimension`` is not a member of :data:`DIMENSIONS`.
    """
    if dimension not in _RUBRICS:
        raise ValueError(f"Unknown G-Eval dimension {dimension!r}; expected one of {DIMENSIONS}")

    t = transcript if len(transcript) <= max_transcript_chars else transcript[:max_transcript_chars]
    rubric = _RUBRICS[dimension]
    return (
        "You are an expert evaluator of podcast episode summaries.\n"
        "You will be given:\n"
        "  1. The episode transcript (the ground truth).\n"
        "  2. A candidate summary written by some model.\n"
        "  3. A scoring rubric for ONE specific dimension.\n\n"
        "Score the summary on the rubric below, then reply with a single JSON\n"
        "object and nothing else.\n\n"
        "### Rubric\n"
        f"{rubric}\n\n"
        "### Transcript (may be truncated)\n"
        f"{t}\n\n"
        "### Candidate summary\n"
        f"{summary}\n\n"
        "### Output format\n"
        f'Reply with a single JSON object: {{"dimension": "{dimension}", '
        '"score": <integer 1-5>, "explanation": "<one or two sentences>"}}\n'
        "Do not include markdown code fences or any text outside the JSON."
    )


# ---------------------------------------------------------------------------
# Response parsing


@dataclass(frozen=True)
class DimensionScore:
    """One judge's score for one (summary, dimension) pair."""

    dimension: str
    score: int  # 1..5
    explanation: str
    judge_model: str = ""
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9]*\s*|\s*```$", re.MULTILINE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_dimension_response(
    text: str,
    *,
    expected_dimension: str,
    judge_model: str = "",
) -> DimensionScore:
    """Parse the judge's reply into a :class:`DimensionScore`.

    Strict-ish: strips code fences and locates the first ``{...}`` object via
    regex (robust to judges that occasionally prepend a stray sentence
    despite the format instructions).

    Args:
        text: Raw assistant text from a :class:`JudgeResult`.
        expected_dimension: Sanity check: judge's ``dimension`` field must match.
        judge_model: Recorded on the returned ``DimensionScore`` for traceability.

    Raises:
        ValueError: parse failure, missing keys, score outside [1, 5], or
            dimension mismatch.
    """
    cleaned = _CODE_FENCE_RE.sub("", text or "").strip()
    if not cleaned:
        raise ValueError("Empty judge response")
    # If the judge prepended commentary, locate the first JSON object.
    if not cleaned.startswith("{"):
        m = _JSON_OBJECT_RE.search(cleaned)
        if not m:
            raise ValueError(f"No JSON object found in judge response: {cleaned[:120]!r}")
        cleaned = m.group(0)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge response is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Judge JSON must be an object")

    dim = data.get("dimension")
    if dim != expected_dimension:
        raise ValueError(f"Judge returned dimension={dim!r} but expected {expected_dimension!r}")
    raw_score = data.get("score")
    if raw_score is None:
        raise ValueError("Judge JSON missing 'score'")
    try:
        score = int(raw_score)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"score is not an integer: {raw_score!r}") from exc
    if not 1 <= score <= 5:
        raise ValueError(f"score out of range [1, 5]: {score}")
    explanation = str(data.get("explanation") or "").strip()
    return DimensionScore(
        dimension=expected_dimension,
        score=score,
        explanation=explanation,
        judge_model=judge_model,
    )


# ---------------------------------------------------------------------------
# Aggregation


@dataclass
class SummaryScore:
    """Aggregate G-Eval result for one candidate summary from one judge.

    ``per_dimension`` maps each dimension name to its :class:`DimensionScore`.
    ``mean`` is the unweighted average of the four 1-5 integer scores (so
    range is also [1, 5]).
    ``errors`` records dimensions where the judge call failed entirely; those
    dimensions are excluded from ``mean``.
    """

    run_id: str
    episode_id: str
    judge_model: str
    per_dimension: Dict[str, DimensionScore] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    @property
    def mean(self) -> Optional[float]:
        vals = [d.score for d in self.per_dimension.values()]
        if not vals:
            return None
        return sum(vals) / len(vals)

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSONL-ready dict for one (run, episode, judge) row."""
        return {
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "judge_model": self.judge_model,
            "per_dimension": {
                name: {
                    "score": d.score,
                    "explanation": d.explanation,
                    "cost_usd": d.cost_usd,
                    "prompt_tokens": d.prompt_tokens,
                    "completion_tokens": d.completion_tokens,
                }
                for name, d in self.per_dimension.items()
            },
            "errors": dict(self.errors),
            "mean": self.mean,
            "total_cost_usd": self.total_cost_usd,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
        }


# A callable that takes a prompt string and returns a JudgeResult — i.e. the
# ``.score`` method of any judge in ``judges/``. Typed as a Protocol-like alias
# to keep ``score_summary`` decoupled from the concrete judge classes.
JudgeCallable = Callable[..., JudgeResult]


def score_summary(
    *,
    run_id: str,
    episode_id: str,
    transcript: str,
    summary: str,
    judge: Any,
    dimensions: Tuple[str, ...] = DIMENSIONS,
    max_tokens_per_call: int = 512,
) -> SummaryScore:
    """Score one summary on all four dimensions through one judge.

    The judge object must expose ``score(prompt, max_tokens=...)`` returning a
    :class:`JudgeResult`, and a ``model`` attribute (matches the judges in
    :mod:`podcast_scraper.evaluation.judges`).

    A per-dimension parse / transport failure is recorded in ``errors`` but
    does not abort the remaining dimensions.

    Args:
        run_id: Identifier of the source run dir (for traceability).
        episode_id: Episode id from ``predictions.jsonl``.
        transcript: Episode transcript text.
        summary: Candidate summary text.
        judge: A judge instance from :mod:`podcast_scraper.evaluation.judges`.
        dimensions: Which dimensions to score (default: all four).
        max_tokens_per_call: Output cap per dimension (small — replies are short).
    """
    judge_model = getattr(judge, "model", "unknown")
    result = SummaryScore(run_id=run_id, episode_id=episode_id, judge_model=judge_model)

    for dim in dimensions:
        prompt = build_dimension_prompt(dimension=dim, transcript=transcript, summary=summary)
        try:
            jr = judge.score(prompt, max_tokens=max_tokens_per_call)
        except JudgeUnavailableError as exc:
            logger.warning(
                "Judge %s failed on %s/%s/%s: %s", judge_model, run_id, episode_id, dim, exc
            )
            result.errors[dim] = f"transport: {exc}"
            continue
        try:
            dim_score = parse_dimension_response(
                jr.text, expected_dimension=dim, judge_model=judge_model
            )
        except ValueError as exc:
            logger.warning(
                "Judge %s parse fail on %s/%s/%s: %s — text=%r",
                judge_model,
                run_id,
                episode_id,
                dim,
                exc,
                (jr.text or "")[:160],
            )
            result.errors[dim] = f"parse: {exc}"
            continue
        # Attach cost / token bookkeeping
        dim_score = DimensionScore(
            dimension=dim_score.dimension,
            score=dim_score.score,
            explanation=dim_score.explanation,
            judge_model=judge_model,
            cost_usd=jr.cost_usd,
            prompt_tokens=jr.prompt_tokens,
            completion_tokens=jr.completion_tokens,
        )
        result.per_dimension[dim] = dim_score
        result.total_cost_usd += jr.cost_usd
        result.total_prompt_tokens += jr.prompt_tokens
        result.total_completion_tokens += jr.completion_tokens
    return result


# ---------------------------------------------------------------------------
# Two-judge agreement helpers (for #940 Track 1 + #932 cross-check)


def agreement_rate(
    a_scores: List[DimensionScore],
    b_scores: List[DimensionScore],
    *,
    tolerance: int = 1,
) -> Tuple[float, int, int]:
    """Compute the agreement rate between two judges' per-dimension scores.

    Two scores are considered to agree iff ``|a - b| <= tolerance``. The
    default tolerance of 1 follows the G-Eval paper's "exact-or-adjacent"
    convention on a 1-5 scale — accounting for the inherent noise of judges
    sometimes flipping between adjacent anchors.

    Args:
        a_scores: Sequence of :class:`DimensionScore` from judge A.
        b_scores: Same shape from judge B. Order must match ``a_scores`` —
            zip alignment, not key lookup.
        tolerance: Max abs-diff for two scores to count as agreement.

    Returns:
        ``(rate, agreements, total)`` — ``rate`` is in ``[0.0, 1.0]``; when
        ``total == 0`` the rate is ``0.0`` (caller treats as "no signal").
    """
    if len(a_scores) != len(b_scores):
        raise ValueError(
            f"a_scores and b_scores must align; got {len(a_scores)} vs {len(b_scores)}"
        )
    if not a_scores:
        return (0.0, 0, 0)
    agree = 0
    for a, b in zip(a_scores, b_scores):
        if a.dimension != b.dimension:
            raise ValueError(f"Score alignment broken: {a.dimension!r} vs {b.dimension!r}")
        if abs(a.score - b.score) <= tolerance:
            agree += 1
    return (agree / len(a_scores), agree, len(a_scores))
