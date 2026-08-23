"""Size an ingestion window by AUDIO MINUTES, not episode count (#1658).

``cost_soft_cap_usd_per_run`` is checked against MODELLED cost, which is dominated by
transcription minutes. So a flat episode count means completely different things per feed:

    The a16z Show            49 min median  ->  ~28 episodes per safe window
    Dwarkesh / Pragmatic Eng 85-87 min      ->  ~16
    Lenny's / Ideas of India 92-93 min      ->  ~15
    Latent Space             ~75 min        ->  ~18   (measured: 15 eps = $5.64)

This is not a tuning nicety. Tripping the cap is what caused **both** the G1 silent wedge and
the G2 executor crash (#1620) — long-form feeds blew the cap on a window size that was
perfectly safe for short-form ones, while short-form feeds ran needlessly small jobs.

Measured context: over a window where the corpus went 434 -> 519 (+85 episodes), modelled cost
rose **+$22.21** ($0.26/episode) while real LiteLLM spend rose **+$2.02** ($0.024/episode).
The ~11x gap is Deepgram on a free allowance — but the cap is enforced on the MODELLED number,
so the modelled number is what governs batch sizing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

#: Target audio-minutes per job. ~$7 modelled, leaving margin under the $10 cap.
DEFAULT_TARGET_AUDIO_MINUTES = 1400

#: Never emit a window below this — a 1-episode job pays the full-corpus finalize pass for
#: almost no work (G-opt finding 2), so tiny windows are the expensive way to be careful.
MIN_WINDOW_EPISODES = 5

#: Never emit a window above this regardless of how short the episodes are: an enormous job
#: is a long uninterruptible unit of work, and the supervision bounds (#1620) are per-job.
MAX_WINDOW_EPISODES = 50


@dataclass(frozen=True)
class WindowPlan:
    """A recommended window, carrying the arithmetic that produced it.

    The inputs travel with the answer on purpose: a bare number invites someone to "round it
    up a bit", which is precisely how a long-form feed trips the cap.
    """

    episodes: int
    median_episode_minutes: float
    target_audio_minutes: int
    projected_audio_minutes: float
    clamped: Optional[str] = None

    def explain(self) -> str:
        """One line an operator can paste into a run log."""
        base = (
            f"{self.episodes} episodes "
            f"(median {self.median_episode_minutes:.0f} min -> "
            f"~{self.projected_audio_minutes:.0f} audio-minutes, "
            f"target {self.target_audio_minutes})"
        )
        return f"{base} [clamped: {self.clamped}]" if self.clamped else base


def plan_window(
    median_episode_minutes: float,
    *,
    target_audio_minutes: int = DEFAULT_TARGET_AUDIO_MINUTES,
    min_episodes: int = MIN_WINDOW_EPISODES,
    max_episodes: int = MAX_WINDOW_EPISODES,
) -> WindowPlan:
    """Recommend a window size for a feed with the given median episode length.

    ``window = target_audio_minutes / median_minutes``, then clamped. A non-positive or
    unknown median falls back to the minimum window rather than guessing large — being wrong
    small costs an extra job, being wrong large trips the cap and wedges the run.
    """
    if median_episode_minutes <= 0:
        return WindowPlan(
            episodes=min_episodes,
            median_episode_minutes=median_episode_minutes,
            target_audio_minutes=target_audio_minutes,
            projected_audio_minutes=0.0,
            clamped="unknown_median_using_minimum",
        )

    raw = int(target_audio_minutes // median_episode_minutes)
    clamped: Optional[str] = None
    episodes = raw
    if episodes < min_episodes:
        episodes, clamped = min_episodes, "below_minimum"
    elif episodes > max_episodes:
        episodes, clamped = max_episodes, "above_maximum"

    return WindowPlan(
        episodes=episodes,
        median_episode_minutes=median_episode_minutes,
        target_audio_minutes=target_audio_minutes,
        projected_audio_minutes=episodes * median_episode_minutes,
        clamped=clamped,
    )
