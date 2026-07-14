"""The LLM proposes a speaker; the episode text has to corroborate it.

An LLM speaker detector returns a name list and a success flag, and nothing checks either. On Hard
Fork's "OpenAI's Big Reset", qwen3.5:35b returned::

    speakers:  ['Casey Newton', 'Kevin Roose', 'Elon Musk', 'Sam Altman',
                'Dr. Adam Rodman', 'David Duvenaud']
    succeeded: True

Elon Musk appears in that description only as the man SUING OpenAI; Sam Altman only as the man who
runs it. Neither says a word. The names were handed to the diarizer, assigned positionally to voice
clusters, and every insight the actual guest spoke was published under Elon Musk's name. The prompt
forbids exactly this and the model ignored it — a prompt is not an enforcement mechanism.

So the model's answer is a *proposal*, and it is checked against the same text the model was shown:
a guest must carry an interview cue next to their name (``_is_likely_actual_guest``, the same
deterministic matcher the NER detector uses). Hosts are exempt — they come from the feed / config /
host cache, and are corroborated by their own path, not by an episode-level interview cue.

Dropping a real guest costs an unnamed voice cluster (``SPEAKER_01``). Keeping a fake one puts words
in a named person's mouth. Those are not symmetric, and this gate is tuned accordingly (#876).

The NER detector already filters its guests through the same predicate, so this is a no-op there —
it is idempotent by construction, and one gate covers every provider.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Set

from .guests import _is_likely_actual_guest

logger = logging.getLogger(__name__)


def corroborate_guests(
    guests: Iterable[str],
    episode_title: str,
    episode_description: Optional[str],
    known_hosts: Optional[Set[str]] = None,
) -> List[str]:
    """Keep only the proposed guests the episode text actually introduces as speakers.

    Args:
        guests: Names a detector proposed as guests (LLM or NER).
        episode_title: The episode title.
        episode_description: The episode description — the evidence. This is the *same* text the
            LLM was shown, so corroborating against it is a fair test of the model's claim.
        known_hosts: Hosts, which are never subject to the interview-cue test.

    Returns:
        The corroborated guests, order preserved.
    """
    hosts_lower = {h.lower().strip() for h in (known_hosts or set())}

    kept: List[str] = []
    rejected: List[str] = []
    for name in guests:
        clean = (name or "").strip()
        if not clean:
            continue
        if clean.lower() in hosts_lower:
            continue
        if _is_likely_actual_guest(clean, episode_title, episode_description):
            kept.append(clean)
        else:
            rejected.append(clean)

    if rejected:
        logger.warning(
            "  → Rejected %d uncorroborated speaker(s): %s. The episode text names them but never "
            "introduces them as speaking. A wrong name on a voice is worse than no name.",
            len(rejected),
            ", ".join(rejected),
        )
    return kept
