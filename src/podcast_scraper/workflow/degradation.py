"""Graceful degradation policy for pipeline failures.

This module defines and implements degradation policies for handling
component failures gracefully, allowing the pipeline to continue
processing when non-critical stages fail.
"""

import logging
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DegradationPolicy(BaseModel):
    """Configuration model for graceful degradation.

    This policy defines how the pipeline should behave when
    various stages fail. Default behavior is to save partial results
    and continue processing.
    """

    save_transcript_on_summarization_failure: bool = Field(
        default=True,
        description=(
            "If True, save transcript even if summarization fails. "
            "Default: True (save partial results)."
        ),
    )
    save_summary_on_entity_extraction_failure: bool = Field(
        default=True,
        description=(
            "If True, save summary even if entity extraction fails. "
            "Default: True (save partial results)."
        ),
    )
    fallback_provider_on_failure: Optional[str] = Field(
        default=None,
        description=(
            "Optional fallback provider name to use when primary provider fails. "
            "Example: 'transformers' to fallback to local ML provider. "
            "Default: None (no fallback)."
        ),
    )
    continue_on_stage_failure: bool = Field(
        default=True,
        description=(
            "If True, continue processing remaining episodes when a stage fails. "
            "If False, fail fast on first stage failure. "
            "Default: True (continue processing)."
        ),
    )


def handle_stage_failure(
    stage: Literal["summarization", "entity_extraction", "transcription", "metadata"],
    error: Exception,
    policy: DegradationPolicy,
    episode_idx: Optional[int] = None,
    context: Optional[dict[str, Any]] = None,
) -> bool:
    """Handle a stage failure according to degradation policy.

    Args:
        stage: Stage that failed (summarization, entity_extraction, etc.)
        error: Exception that caused the failure
        policy: Degradation policy configuration
        episode_idx: Optional episode index for logging
        context: Optional context dictionary with additional info

    Returns:
        True if processing should continue, False if should fail fast
    """
    episode_prefix = f"[{episode_idx}] " if episode_idx is not None else ""
    error_msg = str(error)

    if stage == "summarization":
        if policy.save_transcript_on_summarization_failure:
            logger.warning(
                f"{episode_prefix}Summarization failed: {error_msg}. "
                "Saving transcript without summary "
                "(degradation policy: save_transcript_on_summarization_failure=True)."
            )
            return policy.continue_on_stage_failure
        else:
            logger.error(
                f"{episode_prefix}Summarization failed: {error_msg}. "
                "Not saving transcript "
                "(degradation policy: save_transcript_on_summarization_failure=False)."
            )
            return False

    elif stage == "entity_extraction":
        if policy.save_summary_on_entity_extraction_failure:
            logger.warning(
                f"{episode_prefix}Entity extraction failed: {error_msg}. "
                "Saving summary without entities "
                "(degradation policy: save_summary_on_entity_extraction_failure=True)."
            )
            return policy.continue_on_stage_failure
        else:
            logger.error(
                f"{episode_prefix}Entity extraction failed: {error_msg}. "
                "Not saving summary "
                "(degradation policy: save_summary_on_entity_extraction_failure=False)."
            )
            return False

    elif stage == "transcription":
        # Transcription failures are generally fatal (no transcript = no processing)
        logger.error(
            f"{episode_prefix}Transcription failed: {error_msg}. "
            "Cannot continue without transcript."
        )
        return False

    elif stage == "metadata":
        # Metadata failures are generally non-fatal (transcript still saved)
        logger.warning(
            f"{episode_prefix}Metadata generation failed: {error_msg}. "
            "Transcript saved, but metadata not generated."
        )
        return policy.continue_on_stage_failure

    else:
        # Unknown stage - default to continue if policy allows
        logger.warning(
            f"{episode_prefix}Unknown stage '{stage}' failed: {error_msg}. "
            f"Continuing based on policy: "
            f"continue_on_stage_failure={policy.continue_on_stage_failure}."
        )
        return policy.continue_on_stage_failure
