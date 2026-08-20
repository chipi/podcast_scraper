"""Custom exceptions for podcast_scraper providers.

This module defines structured exceptions for provider initialization and runtime
errors. Using typed exceptions improves:
- Error messages with actionable suggestions
- Test assertions on specific failure causes
- Operator debugging and UX

Exception Hierarchy:
    ProviderError (base)
    ├── ProviderConfigError - Configuration issues
    ├── ProviderDependencyError - Missing dependencies
    ├── ProviderAuthError - Authentication failures
    └── ProviderRuntimeError - Runtime operation failures
"""

from typing import Optional


class ProviderError(Exception):
    """Base exception for all provider-related errors.

    Attributes:
        provider: Name of the provider (e.g., "OpenAI", "MLProvider/Whisper")
        message: Human-readable error message
        suggestion: Optional suggestion for resolving the error
    """

    def __init__(
        self,
        message: str,
        provider: str = "Unknown",
        suggestion: Optional[str] = None,
    ) -> None:
        self.provider = provider
        self.message = message
        self.suggestion = suggestion
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the full error message with provider and suggestion."""
        parts = [f"[{self.provider}] {self.message}"]
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " ".join(parts)


class ProviderConfigError(ProviderError):
    """Raised when provider configuration is invalid or missing.

    Common causes:
    - Missing API keys
    - Invalid model names
    - Incompatible configuration combinations

    Example:
        >>> raise ProviderConfigError(
        ...     message="API key not provided",
        ...     provider="OpenAI",
        ...     config_key="openai_api_key",
        ...     suggestion="Set OPENAI_API_KEY environment variable"
        ... )
    """

    def __init__(
        self,
        message: str,
        provider: str = "Unknown",
        config_key: Optional[str] = None,
        suggestion: Optional[str] = None,
    ) -> None:
        self.config_key = config_key
        if config_key and config_key not in message:
            message = f"{message} (config key: {config_key})"
        super().__init__(message=message, provider=provider, suggestion=suggestion)


class ProviderDependencyError(ProviderError):
    """Raised when required dependencies are missing or unavailable.

    Common causes:
    - Python package not installed (whisper, spacy, transformers)
    - ML model not downloaded or cached
    - System dependencies missing

    Example:
        >>> raise ProviderDependencyError(
        ...     message="spaCy model 'en_core_web_sm' not found",
        ...     provider="MLProvider/spaCy",
        ...     dependency="en_core_web_sm",
        ...     suggestion="Install with: python -m spacy download en_core_web_sm"
        ... )
    """

    def __init__(
        self,
        message: str,
        provider: str = "Unknown",
        dependency: Optional[str] = None,
        suggestion: Optional[str] = None,
    ) -> None:
        self.dependency = dependency
        if dependency and dependency not in message:
            message = f"{message} (dependency: {dependency})"
        super().__init__(message=message, provider=provider, suggestion=suggestion)


class ProviderAuthError(ProviderError):
    """Raised when authentication with a provider fails.

    Common causes:
    - Invalid API key
    - Expired credentials
    - Insufficient permissions

    Example:
        >>> raise ProviderAuthError(
        ...     message="Invalid API key",
        ...     provider="OpenAI",
        ...     suggestion="Check your API key at https://platform.openai.com/api-keys"
        ... )
    """

    def __init__(
        self,
        message: str,
        provider: str = "Unknown",
        suggestion: Optional[str] = None,
    ) -> None:
        super().__init__(message=message, provider=provider, suggestion=suggestion)


class ProviderRuntimeError(ProviderError):
    """Raised when a provider operation fails at runtime.

    Common causes:
    - Network errors
    - API rate limiting
    - Model inference failures
    - Invalid input data

    Example:
        >>> raise ProviderRuntimeError(
        ...     message="Transcription failed: audio file too large",
        ...     provider="OpenAI/Transcription",
        ...     suggestion="Split audio into chunks under 25MB"
        ... )
    """

    def __init__(
        self,
        message: str,
        provider: str = "Unknown",
        suggestion: Optional[str] = None,
    ) -> None:
        super().__init__(message=message, provider=provider, suggestion=suggestion)


class GILGroundingUnsatisfiedError(ProviderRuntimeError):
    """Raised when gi_fail_on_missing_grounding is True and no quotes passed QA+NLI.

    Not caught by the generic GIL retry loop in metadata generation; propagates to fail
    the episode so CI or strict manual runs cannot succeed with empty grounding.
    """


class ProviderNotInitializedError(ProviderError):
    """Raised when a provider method is called before initialization.

    This indicates a programming error where initialize() was not called
    before using provider capabilities.

    Example:
        >>> raise ProviderNotInitializedError(
        ...     provider="MLProvider/Whisper",
        ...     capability="transcription"
        ... )
    """

    def __init__(
        self,
        provider: str = "Unknown",
        capability: Optional[str] = None,
    ) -> None:
        self.capability = capability
        cap_str = f" for {capability}" if capability else ""
        message = f"Provider not initialized{cap_str}. Call initialize() first."
        super().__init__(
            message=message,
            provider=provider,
            suggestion="Call initialize() before using the provider",
        )


class RecoverableSummarizationError(Exception):
    """Raised when summarization fails but processing can continue.

    This exception indicates that summarization failed due to a known
    recoverable issue (e.g., tokenizer threading errors in parallel execution),
    and metadata generation should continue without the summary rather than
    failing the entire episode.

    Attributes:
        episode_idx: Index of the episode that failed summarization
        reason: Human-readable reason, for the log line
        code: Stable machine-readable slug for the stage ledger (#1647) and for grouping in
            reports. Prose drifts between releases and cannot be grouped; the slug is the part
            an operator filters on when asking "how many episodes lost their summary, and to
            what?". Defaults to ``unspecified`` so an un-migrated raise site is VISIBLE in the
            ledger as an unclassified degradation rather than absent from it.
    """

    #: Every way an episode can end up persisted without a summary. Kept together so the set is
    #: enumerable — the audit and the repair work-list both key on it, and a new degradation path
    #: that forgets to add itself here shows up as ``unspecified``.
    SCHEMA_INVALID_AFTER_REROLL = "schema_invalid_after_reroll"
    TOKENIZER_THREADING = "tokenizer_threading"
    PROVIDER_CONTENT_REJECTED = "provider_content_rejected"
    PROMPT_EXAMPLES_LEAKED = "prompt_examples_leaked"
    UNSPECIFIED = "unspecified"

    def __init__(self, episode_idx: int, reason: str, *, code: str = UNSPECIFIED) -> None:
        self.episode_idx = episode_idx
        self.reason = reason
        self.code = code
        message = (
            f"[{episode_idx}] Summarization failed (recoverable): {reason}. "
            "Metadata generation will continue without summary."
        )
        super().__init__(message)


#: Causes worth ONE more attempt before the episode is written without a summary (#1686).
#:
#: Membership is a claim about the CAUSE, not about how much we want the summary. A tokenizer
#: "Already borrowed" is a race between workers on a shared Rust tokenizer — the code's own
#: comment calls it "a known threading issue" that "can occur in parallel execution" — so the
#: same input on a second pass can genuinely succeed.
#:
#: The exclusions matter as much as the members:
#:   PROVIDER_CONTENT_REJECTED — the provider refused THIS input. Same input, same refusal;
#:       a retry only spends money to fail identically.
#:   SCHEMA_INVALID_AFTER_REROLL — already retried once, in-place, by ADR-148. Retrying here
#:       would make it two.
#:   PROMPT_EXAMPLES_LEAKED — arguably re-rollable, but a model reproducing its own prompt
#:       examples is a PROMPT defect, and quietly re-rolling it hides the thing worth fixing.
#:   UNSPECIFIED — an untagged path makes no claim about its cause, and "unknown" is not
#:       "transient". It degrades, visibly, and someone classifies it.
TRANSIENT_SUMMARY_FAILURES = frozenset({RecoverableSummarizationError.TOKENIZER_THREADING})
