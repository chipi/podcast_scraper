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
        reason: Reason for the recoverable failure
    """

    def __init__(self, episode_idx: int, reason: str) -> None:
        self.episode_idx = episode_idx
        self.reason = reason
        message = (
            f"[{episode_idx}] Summarization failed (recoverable): {reason}. "
            "Metadata generation will continue without summary."
        )
        super().__init__(message)
