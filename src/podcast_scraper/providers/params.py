"""Typed provider parameter models for AI experiments.

This module defines Pydantic models for provider parameters, enabling
type-safe, validated parameter passing for experiments while maintaining
backward compatibility with Config-based usage.

See ADR-028 for design rationale.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class SummarizationParams(BaseModel):
    """Parameters for summarization providers.

    This model defines all parameters that can be passed to summarization
    providers (both transformers and OpenAI). Parameters are validated and
    have sensible defaults.

    Attributes:
        model_name: Model identifier (e.g., "facebook/bart-large-cnn", "gpt-4o-mini")
        max_length: Maximum summary length in tokens
        min_length: Minimum summary length in tokens
        chunk_size: Chunk size in tokens for long transcripts (None = auto-detect)
        chunk_parallelism: Number of chunks to process in parallel (CPU only)
        use_word_chunking: Use word-based chunking instead of token-based (None = auto-detect)
        word_chunk_size: Chunk size in words for word-based chunking
        word_overlap: Overlap in words for word-based chunking
        prompt: Optional custom prompt/instruction
        temperature: Temperature for generation (0.0-2.0, OpenAI only)
        device: Device for model execution ("cpu", "cuda", "mps", or None for auto)
        reduce_model: Optional separate model for REDUCE phase (transformers only)
        cache_dir: Custom cache directory for transformer models (None = default)

    Example:
        >>> params = SummarizationParams(
        ...     model_name="facebook/bart-large-cnn",
        ...     max_length=150,
        ...     min_length=30,
        ...     device="mps"
        ... )
    """

    model_name: str = Field(
        ...,
        description="Model identifier (e.g., 'facebook/bart-large-cnn', 'gpt-4o-mini')",
    )
    max_length: int = Field(
        default=150,
        ge=1,
        description="Maximum summary length in tokens",
    )
    min_length: int = Field(
        default=30,
        ge=1,
        description="Minimum summary length in tokens",
    )
    chunk_size: Optional[int] = Field(
        default=None,
        ge=1,
        description="Chunk size in tokens for long transcripts (None = auto-detect)",
    )
    chunk_parallelism: int = Field(
        default=1,
        ge=1,
        description="Number of chunks to process in parallel (CPU only, transformers only)",
    )
    use_word_chunking: Optional[bool] = Field(
        default=None,
        description="Use word-based chunking instead of token-based (None = auto-detect)",
    )
    word_chunk_size: Optional[int] = Field(
        default=1000,
        ge=1,
        description="Chunk size in words for word-based chunking",
    )
    word_overlap: Optional[int] = Field(
        default=150,
        ge=0,
        description="Overlap in words for word-based chunking",
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Optional custom prompt/instruction",
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Temperature for generation (OpenAI only, 0.0-2.0)",
    )
    device: Optional[str] = Field(
        default=None,
        description="Device for model execution ('cpu', 'cuda', 'mps', or None for auto)",
    )
    reduce_model: Optional[str] = Field(
        default=None,
        description="Optional separate model for REDUCE phase (transformers only)",
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Custom cache directory for transformer models (None = default)",
    )

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: Optional[str]) -> Optional[str]:
        """Validate device is one of allowed values."""
        if value is None:
            return None
        valid_devices = {"cpu", "cuda", "mps", "auto"}
        if value.lower() not in valid_devices:
            raise ValueError(f"device must be one of {valid_devices}, got: {value}")
        return value.lower()

    @model_validator(mode="after")
    def _validate_word_overlap(self) -> "SummarizationParams":
        """Validate word_overlap is less than word_chunk_size."""
        if (
            self.word_overlap is not None
            and self.word_chunk_size is not None
            and self.word_overlap >= self.word_chunk_size
        ):
            raise ValueError(
                f"word_overlap ({self.word_overlap}) must be less than "
                f"word_chunk_size ({self.word_chunk_size})"
            )
        return self

    def model_dump(self, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[override]
        """Return dict representation, excluding None values for experiment configs."""
        return {k: v for k, v in super().model_dump(**kwargs).items() if v is not None}


class TranscriptionParams(BaseModel):
    """Parameters for transcription providers.

    This model defines all parameters that can be passed to transcription
    providers (both Whisper and OpenAI). Parameters are validated and have
    sensible defaults.

    Attributes:
        model_name: Model identifier (e.g., "base.en", "whisper-1")
        device: Device for model execution ("cpu", "cuda", "mps", or None for auto)
        language: Language code for transcription (e.g., "en", "fr", None = auto-detect)

    Example:
        >>> params = TranscriptionParams(
        ...     model_name="base.en",
        ...     device="mps",
        ...     language="en"
        ... )
    """

    model_name: str = Field(
        ...,
        description="Model identifier (e.g., 'base.en', 'whisper-1')",
    )
    device: Optional[str] = Field(
        default=None,
        description="Device for model execution ('cpu', 'cuda', 'mps', or None for auto)",
    )
    language: Optional[str] = Field(
        default=None,
        description="Language code for transcription (e.g., 'en', 'fr', None = auto-detect)",
    )

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: Optional[str]) -> Optional[str]:
        """Validate device is one of allowed values."""
        if value is None:
            return None
        valid_devices = {"cpu", "cuda", "mps", "auto"}
        if value.lower() not in valid_devices:
            raise ValueError(f"device must be one of {valid_devices}, got: {value}")
        return value.lower()

    def model_dump(self, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[override]
        """Return dict representation, excluding None values for experiment configs."""
        return {k: v for k, v in super().model_dump(**kwargs).items() if v is not None}


class SpeakerDetectionParams(BaseModel):
    """Parameters for speaker detection providers.

    This model defines all parameters that can be passed to speaker detection
    providers (both spaCy and OpenAI). Parameters are validated and have
    sensible defaults.

    Attributes:
        model_name: Model identifier (e.g., "en_core_web_sm", "gpt-4o-mini")
        temperature: Temperature for generation (0.0-2.0, OpenAI only)

    Example:
        >>> params = SpeakerDetectionParams(
        ...     model_name="en_core_web_sm"
        ... )
    """

    model_name: str = Field(
        ...,
        description="Model identifier (e.g., 'en_core_web_sm', 'gpt-4o-mini')",
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Temperature for generation (OpenAI only, 0.0-2.0)",
    )

    def model_dump(self, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[override]
        """Return dict representation, excluding None values for experiment configs."""
        return {k: v for k, v in super().model_dump(**kwargs).items() if v is not None}
