"""Summarization provider protocol and factory.

This package contains:
- base.py: SummarizationProvider protocol definition
- factory.py: Factory function for creating summarization providers
"""

from . import base, factory

__all__ = ["base", "factory"]
