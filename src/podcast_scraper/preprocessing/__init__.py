"""Preprocessing module for transcript cleaning and normalization.

This module provides:
- Individual cleaning functions (clean_transcript, remove_sponsor_blocks, etc.)
- Registered preprocessing profiles (versioned cleaning pipelines)
- Provider-agnostic preprocessing that works with all providers
"""

from __future__ import annotations

# Re-export all public attributes from core.py
# This includes functions, constants, and patterns
import sys

# Import and re-export everything from core.py
from . import core, profiles

_current_module = sys.modules[__name__]
for attr_name in dir(core):
    if not attr_name.startswith("_"):
        # Import into this module's namespace
        setattr(_current_module, attr_name, getattr(core, attr_name))

# Re-export profile functions
from .profiles import (
    apply_profile,
    DEFAULT_PROFILE,
    get_profile,
    list_profiles,
    register_profile,
)

# Update __all__ to include both cleaning functions and profile functions
__all__ = [
    # Profile functions (explicitly listed)
    "DEFAULT_PROFILE",
    "apply_profile",
    "get_profile",
    "list_profiles",
    "register_profile",
    "profiles",
    "core",
    # Cleaning functions are included via dir() above, but we can't list them
    # explicitly here because they're dynamically added
]
