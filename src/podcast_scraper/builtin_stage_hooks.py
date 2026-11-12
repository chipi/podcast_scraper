"""Public's own stages on the pipeline's stage hooks (E04 S2): none today.

Every stage that used to be built in (speakers and diarization, grounded insights, the identity
ledger, the advanced KG passes, ad cleaning, audio sync) moved to the Close Listening stages
extension with the E04 move, which registers them through ``Extension.pipeline_hooks``. Without
it each hook answers its plain default (``podcast_scraper.stage_hooks``). A stage public adds
later registers here, before every extension.
"""

from __future__ import annotations

from typing import Dict, Tuple

from .stage_hooks import PipelineHooks

#: Each provider class's stage methods registered by public itself (``providers.extensible``).
PROVIDER_METHODS: Dict[str, Tuple[str, ...]] = {}

BUILTIN = PipelineHooks(name="builtin", provider_methods=PROVIDER_METHODS)
