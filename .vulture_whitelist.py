# vulture whitelist for false positives
# These are used dynamically or externally
# noqa: F821 - This file intentionally uses undefined _ as placeholder for vulture

_ = type("_", (), {})()  # Dummy object for vulture whitelisting

# Pydantic validators are called by framework
_.model_validator  # unused method  # noqa: B018
_.field_validator  # unused method  # noqa: B018

# Click decorators
_.callback  # unused method  # noqa: B018

# Test fixtures
_.fixture  # unused function  # noqa: B018

# TYPE_CHECKING-only imports (used in string annotations in main code)
rich = None  # cli.py
Pipeline = None  # summarizer.py
KGProximitySearch = None  # search/retrieval.py (added 2026-06-09 — was failing CI)
assert (
    rich is None and Pipeline is None and KGProximitySearch is None
)  # use so vulture does not report in this file

# Protocol method parameters — part of the interface contract, "unused" only
# because Protocol bodies are `...`. Added 2026-06-20 (was failing CI on #1037).
_.doc_ids  # search/protocol.py — VectorStore.batch_upsert + .delete  # noqa: B018
_.metadata_list  # search/protocol.py — VectorStore.batch_upsert  # noqa: B018
_.query_embedding  # search/protocol.py — VectorStore.search  # noqa: B018
_.overfetch_factor  # search/protocol.py — VectorStore.search  # noqa: B018
