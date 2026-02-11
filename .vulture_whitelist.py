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
assert rich is None and Pipeline is None  # use so vulture does not report in this file
