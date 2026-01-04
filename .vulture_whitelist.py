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
