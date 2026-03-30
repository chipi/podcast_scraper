# Local pip wheels (optional)

The `wheels/spacy/` directory holds spaCy English model `.whl` files when you run
`make download-spacy-wheels`. Those files are **gitignored**; this README is committed so the
folder layout is documented.

**Using them:** run `make init` (or other Makefile targets that invoke pip); if `*.whl` files
exist under `wheels/spacy/`, the Makefile sets `PIP_FIND_LINKS` automatically.
`scripts/setup_venv.sh` does the same for its first `pip install -e .`.

**Persist in an active shell:** after `source .venv/bin/activate`, you can set `PIP_FIND_LINKS`
each time via the Makefile, or patch the venv once so activation does it automatically:

```bash
python3 scripts/patch_venv_activate_spacy_wheels.py
source .venv/bin/activate
```

Re-run the patch script if you recreate `.venv`. The patch lives only under `.venv/` (not in git).

See [Dependencies Guide](../docs/guides/DEPENDENCIES_GUIDE.md#optional-local-wheel-cache-for-spacy-models)
for details and manual `pip` usage.
