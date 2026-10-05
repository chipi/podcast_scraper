"""pipeline-check: run the pipeline under variants, record every decision, compare runs (#2287).

A variant is a set of parameters the pipeline is run with — today a locale (the feed's declared
language tag), tomorrow a profile or any config override. The tool answers two questions with one
machinery:

* **code vs code, same variant** — did a change alter the pipeline's behaviour?
  (``BASE=main``: the base ref and the candidate are run from separate checkouts on the same input)
* **variant vs variant, same code** — how does the pipeline treat these differently?
  (``LOCALES="en en-US"``: must be identical; ``LOCALES="pt-BR pt-PT"``: reported)

Layout:

* :mod:`.recorder` — runs inside the pipeline's own process; discovers keyed decision maps and
  decision functions and logs every choice they make;
* :mod:`.drivers` — drives the deterministic stages over a fixture corpus, one variant at a time;
* :mod:`.worker` — the process entry point: one checkout, one corpus, many variants -> one JSON;
* :mod:`.compare` — applies ``expectations.yaml`` to the workers' JSON;
* :mod:`.report` — the one-page verdict;
* :mod:`.cli` — ``make pipeline-check``: refs -> checkouts, a worker per side, compare, report.

The runbook is ``docs/guides/PIPELINE_CHECK.md``.
"""
