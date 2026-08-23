"""Corpus quality assessment — stage outcomes and attribution health (#1647).

Separate from ``evaluation`` (which scores model output against references) and from the
coverage endpoints (which count artifact presence). This package answers a different
question: *did the pipeline actually do its work, and can the corpus attribute what it
produced to a person?*

That question had no home before #1646, which is why the answer was "yes" for a corpus where
23 % of insights were unusable.
"""
