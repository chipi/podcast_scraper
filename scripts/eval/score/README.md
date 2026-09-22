# Application ranking gates

These two scorers stayed in the public repo when the eval research surface moved
to `chipi/podcast-scraper-eval-data` in arc 2, and the reason is worth stating
because the path says "eval":

    rank_discover_v1.py    #1139 gate — personalized discovery ranking must
                           beat plain recency on the seeded personas
    rank_scenarios_v1.py   #71 — the ranking-scenario corpus must keep
                           DISCRIMINATING

Both import **only** `podcast_scraper.server.*` — no eval library, no research
data — and `rank_discover_v1` reads public synthetic fixtures
(`tests/fixtures/app-validation-corpus/v3`,
`tests/fixtures/ground-truth/v3/seeded_users/`). They are application quality
gates that happened to live under `scripts/eval/` by naming accident, and their
tests (`tests/integration/server/test_rank_*`) are gates on the server, not
research.

Moving them broke `test_rank_discover_eval` with `n_users: 0` — the seeded
personas are public fixtures, so the scorer found nothing in the private repo.
That failure is what identified them; the blanket rule "everything under
scripts/eval moves" was right for 133 of 135 files and wrong for these two.
