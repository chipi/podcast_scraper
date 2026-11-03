# Application ranking gates

These two scorers are application ranking gates, not research, and the reason is
worth stating because the path says "eval":

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
