# MCP core server — e2e test guide

The **standard corpus** for the pivot-chain e2e is the committed UI tier-3 fixture
`tests/fixtures/app-validation-corpus/v3` (deterministic, version-pinned, CI-available).
Its two-tier search index is **built at test setup** (offline, cached MiniLM) — never
committed, to avoid a binary lance blob + lance-format-version coupling. The larger
`.test_outputs/manual/prod-v2/corpus` (209 episodes) is an optional local alternative.

Three ways to exercise the RFC-095 cross-surface server, all driving the same "golden case"
that forces cross-surface pivoting.

## 1. CI e2e test (standard — the synthetic corpus)

```sh
.venv/bin/python -m pytest tests/integration/test_mcp_pivot_chain_e2e.py -v
```

Builds the index at setup if absent, then asserts ids flow search→graph across surfaces.
Skips cleanly where the embedding model isn't available (model-less unit CI); runs fully in
the ML tier / locally. This is the CI-able, standardized regression guard.

## 2. Automated harness (ad-hoc, any corpus)

```sh
# standard synthetic corpus (build the index once, offline):
HF_HUB_OFFLINE=1 .venv/bin/python -m podcast_scraper.cli index-two-tier \
  --output-dir tests/fixtures/app-validation-corpus/v3
.venv/bin/python scripts/mcp_e2e_pivot_chain.py --corpus tests/fixtures/app-validation-corpus/v3

# or the larger local prod-v2 snapshot (already indexed):
.venv/bin/python scripts/mcp_e2e_pivot_chain.py --corpus .test_outputs/manual/prod-v2/corpus
```

Drives the golden pivot chain through the registered tools and prints a readable trace,
asserting ids flow surface→surface (centrality → momentum → relational → temporal → search →
**pivot bridge** → graph → composite). Exit 0 = the chain connected. Read-only.

## 3. Real Claude MCP client (wire layer — the true dogfood)

Point a Claude MCP client at the stdio server, then give it the golden prompt and watch it
pick + chain the tools itself.

**Claude Code** (this repo):

```sh
claude mcp add podcast-corpus -- \
  .venv/bin/python -m podcast_scraper.cli mcp --corpus "$PWD/.test_outputs/manual/prod-v2/corpus"
```

**Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "podcast-corpus": {
      "command": "/ABSOLUTE/PATH/.venv/bin/python",
      "args": ["-m", "podcast_scraper", "mcp",
               "--corpus", "/ABSOLUTE/PATH/.test_outputs/manual/prod-v2/corpus"]
    }
  }
}
```

Requires the `.[dev,search]` extras (the MCP SDK ships in `[dev]`). Verify the server
launches: `.venv/bin/python -m podcast_scraper.cli mcp --corpus <dir>` (it waits on stdio).

### The golden prompt

> *"Using the podcast corpus tools: what's the most contested topic in the corpus, who's on
> each side, how has the conversation evolved over time, and how do the two loudest voices
> differ? Show me the grounding as you go."*

### What to watch for (the pivot chain)

A good run chains across surfaces without you telling it which tools to use:

1. `topic_perspective_leaders` → the most-debated topic (centrality)
2. `corpus_trending(kind=topic)` → is it also rising?
3. `who_said_about_topic` → the sides, grouped by speaker
4. `topic_conversation_arc` → sentiment/volume over time
5. `search_corpus(topic=…)` → grounded hits, each carrying a `pivot` handle
6. `insight_detail(pivot.id)` → **the bridge**: a search hit → its entities/topics
7. `entity_neighborhood` / `ego_network` → expand a key voice
8. `compare_subjects(A, B)` → contrast the two loudest voices

Or the one-call shortcut: `entity_dossier(topic_id)` / `episode_digest(metadata_path)`.

**Usability signals to note** (feed back into tool descriptions): did Claude discover the
`pivot.id` handle on its own? Did it reach for the composite dossiers, or hand-chain? Were
any descriptions ambiguous enough that it picked the wrong tool?

*Corpus path assumes the prod-v2 snapshot under `.test_outputs/manual/`. Point `--corpus` at
any corpus dir with a `search/` index + `*.gi.json` to test another.*
