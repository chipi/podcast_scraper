# Workbench: the public viewer's rename, core fixes, then news (2026-11-17)

The public viewer (`web/workbench`, shown as "Podcast Intelligence Platform") becomes
**Workbench**: the operator's tool for a general intelligence platform. The grounded-insight UI
left it for the private viewer (ADR-170), and news is about to make it more than a podcast tool.
The words it uses move from podcast- and corpus-specific terms to general ones; news is where the
general terms first earn their keep.

**Order (the operator, 2026-11-17):** the rename first, then the core fixes, then news. Nothing
here is built yet.

## 1. The rename

### What is renamed

| Today | Workbench | Notes |
| --- | --- | --- |
| `web/workbench/` | `web/workbench/` | `git mv`, history kept |
| `package.json` `"name": "gi-kg-viewer"` | `"workbench"` | and `package-lock.json` |
| Page title and header "Podcast Intelligence Platform" | "Workbench" | `index.html:14`, `App.vue:905`, `LoginView.vue:32` |
| "GI/KG viewer" in docs (167 mentions in 87 files) | "Workbench" | prose only; history docs (RFCs, ADRs, releases) keep their words |
| `podcast-scraper-stack-viewer` image, `viewer` compose service, `docker/viewer/` | `podcast-scraper-stack-workbench`, `workbench`, `docker/workbench/` | decided 2026-11-17; the image is published under both names until the deployment switches (chipi/infra #84) |

### Where the path is load-bearing

`gi-kg-viewer` appears 820 times in 135 tracked files. The ones that break something if missed:

- **The server mounts the built app:** `src/podcast_scraper/server/app.py:179`
  (`web/workbench/dist`) and the `--no-spa` help in `server/cli_handlers.py:118`.
- **CI:** path filters and working directories in `python-app.yml` (5 places in the filters, more
  in the viewer jobs) and `nightly.yml` (cache paths, working directories, artifact paths).
- **The viewer image:** `docker/workbench/Dockerfile:53,56` and `.dockerignore:58-59`.
- **Make and scripts:** `Makefile:43` (`WEB_VIEWER_DIR`), `scripts/tools/gates_changed.sh`,
  `scripts/tools/check_doc_structure.py`, `scripts/docs_strip_emoji.py`,
  `scripts/build_production_shaped_fixture.py`.
- **Ignores:** about 30 `web/workbench/...` rules in `.gitignore`.
- **The private viewer (Studio) carries its own copy** under `apps/studio/web` with the same
  package name, and its `platform-root.mjs` finds the platform by walking up from either layout.
  Its tests read `UXS-001-workbench.md` by path. Studio's owner changes its side; the public PR
  says what moved.
- **The deployment** (operator tooling, not in this repository) builds and runs the viewer image.
  Whether the image and service are renamed is the operator's call; renaming them is a deploy
  change, not a code change.

### Browser storage keys

Saved per-browser settings use the old name: `gi-kg-viewer-theme` (theme), `gikg.library.mode`
(Library browse mode) and `gi-kg-graph-minimap` (graph minimap). Renaming them without a step
resets every operator's settings once. The rename reads the old key when the new one is absent,
writes the new one, and removes the old one.

### Kept as they are (for now)

- **API routes** (`/api/corpus/...`, `/api/episodes/...`): the private apps and the MCP tools use
  them. Renaming them is its own change, if ever.
- **`data-testid` values:** they are the e2e contract (`E2E_SURFACE_MAP.md`); a rename of copy does
  not need to move them.
- **Code identifiers** such as `focusEpisode` or `corpusPath`: renamed only where a general term
  replaces a podcast one in the same change, not in bulk.

## 2. The vocabulary

The visible copy is written into the components (no copy file today). Rough counts of the words a
user reads: "Episode(s)" about 67, "Feed" 16, "Corpus" 13, "Speaker" 10, "Podcast" 8, "Transcript"
8. Proposal: one copy module (`src/copy/`) holds the terms, so a later change is one edit.

| Today | Proposed general term | Where the specific word stays |
| --- | --- | --- |
| Corpus ("Corpus path", "corpus") | **kept** (recommended 2026-11-17 after the operator asked; not objected to): not podcast-specific, and the platform's word for the store | |
| Episode | **Item** | "Episode" as a podcast item's type label; "Article" for news |
| Show, Feed | **Source** | "Show" or "Publication" as the type label |
| Speaker | **Person** | "Speaker" and "Author" as roles |
| Transcript | **Text** | "Transcript" for a podcast item, never shown for news (ADR-179) |
| Podcast Intelligence Platform | **Workbench** | |

Topic, entity, person, digest, search, graph and dashboard are general already and stay.

## 3. The core fixes (from the public-only review, 2026-11-17)

Found on a public-only install against `tests/fixtures/app-validation-corpus/v3`. Each bug gets a
failing test first.

1. **Configuration > Index "Full rebuild" swallows its failure:** it says running, then "No index
   built yet", with no error shown.
2. **Dashboard > Pipeline contradicts itself:** a job that failed in 0.2 s shows "running" beside
   "No queued or running jobs", and the outcomes count 0 failed.
3. **Ops probes `127.0.0.1:8000`** (health, version, runs) when the API runs on another port.
4. **Dashboard > Coverage:** `GET /api/corpus/documents/manifest` returns 404.
5. **`make serve-api` signs no one in by default:** every dev user gets 403 until
   `APP_SIGNUP_MODE=open`.
6. **The corpus path starts empty** although the server was started with `--output-dir`.
7. **Empty or odd without grounded insights:** the topic rail's Details tab is three buttons; the
   person rail is a name and two buttons; the graph's type chip says "Person (0)" while person nodes
   are drawn; Dashboard > Intelligence has a "TOPICS" heading with nothing under it; the command
   palette prints the raw token `no_index`.
8. **Grounded-insight leftovers in about 15 components** (copy such as "insights", "Show on graph",
   "GI Person"; `api/cilApi.ts` and `api/relationalApi.ts` calling routes that do not exist on
   public). Decide per item: remove, or keep as an extension point the private viewer fills.
9. **The UX docs** (`docs/uxs/`) still describe insights, quotes, the position tracker and Explore as
   this viewer's; none mentions the private viewer.
10. **The show node page finds its feed by title** (`PodcastNodeView.vue:36-48`; the database-era
    sweep's R16, [WORKBENCH-DB-ERA-SWEEP-2026-11-17.md](WORKBENCH-DB-ERA-SWEEP-2026-11-17.md)). R3b
    changes GI's show id to KG's, and the page then goes blank, so this ships in the news release
    before R3b, on the read Common provides (show id beside the title, or R3b's old-to-new id map).

Not checked yet: search with a built index, the Digest after a successful job (#2486), canvas
clicks on topic and person nodes, phone width and the light theme.

## 4. Then news

UXS-019 (draft, on branch `docs/uxs-019-news`, not merged) is written in
today's words. It is revised to the vocabulary above before it is built.

## Decided (2026-11-17)

1. **"Corpus" stays** (recommended after the operator asked; not objected to).
2. **The image, the compose service and `docker/viewer/` are renamed now**, in the news release:
   infra's main reads either layout at deploy time (chipi/infra #84, step 1), public publishes the
   image under both names until a prod deploy has run on `-workbench` (step 3 then drops the old
   name). The DR drill's `tests/stack-test/stack-viewer.spec.ts` keeps its file name: infra runs it
   by name.
3. **UXS-001 and the IA doc are renamed** (`UXS-001-workbench.md`, `WORKBENCH_IA.md`). The private
   viewer builds against a pinned public commit and switches its paths when it repins.
