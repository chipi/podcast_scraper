# ADR-166: Installed extensions — how packages add to the platform

- **Status**: Accepted
- **Date**: 2026-10-31
- **Authors**: Marko Dragoljevic
- **See Also**: `src/podcast_scraper/extensions.py`, `tests/integration/server/test_extensions.py`,
  [ADR-104](ADR-104-enrichment-layer-boundary-vs-kg-direct.md)

## Context & Problem Statement

The platform (pipeline, corpus, search, the operator API and the kernel the client apps sign in
to) had its product surfaces wired in by direct import: `server/app.py` mounted the player's
routers one by one, `server/scheduler.py` imported the digest dispatcher, `cli.py` imported the
MCP handlers, account deletion reached into player state and MCP tokens, and the enricher
registries imported every enricher class. Nothing could be added to or left out of the platform
without editing it, and a second client app would have had to copy the kernel to get sign-in,
users and sessions.

## Decision

### 1. The platform runs on its own

With nothing extra installed, the platform serves the pipeline, the corpus, plain search and the
operator API. Every surface added through an extension is optional, and every place the platform
reads one has an empty default.

### 2. Dependencies run one way

An extension imports the platform; the platform never imports an extension. It finds them only
through the discovery below.

### 3. Discovery through entry points

A package registers in the `podcast_scraper.extensions` entry-point group. Each entry point names
a module that exposes one `EXTENSION` object (`podcast_scraper.extensions.Extension`). The
platform loads them once, at first use.

### 4. Extension interface

An `Extension` carries, each part optional:

- routers per serve plane (operator, app, internal, root), mounted under the same serve postures as
  the core routers;
- user-lifecycle hooks (account created, account deleted) and server hooks (app configured,
  server started, with a stop callback);
- scheduled-job kinds for the platform scheduler;
- CLI subcommands;
- sign-in providers by name;
- themes and storylines (`search.groupings`: readers, index rows, the builder, search operators);
- share-card extras (trend stat, photos, logos);
- the listener engagement series;
- cache stats for `/api/ops/cache-stats`.

The player, the MCP server with its sign-in providers, and the enrichers outside the base set are
in-tree today and register through this same interface, so the interface is exercised by the
code it was built for.

### 5. Enrichers come from the platform and from extensions

The enricher, query-enricher, scorer and provider-type registries are fed by the platform's own
base set plus each extension's enrichment contribution. Wiring looks enrichers up by id, never by
class import. A profile that lists an enricher no installed code provides runs without it.

### 6. An installed package wins over an in-tree copy

When the same extension name is found both as an installed entry point and in-tree, the installed
one loads and the in-tree one is skipped; each extension loads once. An image that installs a
newer build of an extension over the platform therefore runs that build.

## Consequences

- **Positive**: surfaces can be added without editing the platform; a second client app reuses
  sign-in, users and sessions; the extension boundary is explicit and tested
  (`tests/integration/server/test_extensions.py` uses a fake extension).
- **Negative**: one level of indirection when reading where a route or job comes from
  (`extensions.py` is the place to look).
- **Neutral**: behaviour with the in-tree extensions only is unchanged.

## Alternatives Considered

1. **Keep direct imports behind feature flags.** Rejected: the platform still has to know every
   surface by name, and a flag does not remove the import.
2. **A plugin directory scanned at startup.** Rejected: entry points are the packaging standard,
   need no path conventions, and work in an installed image.
3. **In-tree copy wins.** This was the first behaviour; rejected because an installed newer build
   would silently not run.
