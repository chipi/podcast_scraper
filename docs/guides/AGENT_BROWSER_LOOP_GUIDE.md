# Agent-Browser Closed Loop Guide

**Status:** Reference Guide
**Applies to:** Local development, agent-assisted UI debugging, E2E test workflows
**Last updated:** April 2026

---

## Overview

This guide documents the **closed loop between an AI coding agent and the browser**
for UI development in this project. The loop operates in two modes:

- **Automated mode** — the agent runs `make test-ui-e2e` (Playwright, headless),
  reads structured feedback, fixes code, and re-runs until green. The browser
  equivalent of `make test` for Python.
- **Live co-development mode** — you and the agent share the same Chrome session.
  You direct ("fix this", "add a spinner here", "why is this column empty"), the
  agent sees the DOM/console/network, edits code, Vite hot-reloads, and both of you
  see the result instantly. Bug fixing and iterative feature work are the same loop —
  only the starting prompt differs.

Both modes feed back into each other: automated catches regressions; live
co-development catches UX issues and lets you steer the agent with visual context.

**Companion guide:** [Agent-Pipeline Feedback Loop Guide](AGENT_PIPELINE_LOOP_GUIDE.md)
covers the Python-side loop (`make ci`, acceptance tests, `--monitor`, `metrics.json`).
Same principle — give the agent direct access to structured feedback — applied to
the pipeline instead of the browser.

### Two tools, two jobs

| Tool | Job | Analogy |
| ---- | --- | ------- |
| **Playwright MCP** | *Driving* — navigate, click, fill, assert | Automated QA engineer |
| **Chrome DevTools MCP** | *Debugging* — network tab, console, DOM, traces | Human dev with DevTools open |

For a full closed loop you use both: Playwright runs the test suite (automated mode),
Chrome DevTools MCP gives the agent eyes into your live session (co-development mode).

### Why MCP?

MCP (Model Context Protocol) is the communication layer between the agent and the
browser tool. It has nothing to do with remote vs local — the browser, the MCP server,
and your app all run on your machine. Without MCP, the agent has no formalized way to
invoke browser actions and get structured feedback. MCP is the glue.

---

## Browser choice

**Use Chrome (or Chromium/Edge) for manual dev debugging. Firefox for automated E2E.**

The manual "agent attaches to your live session" workflow requires Chrome DevTools
Protocol (CDP), which is Chrome-only. Firefox has no equivalent.

This project's Playwright E2E suite (`make test-ui-e2e`) runs **Firefox** headlessly
for cross-browser coverage (see `web/gi-kg-viewer/playwright.config.ts`). That is
separate from the agent-browser loop described here.

| Context | Browser | Why |
| ------- | ------- | --- |
| `make test-ui-e2e` (CI, automated) | Firefox | Cross-browser coverage, existing config |
| Agent-driven exploration (MCP) | Chromium (Playwright default) | Playwright MCP launches its own Chromium |
| Live co-development | Chrome | CDP required for DevTools MCP attachment |

---

## Automated mode

### How it works

The agent opens a headless Chromium browser via Playwright MCP, navigates your app,
and gets structured feedback after **each action** — not just pass/fail at the end.

What the agent sees is an **accessibility snapshot**, not a screenshot:

```text
- heading "Podcast Intelligence Platform" [level=1]
- tab "Digest" [selected]
- tab "Library"
- tab "Graph"
- textbox "Search corpus…" [ref=e5]
- button "Search" [ref=e12]
```

It uses `ref=e12` to click — no CSS selectors, no pixel coordinates, no fragile
locators.

With the `devtools` capability enabled, the agent also sees:

- **Console messages** — JS errors, warnings, `console.log` output
- **Network requests** — URL, method, status, request/response payload
- **Performance traces** — via Chrome DevTools MCP

### Typical flow

```text
You:    "Navigate to the viewer, load the graph, search for 'machine learning',
         and check what API calls fire"

Agent:  browser_navigate → http://127.0.0.1:5174
        browser_click    → "Graph" tab
        browser_fill     → search box
        browser_click    → "Search" button
        browser_network_requests → [sees /api/search?q=machine+learning, 200, payload]
        browser_console_messages → [no errors]
        → reports findings
```

### E2E as a validation gate (the primary workflow)

`make test-ui-e2e` is the **browser equivalent of `make test`**. After every viewer
change the agent runs it, reads failures, fixes code, and re-runs until green — the
same loop you already have for Python.

| Python workflow | Browser workflow |
| --------------- | ---------------- |
| Edit `src/` | Edit `web/gi-kg-viewer/src/` |
| `make test` (pytest) | `make test-ui-e2e` (Playwright) |
| Read failure → fix → re-run | Read failure → fix → re-run |
| Green → done | Green → done |

The E2E suite for the GI/KG viewer:

| Aspect | Value |
| ------ | ----- |
| Run command | `make test-ui-e2e` |
| Config | `web/gi-kg-viewer/playwright.config.ts` |
| Browser | Firefox (headless) |
| Port | `127.0.0.1:5174` (dedicated, avoids dev server on 5173) |
| Specs | `web/gi-kg-viewer/e2e/*.spec.ts` |
| Surface contract | `web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md` |

When a UI feature is done, the agent should:

1. **Update or write specs** — add/modify `e2e/*.spec.ts` to cover the new behavior.
   The `E2E_SURFACE_MAP.md` lists every surface, its selectors, and which spec owns
   it — the agent reads this to know what vocabulary to use and which spec to extend.
2. **Run `make test-ui-e2e`** — the full suite, not just the new spec. Catches
   regressions in other surfaces.
3. **Read failures, fix, re-run** — use the three artifacts below.
4. **Green → commit** — the feature is validated.

This is the bread-and-butter loop. Everything else in this guide (live
co-development, MCP exploration) builds on top of it.

### What the agent reads after `make test-ui-e2e`

Three artifacts are generated on every run. The agent reads them directly as files —
no copy-paste needed.

**1. Terminal output (always available)**

The `list` reporter prints pass/fail per test with assertion errors inline. The agent
sees this directly from the Shell tool output. For most failures, this is enough.

**2. JSON results — `web/gi-kg-viewer/e2e-results.json`**

A structured JSON file with per-test `title`, `status` (`passed`/`failed`/`timedOut`),
`duration` (ms), and on failure: `error.message`, `error.stack`, and the failing
`location` (file + line). The agent can read this to get a machine-parseable summary
without scraping terminal text.

Example of what the agent sees for a failure:

```json
{
  "title": "shows corpus summary counts when API and corpus path are available",
  "status": "unexpected",
  "results": [{
    "status": "failed",
    "duration": 15023,
    "error": {
      "message": "Expected: visible\nReceived: hidden",
      "location": { "file": "dashboard.spec.ts", "line": 52 }
    }
  }]
}
```

**3. Trace files — `web/gi-kg-viewer/test-results/` (on failure)**

When a test fails locally, Playwright records a trace zip containing DOM snapshots,
network requests, console logs, and screenshots at every step. The agent can:

- Read the trace directory listing to find which tests failed
- Run `npx playwright show-trace <path>/trace.zip` to open the trace viewer
- Or describe the trace path to you so you can open it in a browser

Traces are only persisted for **failing** tests (locally: `retain-on-failure`; CI:
`on-first-retry`). Passing tests do not leave trace files.

**4. HTML report — `web/gi-kg-viewer/playwright-report/`**

A rich HTML report with screenshots, trace links, and error details. The agent cannot
read HTML directly, but can tell you to open it:

```bash
npx playwright show-report web/gi-kg-viewer/playwright-report
```

### How to instruct the agent

You don't need special prompts — the agent already runs `make test-ui-e2e` and reads
the terminal output. But you can get more out of it:

**Basic (already works):**

```text
"Run make test-ui-e2e and fix any failures"
```

The agent runs the command, reads terminal output, diagnoses failures, fixes code,
re-runs.

**With JSON results (richer analysis):**

```text
"Run make test-ui-e2e, then read web/gi-kg-viewer/e2e-results.json
 and give me a summary: how many passed, failed, total duration,
 and details on any failures"
```

The agent reads the structured JSON and reports a clean summary with test names,
durations, and error messages.

**With trace inspection (deep debugging):**

```text
"Run make test-ui-e2e. If anything fails, check the trace files
 in web/gi-kg-viewer/test-results/ — look at the DOM snapshot
 and network requests at the point of failure"
```

The agent lists the trace directory, reads the trace metadata, and reports what the
DOM and network looked like when the assertion failed.

**Full loop:**

```text
"I just finished the Digest topic bands. Update the E2E specs to cover
 topic band rendering and click-to-search. Run make test-ui-e2e.
 If failures, read e2e-results.json for details, fix, and re-run
 until green."
```

### Interactive exploration (beyond the test suite)

The agent-browser loop via Playwright MCP **complements** the test suite — it does
not replace it. Use MCP when:

- You need the agent to **explore** an unfamiliar UI state interactively
- You want to **write a new spec** and need the agent to discover selectors first
- A test failure is confusing and you want the agent to **step through the flow**
  with console and network visibility

---

## Live co-development mode

You and the agent share the same Chrome session. The agent is attached via Chrome
DevTools MCP and can see everything you see — DOM, console, network. You direct
changes in natural language, the agent edits code, Vite hot-reloads the change in
your browser, and the agent observes the result through the same session. You react,
the agent iterates. The loop is continuous.

Bug fixing and feature work are the same workflow — only the opening prompt differs:

- **Bug:** "The Library episode list shows no publish dates — check the API response"
- **Feature:** "Add a loading spinner to the Library while episodes are fetching"
- **Polish:** "Make the topic pills wrap instead of overflow, and use the accent color"

In all three cases the mechanics are identical: agent sees the current state, edits
code, sees the hot-reloaded result, you confirm or redirect.

### Setup: launch Chrome with debugging port

```bash
# macOS — launch Chrome with CDP enabled
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --remote-debugging-port=9222 \
  --user-data-dir=/tmp/chrome-debug-profile
```

Add a shell alias for convenience:

```bash
# In your .zshrc
alias chrome-dev='/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug-profile'
```

### How the loop works

```text
1. You open Chrome (chrome-dev), navigate to http://127.0.0.1:5174
2. Vite dev server is running (make serve or npm run dev in web/gi-kg-viewer)
3. Agent attaches via Chrome DevTools MCP — now shares your session
4. You direct: "The episode metadata row is too cramped — add spacing
   between the feed name and the publish date"
5. Agent sees:
   - The DOM layout you are looking at
   - The CSS computed styles on the metadata row
   - Any console warnings
6. Agent edits the Vue component (e.g. LibraryView.vue)
7. Vite hot-reloads → your browser updates in place
8. Agent sees the new DOM state through the same attached session
9. Agent confirms: "Spacing added — 8px gap between feed and date.
   No console errors. Does this look right?"
10. You: "Good, but also truncate long feed names with ellipsis"
11. Agent edits again → hot-reload → agent verifies → you confirm
12. When satisfied: agent runs make test-ui-e2e to catch regressions
```

### What makes this work

- **Shared session** — the agent inherits your exact browser context (navigation
  state, loaded corpus, scroll position). No reproduction needed.
- **Vite HMR** — code changes appear in your browser within milliseconds. The agent
  does not need to refresh or re-navigate.
- **Agent observes the effect** — after each edit, the agent reads the updated DOM,
  console, and network through the still-attached DevTools session. It can
  self-correct before you even react.
- **You stay in control** — the agent never navigates away or changes your browser
  state. It only edits code; the browser reflects the change via hot-reload.

---

## The full closed loop

The two modes feed into each other naturally:

```text
┌─────────────────────────────────────────────────┐
│  LIVE CO-DEVELOPMENT (you + agent, same Chrome) │
│                                                 │
│  You direct → agent edits → Vite reloads        │
│  → agent sees result → you react → repeat       │
│                                                 │
│  When satisfied ↓                               │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  AUTOMATED VALIDATION (agent alone, headless)   │
│                                                 │
│  Agent updates e2e/*.spec.ts                    │
│  → make test-ui-e2e (full suite, Firefox)       │
│  → failures? fix + re-run                       │
│  → green ✓                                      │
└─────────────────────────────────────────────────┘
                    ↓
            Ready to commit
                    ↓
        (or back to live co-dev if
         you spot something new)
```

---

## Console and network access

With the `devtools` capability enabled, the agent has access to:

| Channel | What it sees |
| ------- | ------------ |
| Console | JS errors, warnings, your `console.log` output |
| Network | URL, method, status code, request payload, response body |
| Performance | Traces via Chrome DevTools MCP (`performance_start_trace`, `performance_analyze_insight`) |

Example prompt that exercises all three:

```text
"Navigate to /dashboard, trigger a data refresh, then check:
 1. Did the /api/corpus/stats request return 200?
 2. What payload did it return?
 3. Any console errors during the refresh?"
```

The agent answers all three from a single flow — no manual DevTools needed.

---

## MCP vs Playwright CLI

Playwright ships both an MCP server and a standalone CLI. The tradeoff is token cost
vs reasoning depth:

| | MCP | CLI |
| --- | --- | --- |
| Token cost | Higher (full accessibility tree in context) | ~4x cheaper |
| Best for | Reasoning about unknown page structure, persistent browser state | Well-defined, repeatable automation |
| Session model | Browser stays alive across turns | Command-per-invocation |

**Practical rule:** start with CLI for known flows (navigate, click, assert). Switch
to MCP when the agent needs to reason about what it finds — exploratory debugging,
unfamiliar UI states, complex multi-step reproduction.

---

## Use cases for this project

### UC-1: Feature done → write specs → green gate

**Problem:** You finish a UI feature (new Episode rail, Dashboard chart, search
filter). You need to validate it works and doesn't break anything — the same way
`make test` validates Python changes.

**Tool:** `make test-ui-e2e` (the gate) + Playwright MCP (optional exploration).

**Flow:**

1. Agent finishes the Vue component / store / API integration
2. Agent reads `E2E_SURFACE_MAP.md` to find the owning spec and selectors
3. Agent updates or creates `e2e/*.spec.ts` to cover the new behavior
4. Agent runs `make test-ui-e2e` — full suite
5. Failures → agent reads Playwright output, diagnoses, fixes code or spec, re-runs
6. Green → ready to commit

**When the spec itself is hard to write:** the agent can switch to Playwright MCP
to interactively explore the page — discover what the accessibility tree looks like,
which `ref` values to use, what network calls fire — then translate that into a
proper spec.

**Prompt example:**

```text
"I just finished the Episode detail rail. Update the E2E specs to cover:
 - clicking an episode in Library opens the rail
 - the rail shows episode title, metadata, and key points
 - clicking 'Open in graph' switches to the Graph tab
 Run make test-ui-e2e and fix any failures."
```

### UC-2: Live co-development on the GI/KG viewer

**Scenario A — bug:** The graph loads but search results don't highlight nodes.

1. You're in Chrome, looking at the graph after a search
2. "Attach to my browser. I searched 'machine learning' — 5 hits but no nodes
   highlight. Check the search API response and console."
3. Agent inspects `/api/search?q=machine+learning` payload, console, `.graph-canvas`
4. Agent finds the mismatch, edits the Vue component, Vite reloads
5. You see nodes highlight — "good, but the highlight color is too faint"
6. Agent adjusts the CSS, reload, you confirm
7. Agent runs `make test-ui-e2e` to lock it in

**Scenario B — feature:** You want to add a "Prefill search" button to the Episode
detail rail.

1. You're in Chrome, looking at the Episode rail in the Library tab
2. "Add a 'Prefill search' button below the episode title. When clicked, it should
   fill the search box with the episode title."
3. Agent sees the rail DOM, edits `EpisodeRail.vue`, adds the button + store wiring
4. Vite reloads — button appears in your browser
5. You click it — search box fills — "works, but put it next to 'Open in graph'"
6. Agent moves it, reload, you confirm
7. Agent updates the E2E spec, runs `make test-ui-e2e` → green

Both scenarios are the same loop: you direct, agent edits, hot-reload, agent
observes, you react.

### UC-3: Exploratory validation of new UI (agent drives browser)

**Problem:** You want the agent to click through the app and report what it sees —
a quick smoke test beyond what the spec suite covers.

**Tool:** Playwright MCP (automated mode).

**Flow:**

1. Agent navigates to `http://127.0.0.1:5174`
2. Clicks through each tab — Digest, Library, Graph, Dashboard
3. Checks accessibility snapshot after each click (did the tab render content?)
4. Checks console for JS errors
5. Checks network: did the right API endpoints fire?
6. Reports pass/fail per tab with evidence

**Prompt example:**

```text
"Navigate to localhost:5174, click each main tab in sequence. After each one:
 - confirm the tab content rendered (not blank)
 - check for console errors
 - list any network calls made
 Report a summary table."
```

### UC-4: Run comparison tool (Streamlit) — network profiling

**Problem:** The Streamlit comparison tool (`make run-compare`, port 8501) loads
slowly or shows stale data. You don't know if it's a slow API call, a large payload,
or a rendering issue.

**Tool:** Chrome DevTools MCP attached to your live Chrome session.

**Flow:**

1. Open the Streamlit app in Chrome (`chrome-dev`), navigate to the Performance tab
2. Trigger the slow action (load a comparison, switch pages)
3. Tell the agent: "Profile the network calls that just fired — which took longest,
   what did they return, any errors?"
4. Agent reports: timing per request, payload sizes, any 4xx/5xx
5. If a specific call is the culprit: "Trace that back to the Python handler and
   tell me where the bottleneck is"

**Boundary:** this covers browser-visible HTTP traffic only. If the bottleneck is
inside a Python function that Streamlit calls synchronously (e.g., a slow model
inference), the agent sees a slow response but not *why* it's slow internally — that
is `py-spy` territory (RFC-064).

### UC-5: Catching silent failures in pipeline output UI

**Problem:** The pipeline runs, produces output, the UI shows "success" — but the
displayed data is wrong or incomplete. No exception was raised.

**Tool:** Chrome DevTools MCP (live co-development mode).

**Flow:**

1. You see wrong data in the viewer ("episode 42 shows 0 entities extracted")
2. Tell the agent: "Look at my browser — the entities count for episode 42 is wrong.
   Check what the API returned for that row."
3. Agent inspects the network response for that specific data fetch
4. Finds: API returned correct data but the Vue component transformation mangled it
   — or: API itself returned 0 (upstream pipeline issue)
5. Narrows the bug to frontend vs backend in one step
6. If frontend: agent fixes the component, Vite reloads, you see the correct count

This is the "network tab tells you why, not just what" pattern.

### UC-6: Post-run UI validation (future)

**Problem:** After a pipeline run, results are written and the viewer should reflect
the new data. You want an automated check that the UI actually updated, not just that
the files were written.

**Tool:** Playwright MCP or CLI (headless, scriptable).

**Flow:**

```text
Post-run hook:
  → Agent navigates to viewer, checks corpus stats
  → Compares: did the episode count increase? Did the latest run appear?
  → Reports UI validation result alongside pipeline output
```

This closes a gap: you currently validate pipeline output quality, but not whether
the *reporting layer* correctly reflects it. A headless Playwright check after each
run catches silent UI regressions without manual review.

**Status:** Aspirational — not yet implemented.

---

## Playwright test agents (future)

Playwright ships three composable agents that can be chained for test generation:

| Agent | Job |
| ----- | --- |
| `planner` | Explores your app, produces a Markdown test plan |
| `generator` | Transforms the plan into Playwright test files |
| `healer` | Executes the suite, automatically repairs failing tests |

Initialize:

```bash
npx playwright init-agents
```

These work on top of Playwright MCP — they are higher-level orchestration, not a
replacement. Not yet used in this project; listed here as a future option for
accelerating test creation.

---

## CI integration

The automated mode is already in CI — the `viewer-e2e` workflow job runs
`make test-ui-e2e` headlessly on every PR. Live co-development is inherently local.

What CI provides today:

| What | Status |
| ---- | ------ |
| `make test-ui-e2e` as PR gate | Done (`viewer-e2e` job) |
| Playwright traces on first retry | Done (`trace: 'on-first-retry'` in config) |
| Trace/report upload as CI artifacts | Not yet — see [WIP ideas](../wip/WIP-agent-browser-ci-ideas.md) |

**Practical next step:** upload Playwright trace zips and the HTML report as CI
artifacts on failure. When a CI-only failure is hard to reproduce locally, the agent
can download the trace and inspect DOM snapshots, network calls, and console logs
post-mortem — closing the loop between CI and the local agent workflow.

For additional CI enhancement ideas (screenshot diffs, console error gate,
accessibility audit), see [WIP: Agent-browser CI ideas](../wip/WIP-agent-browser-ci-ideas.md).

---

## Boundary: what these tools cover

| Concern | Right tool | Guide |
| ------- | ---------- | ----- |
| Browser-visible HTTP traffic | Chrome DevTools MCP | This guide |
| UI rendering, console errors, DOM state | Chrome DevTools MCP / Playwright MCP | This guide |
| Regression testing UI flows | `make test-ui-e2e` / Playwright MCP | This guide |
| Python pipeline timing and resources | `.monitor.log`, `metrics.json` | [Agent-Pipeline Loop](AGENT_PIPELINE_LOOP_GUIDE.md) |
| CI failure diagnosis | Terminal output, test logs | [Agent-Pipeline Loop](AGENT_PIPELINE_LOOP_GUIDE.md) |
| Python process CPU/memory profiling | `py-spy`, memray, RFC-064 profiles | [Agent-Pipeline Loop](AGENT_PIPELINE_LOOP_GUIDE.md) |

The clean split: anything that crosses the HTTP boundary → this guide (browser
tools). Anything inside a Python process →
[Agent-Pipeline Feedback Loop Guide](AGENT_PIPELINE_LOOP_GUIDE.md).

---

## IDE-specific setup

### Cursor

Cursor provides a built-in **`browser-use`** subagent type that can navigate pages,
interact with elements, fill forms, and take screenshots. This is available in agent
mode without additional MCP configuration.

For Playwright MCP and Chrome DevTools MCP, add them to your project or user MCP
config (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["-y", "@playwright/mcp@latest", "--caps", "devtools"],
      "env": {
        "PLAYWRIGHT_MCP_CONSOLE_LEVEL": "warning"
      }
    },
    "devtools": {
      "command": "npx",
      "args": ["-y", "@chrome-devtools/mcp@latest", "--autoConnect"]
    }
  }
}
```

Console level options: `error` | `warning` | `info` | `debug` (each includes more
severe levels).

**autoConnect** (Chrome 144+) requests attachment to your running Chrome session with
your explicit approval in the browser — no need to relaunch Chrome with a debug port.
For older Chrome versions, use the explicit port approach:

```json
{
  "mcpServers": {
    "devtools": {
      "command": "npx",
      "args": ["-y", "@chrome-devtools/mcp@latest",
               "--ws-endpoint", "ws://localhost:9222"]
    }
  }
}
```

### Claude Code

Add MCP servers via the CLI:

```bash
# Playwright MCP with devtools capability
claude mcp add playwright -- npx @playwright/mcp@latest --caps devtools

# Chrome DevTools MCP (autoConnect — Chrome 144+)
claude mcp add devtools -- npx @chrome-devtools/mcp@latest --autoConnect

# Or with explicit debug port
claude mcp add devtools -- npx @chrome-devtools/mcp@latest \
  --ws-endpoint ws://localhost:9222

# Verify
claude mcp list
```

### Prerequisites (both IDEs)

```bash
# Node.js 18+ required
node --version

# Install Playwright browsers (Chromium for MCP, Firefox for E2E suite)
npx playwright install chromium
npx playwright install firefox
```

---

## Quick reference

```bash
# Launch Chrome with debugging port (live co-development)
chrome-dev   # alias from .zshrc setup above

# Run existing E2E suite (automated, Firefox)
make test-ui-e2e

# Start the GI/KG viewer dev server (for manual browsing)
make serve

# Start the Streamlit comparison tool
make run-compare
```

---

## Related documentation

| Document | Relationship |
| -------- | ------------ |
| [E2E Testing Guide](E2E_TESTING_GUIDE.md) | Playwright E2E suite details, surface map, spec conventions |
| [Testing Strategy](../architecture/TESTING_STRATEGY.md) | Where browser E2E fits in the test pyramid |
| [Development Guide](DEVELOPMENT_GUIDE.md) | GI/KG viewer dev workflow, `make serve` |
| [Server Guide](SERVER_GUIDE.md) | FastAPI `/api/*` endpoints the agent inspects |
| [E2E Surface Map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md) | Playwright automation contract — surfaces, selectors, specs |
| [UXS-001](../uxs/UXS-001-gi-kg-viewer.md) | Visual and experience contract for the viewer |
| [Polyglot Repo Guide](POLYGLOT_REPO_GUIDE.md) | Python root vs `web/gi-kg-viewer/` layout |
| [Agent-Pipeline Feedback Loop](AGENT_PIPELINE_LOOP_GUIDE.md) | Python-side companion: CI, acceptance, `--monitor`, `metrics.json` |

### External references

- [Driving vs. Debugging the Browser — Steve Kinney](https://stevekinney.com/writing/driving-vs-debugging-the-browser) — conceptual breakdown, April 2026
- [Chrome DevTools MCP vs Playwright MCP vs CLI — test-lab.ai](https://www.test-lab.ai/blog/chrome-devtools-mcp-vs-playwright-mcp-cli) — decision guide
- [Playwright MCP vs CLI — Shipyard](https://shipyard.build/blog/playwright-mcp-vs-cli/) — token efficiency tradeoffs
- [Playwright MCP official docs](https://playwright.dev/docs/getting-started-mcp)
- [Chrome DevTools MCP — GitHub](https://github.com/ChromeDevTools/devtools-mcp)
