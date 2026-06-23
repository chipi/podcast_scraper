import { expect, test } from "@playwright/test"

const STACK_TEST_CORPUS_PATH = "/app/output"

test.describe("Stack smoke test", () => {
  test("Nginx serves SPA shell", async ({ page }) => {
    await page.goto("/")
    await expect(page.locator("body")).toBeVisible()
  })

  test("API health via Nginx proxy", async ({ request }) => {
    const res = await request.get("/api/health")
    expect(res.ok()).toBeTruthy()
    const body = await res.json()
    expect(body).toHaveProperty("status")
  })

  test("graph canvas mounts after navigating to Graph tab + auto-loading the corpus", async ({
    page,
    request,
  }) => {
    // Pre-seed the corpus path so the viewer's auto-graph-sync flow can
    // resolve a non-empty selection; skip the first-run gesture overlay
    // so it doesn't cover the canvas once it mounts (parity with the
    // viewer e2e helpers).
    await page.addInitScript((corpus) => {
      try {
        window.localStorage.setItem("ps_corpus_path", corpus)
        window.localStorage.setItem("ps_graph_hints_seen", "1")
      } catch {
        /* private mode / quota — ignore */
      }
    }, STACK_TEST_CORPUS_PATH)

    // Sanity: the API actually has artifacts at this corpus path. If
    // this fails the canvas test below cannot pass — surface that
    // upstream diagnosis cleanly instead of waiting 60s for the canvas.
    const artifactsRes = await request.get(
      `/api/artifacts?path=${encodeURIComponent(STACK_TEST_CORPUS_PATH)}`,
    )
    expect(artifactsRes.status(), "GET /api/artifacts must succeed").toBe(200)
    const artifactsBody = (await artifactsRes.json()) as { artifacts?: unknown[] }
    expect(
      Array.isArray(artifactsBody.artifacts) && artifactsBody.artifacts.length,
      "API must return at least one artifact",
    ).toBeGreaterThan(0)

    // Capture console errors so a silent JS failure during auto-load
    // surfaces in the test output instead of just a timeout.
    const consoleErrors: string[] = []
    page.on("console", (msg) => {
      if (msg.type() === "error") consoleErrors.push(msg.text())
    })
    page.on("pageerror", (err) => consoleErrors.push(err.message))

    await page.goto("/")
    // Default tab is Digest; click the Main views nav to switch to
    // Graph. The handler calls ``activateGraphTab()`` → triggers
    // ``syncMergedGraphFromCorpusApi`` → fetches artifacts → loads
    // selected → displayArtifact populates → GraphCanvas mounts.
    await page
      .getByRole("navigation", { name: "Main views" })
      .getByRole("button", { name: "Graph" })
      .click()

    // The Graph tab panel itself renders unconditionally; failing to
    // see it would mean the click didn't fire / the route is wrong.
    await expect(page.getByTestId("graph-tab-panel")).toBeVisible({ timeout: 10_000 })

    // The graph lens defaults to "all time" in stack-test builds
    // (compose-test overlay sets ``VITE_DEFAULT_GRAPH_LENS_DAYS=0``)
    // so static fixture dates aren't filtered out. The auto-load
    // chain triggered by activateGraphTab picks all artifacts and
    // mounts the canvas without further interaction.

    // Then wait for the canvas itself, which only mounts once
    // ``artifacts.displayArtifact`` is non-null.
    await expect(page.locator(".graph-canvas")).toBeVisible({
      timeout: 60_000,
    })

    if (consoleErrors.length) {
      // Allow noisy warnings but flag explicit JS errors so a future
      // regression in the load chain doesn't pass silently.
      const fatal = consoleErrors.filter(
        (e) =>
          !/HMR|deprecated|Vite|dmn_chk.*invalid domain|rejected for invalid domain/i.test(e) &&
          !/"notify",\s*\w+ is null/i.test(e),
      )
      expect(fatal, `console errors during canvas mount:\n${fatal.join("\n")}`).toEqual([])
    }
  })

  test("corpus runs summary lists the produced run with non-zero artifacts", async ({
    request,
  }) => {
    // ``/api/corpus/runs/summary`` always returns the list of ``run.json``
    // documents under the corpus tree (single-feed AND multi-feed). The
    // older ``/api/corpus/documents/run-summary`` only fires for multi-
    // feed batches and was returning 404 on this single-feed stack-test.
    const res = await request.get(
      `/api/corpus/runs/summary?path=${encodeURIComponent(STACK_TEST_CORPUS_PATH)}`,
    )
    const text = await res.text()
    if (res.status() !== 200) {
      throw new Error(`runs/summary ${res.status()}: ${text.slice(0, 500)}`)
    }
    const body = JSON.parse(text) as { runs?: unknown[] }
    expect(Array.isArray(body.runs)).toBe(true)
    expect((body.runs ?? []).length).toBeGreaterThan(0)
    const run = (body.runs ?? [])[0] as Record<string, unknown>
    expect(run.episodes_scraped_total).toBeGreaterThan(0)
    expect(run.gi_artifacts_generated).toBeGreaterThan(0)
    expect(run.kg_artifacts_generated).toBeGreaterThan(0)
    expect(run.errors_total).toBe(0)
  })

  test("airgapped_thin v3.0 capability set lands in pipeline-emitted artifacts", async ({
    request,
  }) => {
    // RFC-097 v3.0 phase-3 Tier-3 verification. This stack-test runs the
    // pipeline in airgapped_thin mode (transformers BART summarization,
    // spaCy NER, NO cloud LLM, NO Ollama). The profile's **capability
    // set** is a strict subset of the full v3.0 vocabulary — see the
    // capability block in ``config/profiles/airgapped_thin.yaml``.
    //
    // What airgapped_thin DOES produce (asserted strictly here):
    //   - schema_version: "3.0" on every GI artifact
    //   - Insight nodes with text from BART summary bullets
    //   - Insight.insight_type ≠ "unknown" via the rule-based classifier
    //   - Insight.position_hint numeric via waterfall step 1 (RSS duration)
    //   - KG Person nodes from speaker detection (hosts + guests)
    //   - KG Podcast node + HAS_EPISODE edges from feed metadata
    //   - Migration safety: ZERO legacy MENTIONS edges pointing at typed
    //     KG nodes (the typed-MENTIONS post-pass rewrites them all)
    //
    // What airgapped_thin does NOT produce (asserted elsewhere):
    //   - KG Organization nodes — requires LLM entity extraction. Lives
    //     in cloud_thin / cloud_balanced / local_dgx_* profiles. The
    //     Organization contract is asserted in the Python integration
    //     surface (tests/integration/server/test_cil_queries.py::
    //     test_v3_vocabulary_full_loop_*) against a hand-crafted fixture.
    //   - MENTIONS_ORG edges — depend on Organization nodes.
    //
    // This split keeps Tier-3 assertions strict on what the profile
    // genuinely produces, with no faked data and no soft assertions.
    // Profiles with richer capability sets get their own Tier-3 specs.

    interface Artifact { relative_path: string; kind: string }
    interface ArtifactsResponse { artifacts?: Artifact[] }
    interface GiNode {
      id?: string
      type?: string
      properties?: Record<string, unknown>
    }
    interface GiEdge { from?: string; to?: string; type?: string }
    interface GiArtifact {
      schema_version?: string
      nodes?: GiNode[]
      edges?: GiEdge[]
    }
    interface KgNode { id?: string; type?: string }
    interface KgArtifact { nodes?: KgNode[]; edges?: GiEdge[] }

    const listRes = await request.get(
      `/api/artifacts?path=${encodeURIComponent(STACK_TEST_CORPUS_PATH)}`,
    )
    expect(listRes.status()).toBe(200)
    const list = (await listRes.json()) as ArtifactsResponse
    const giArtifacts = (list.artifacts ?? []).filter((a) => a.kind === "gi")
    const kgArtifacts = (list.artifacts ?? []).filter((a) => a.kind === "kg")
    expect(giArtifacts.length, "expected at least one gi.json").toBeGreaterThan(0)
    expect(kgArtifacts.length, "expected at least one kg.json").toBeGreaterThan(0)

    // -- Schema versions across all GI artifacts must be 3.0 (post-chunk-9) --
    const giContents: GiArtifact[] = []
    for (const a of giArtifacts) {
      const r = await request.get(
        `/api/artifacts/${encodeURIComponent(a.relative_path)}?path=${encodeURIComponent(
          STACK_TEST_CORPUS_PATH,
        )}`,
      )
      expect(r.status()).toBe(200)
      const body = (await r.json()) as GiArtifact
      giContents.push(body)
      expect(body.schema_version, `${a.relative_path} schema_version`).toBe("3.0")
    }
    const kgContents: KgArtifact[] = []
    for (const a of kgArtifacts) {
      const r = await request.get(
        `/api/artifacts/${encodeURIComponent(a.relative_path)}?path=${encodeURIComponent(
          STACK_TEST_CORPUS_PATH,
        )}`,
      )
      expect(r.status()).toBe(200)
      kgContents.push((await r.json()) as KgArtifact)
    }

    // -- insight_type classifier (3.2): every Insight node must carry a
    //    non-empty insight_type from the enum. ``unknown`` is allowed
    //    only when the upstream insight text is empty/stub; production
    //    insights MUST have a real classified type. --
    const allowedInsightTypes = new Set([
      "claim",
      "recommendation",
      "observation",
      "question",
      "unknown",
    ])
    let totalInsights = 0
    let nonUnknownInsights = 0
    const insightTypesSeen = new Set<string>()
    for (const gi of giContents) {
      for (const n of gi.nodes ?? []) {
        if (n.type !== "Insight") continue
        totalInsights += 1
        const it = (n.properties as Record<string, unknown> | undefined)?.[
          "insight_type"
        ] as string | undefined
        expect(it, `Insight ${n.id} insight_type`).toBeDefined()
        expect(allowedInsightTypes.has(String(it))).toBe(true)
        if (it !== "unknown") {
          nonUnknownInsights += 1
          insightTypesSeen.add(String(it))
        }
      }
    }
    expect(totalInsights, "at least one Insight produced by pipeline").toBeGreaterThan(0)
    // The classifier ran — at least one insight has a non-"unknown" type.
    expect(
      nonUnknownInsights,
      "insight_type classifier must produce ≥1 non-unknown type in the corpus",
    ).toBeGreaterThan(0)
    // Stricter: if the pipeline produced ≥3 non-unknown insights, the
    // classifier must produce ≥2 distinct buckets across them (catches
    // the classifier collapsing to a single default). Below 3 we don't
    // assert diversity — too few datapoints for a meaningful claim.
    if (nonUnknownInsights >= 3) {
      expect(
        insightTypesSeen.size,
        `classifier collapse: ${nonUnknownInsights} non-unknown insights ` +
          `but only ${insightTypesSeen.size} distinct buckets seen: ` +
          `${[...insightTypesSeen].join(", ")}. Classifier may have ` +
          "defaulted everything to one type.",
      ).toBeGreaterThanOrEqual(2)
    }

    // -- position_hint waterfall (3.3): with RSS itunes:duration set
    //    (60s in the fixture), step 1 of the waterfall fires for every
    //    insight that has a supporting Quote with a timestamp. Assert
    //    that AT LEAST one insight in the corpus has a numeric
    //    position_hint within [0, 1]. --
    let numericPositionHints = 0
    for (const gi of giContents) {
      for (const n of gi.nodes ?? []) {
        if (n.type !== "Insight") continue
        const ph = (n.properties as Record<string, unknown> | undefined)?.[
          "position_hint"
        ]
        if (typeof ph === "number") {
          expect(ph).toBeGreaterThanOrEqual(0)
          expect(ph).toBeLessThanOrEqual(1)
          numericPositionHints += 1
        }
      }
    }
    expect(
      numericPositionHints,
      "position_hint waterfall must produce ≥1 numeric value (step 1 from RSS itunes:duration)",
    ).toBeGreaterThan(0)

    // -- Typed MENTIONS family (3.1): when a GI MENTIONS edge points at a
    //    Person/Organization KG node, the typed-MENTIONS post-pass MUST
    //    have rewritten its type to MENTIONS_PERSON / MENTIONS_ORG. A
    //    legacy generic "MENTIONS" pointing at a typed node would mean
    //    the post-pass didn't fire — a precise migration regression. --
    // KG capability assertions (airgapped_thin's subset):
    // - Person nodes from speaker detection — strictly required.
    // - Organization nodes — out of scope (no LLM). Asserted in the
    //   Python integration surface (test_cil_queries.py).
    // - Podcast node + HAS_EPISODE edge — required (feed metadata path).
    const kgPersonIds = new Set<string>()
    let podcastNodeCount = 0
    let hasEpisodeEdgeCount = 0
    for (const kg of kgContents) {
      for (const n of kg.nodes ?? []) {
        if (n.type === "Person") {
          if (typeof n.id === "string") kgPersonIds.add(n.id)
        }
        if (n.type === "Podcast") podcastNodeCount += 1
      }
      for (const e of kg.edges ?? []) {
        if (e.type === "HAS_EPISODE") hasEpisodeEdgeCount += 1
      }
    }
    expect(
      kgPersonIds.size,
      "airgapped_thin: KG must emit at least one Person node from speaker detection",
    ).toBeGreaterThan(0)
    expect(
      podcastNodeCount,
      "airgapped_thin: KG must emit at least one Podcast node from feed metadata",
    ).toBeGreaterThan(0)
    expect(
      hasEpisodeEdgeCount,
      "airgapped_thin: KG must emit at least one HAS_EPISODE edge",
    ).toBeGreaterThan(0)

    // Migration safety: any GI MENTIONS family edge targeting a typed KG
    // node must be the typed variant — the post-pass rewrites legacy
    // MENTIONS → MENTIONS_PERSON. A legacy MENTIONS pointing at a Person
    // KG node would mean the post-pass didn't fire. Vacuously true if no
    // legacy edges exist; meaningful as a regression guard.
    let legacyMentionsToTypedNode = 0
    for (const gi of giContents) {
      for (const e of gi.edges ?? []) {
        const to = typeof e.to === "string" ? e.to : ""
        if (!kgPersonIds.has(to)) continue
        if (e.type === "MENTIONS") legacyMentionsToTypedNode += 1
      }
    }
    expect(
      legacyMentionsToTypedNode,
      "typed-MENTIONS post-pass must have rewritten legacy MENTIONS → MENTIONS_PERSON " +
        "(any legacy MENTIONS pointing at a Person KG node indicates the post-pass did not fire)",
    ).toBe(0)
  })
})
