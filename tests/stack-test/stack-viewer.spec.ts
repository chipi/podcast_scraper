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

  test("v3.0 vocabulary lands in pipeline-emitted artifacts (typed MENTIONS, insight_type, position_hint)", async ({
    request,
  }) => {
    // RFC-097 v3.0 phase-3 (3.4 UI layer) Tier-3 verification: the real
    // pipeline run that seeded /app/output is the production code path
    // for the typed-MENTIONS post-pass (workflow/metadata_generation.py),
    // the insight_type classifier (gi/insight_type_classifier.py), and
    // the position_hint waterfall (workflow → gi.build_artifact →
    // gi/position_hint.py). This test asserts those data points actually
    // land in the artifacts the viewer/queries consume.
    //
    // Defensive: the stack-test pipeline runs in airgapped_thin mode so
    // some insight emission paths may produce stub artifacts (no LLM).
    // Assertions are formulated as "if X is present THEN it must be in
    // v3.0 shape" — they fail loudly on drift but tolerate the thin
    // profile's reduced output volume.

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
    for (const gi of giContents) {
      for (const n of gi.nodes ?? []) {
        if (n.type !== "Insight") continue
        totalInsights += 1
        const it = (n.properties as Record<string, unknown> | undefined)?.[
          "insight_type"
        ] as string | undefined
        expect(it, `Insight ${n.id} insight_type`).toBeDefined()
        expect(allowedInsightTypes.has(String(it))).toBe(true)
        if (it !== "unknown") nonUnknownInsights += 1
      }
    }
    expect(totalInsights, "at least one Insight produced by pipeline").toBeGreaterThan(0)
    // The classifier ran — at least one insight has a non-"unknown" type.
    // (Tolerates a stub-fallback episode but fails if every insight is stub.)
    expect(
      nonUnknownInsights,
      "insight_type classifier must produce ≥1 non-unknown type in the corpus",
    ).toBeGreaterThan(0)

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
    const kgPersonOrgIds = new Set<string>()
    let podcastNodeCount = 0
    let hasEpisodeEdgeCount = 0
    for (const kg of kgContents) {
      for (const n of kg.nodes ?? []) {
        if (n.type === "Person" || n.type === "Organization") {
          if (typeof n.id === "string") kgPersonOrgIds.add(n.id)
        }
        if (n.type === "Podcast") podcastNodeCount += 1
      }
      for (const e of kg.edges ?? []) {
        if (e.type === "HAS_EPISODE") hasEpisodeEdgeCount += 1
      }
    }
    // KG must materialize the typed node types (chunk 3 deliverable).
    expect(
      kgPersonOrgIds.size,
      "KG must emit at least one Person or Organization node",
    ).toBeGreaterThan(0)
    // Podcast + HAS_EPISODE (chunk 3 deliverable).
    expect(podcastNodeCount, "KG must emit at least one Podcast node").toBeGreaterThan(0)
    expect(
      hasEpisodeEdgeCount,
      "KG must emit at least one HAS_EPISODE edge",
    ).toBeGreaterThan(0)

    // Now the cross-layer assertion: GI MENTIONS family edge targeting a
    // KG person/org must use the typed variant.
    let typedMentionsCount = 0
    let legacyMentionsToTypedNode = 0
    for (const gi of giContents) {
      for (const e of gi.edges ?? []) {
        const to = typeof e.to === "string" ? e.to : ""
        if (!kgPersonOrgIds.has(to)) continue
        if (e.type === "MENTIONS_PERSON" || e.type === "MENTIONS_ORG") {
          typedMentionsCount += 1
        } else if (e.type === "MENTIONS") {
          legacyMentionsToTypedNode += 1
        }
      }
    }
    // Strict: legacy MENTIONS edges pointing at typed nodes must NOT exist
    // (the post-pass rewrites them). If this fails, the post-pass didn't
    // fire — exact failure mode the 3.1 wiring closed.
    expect(
      legacyMentionsToTypedNode,
      "typed-MENTIONS post-pass must have rewritten legacy MENTIONS → typed " +
        "(any legacy MENTIONS pointing at a Person/Organization KG node " +
        "indicates the post-pass did not fire)",
    ).toBe(0)
  })
})
