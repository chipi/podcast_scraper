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
        (e) => !/HMR|deprecated|Vite/i.test(e),
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
})
