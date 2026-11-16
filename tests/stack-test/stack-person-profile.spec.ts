/**
 * #1075 chunk 3 — Tier-3 stack-test for the Person Profile viewer surface
 * shipped in #1048 + #1049 + #1050.
 *
 * Sibling to stack-viewer.spec.ts which verifies the data-layer v3
 * contract at API level. This spec walks the *rendered* viewer through
 * the PRD-029 acceptance loop:
 *
 *   1. Mount the viewer against the stack-built corpus.
 *   2. Open the graph and surface a real Person via the subject store.
 *   3. Assert the person rail renders with its Details tab and the profile
 *      panel (per the airgapped_thin capability set).
 *
 * The Positions tab and its click-through to the position arc left public's
 * viewer with E04 step 8b; a build that adds them tests them.
 */
import { expect, test } from "@playwright/test"

const STACK_TEST_CORPUS_PATH = "/app/output"

test.describe("Stack — Person Profile (#1075 chunk 3)", () => {
  test.beforeEach(async ({ page }) => {
    // Pre-seed the corpus path so the auto-graph-sync chain resolves a
    // non-empty selection; skip the first-run gesture overlay so any
    // post-mount click lands on the canvas. Mirrors stack-viewer.spec.ts.
    await page.addInitScript((corpus) => {
      try {
        window.localStorage.setItem("ps_corpus_path", corpus)
        window.localStorage.setItem("ps_graph_hints_seen", "1")
      } catch {
        /* private mode / quota — ignore */
      }
    }, STACK_TEST_CORPUS_PATH)
  })

  test("Person Profile rail renders against a stack-built corpus", async ({
    page,
    request,
  }) => {
    test.skip(
      process.env.STACK_HAS_STAGES !== "1",
      "GI and speaker detection are an installed extension's; public's stack runs public "
        + "images, which write neither (E04 gate 3). Set STACK_HAS_STAGES=1 where it is installed.",
    )
    // Sanity: at least one KG artifact exists. The Person nodes the
    // stack-built airgapped_thin profile emits come from speaker
    // detection and live in kg.json.
    const artifactsRes = await request.get(
      `/api/artifacts?path=${encodeURIComponent(STACK_TEST_CORPUS_PATH)}`,
    )
    expect(artifactsRes.status()).toBe(200)
    type ArtifactRow = { relative_path: string; kind: string }
    const artifactsBody = (await artifactsRes.json()) as { artifacts?: ArtifactRow[] }
    const kgArtifacts = (artifactsBody.artifacts ?? []).filter((a) => a.kind === "kg")
    expect(kgArtifacts.length, "expected at least one kg.json").toBeGreaterThan(0)

    // Pull one KG artifact and find a Person id to focus. The
    // PersonLandingView's subject store reads `personId` directly, so
    // we drive via the DEV-gated `__GIKG_SUBJECT__` window hook —
    // sidesteps the Explore top-speakers UI entry point (which only
    // populates when the explore API has speaker rollups).
    const kgRes = await request.get(
      `/api/artifacts/${encodeURIComponent(kgArtifacts[0].relative_path)}?path=${encodeURIComponent(
        STACK_TEST_CORPUS_PATH,
      )}`,
    )
    expect(kgRes.status()).toBe(200)
    type KgNode = { id?: string; type?: string }
    const kgBody = (await kgRes.json()) as { nodes?: KgNode[] }
    const personNode = (kgBody.nodes ?? []).find((n) => n.type === "Person")
    expect(
      personNode?.id,
      "stack-built KG artifact must contain at least one Person node " +
        "(airgapped_thin emits these from speaker detection)",
    ).toBeTruthy()
    const personId = String(personNode!.id)

    // Capture console errors so a silent JS failure surfaces.
    const consoleErrors: string[] = []
    page.on("console", (msg) => {
      if (msg.type() === "error") consoleErrors.push(msg.text())
    })
    page.on("pageerror", (err) => consoleErrors.push(err.message))

    await page.goto("/")
    // Land on Graph tab + wait for the canvas to mount (same trigger
    // chain as stack-viewer.spec.ts).
    await page
      .getByRole("navigation", { name: "Main views" })
      .getByRole("button", { name: "Graph" })
      .click()
    await expect(page.getByTestId("graph-tab-panel")).toBeVisible({ timeout: 10_000 })
    await expect(page.locator(".graph-canvas")).toBeVisible({ timeout: 60_000 })

    // Drive the subject store via the DEV-gated window hook so we
    // surface a known Person id regardless of which Explore data the
    // backend happens to return.
    await page.evaluate((pid) => {
      const win = window as unknown as {
        __GIKG_SUBJECT__?: { focusPerson?: (id: string) => void }
      }
      const hook = win.__GIKG_SUBJECT__
      if (hook && typeof hook.focusPerson === "function") {
        hook.focusPerson(pid)
      } else {
        throw new Error("__GIKG_SUBJECT__.focusPerson hook unavailable")
      }
    }, personId)

    // === Strict shell assertions — what airgapped_thin always emits ===

    const view = page.getByTestId("person-landing-view")
    await expect(view).toBeVisible({ timeout: 10_000 })
    // Post node-view fold: the standalone Person rail is retired — PLV is
    // embedded in NodeDetail's rail, which owns the header + tabs (PLV's own
    // name header + internal tablist are hidden via `v-if="!embedded"`).
    const rail = page.getByTestId("graph-node-detail-rail")
    // Identity header: the rail titles it "Person" (an off-slice `person:` id
    // resolves via NodeDetail's inferredKindFromId).
    await expect(rail.getByRole("heading").first()).toContainText("Person")

    // NodeDetail rail: the Details tab and the profile panel are public's.
    await expect(page.getByTestId("node-detail-rail-tab-details")).toHaveText(/Details/)
    await expect(page.getByTestId("person-landing-panel-profile")).toBeVisible()

    if (consoleErrors.length) {
      // Allow benign warnings; flag fatal errors. Same noise filter as
      // stack-viewer.spec.ts.
      const fatal = consoleErrors.filter(
        (e) =>
          !/HMR|deprecated|Vite|dmn_chk.*invalid domain|rejected for invalid domain/i.test(
            e,
          ) && !/"notify",\s*\w+ is null/i.test(e),
      )
      expect(
        fatal,
        `console errors during Person Profile walk:\n${fatal.join("\n")}`,
      ).toEqual([])
    }
  })
})
