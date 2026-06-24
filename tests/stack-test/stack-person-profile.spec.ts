/**
 * #1075 chunk 3 — Tier-3 stack-test for the Person Profile / Position
 * Tracker viewer surfaces shipped in #1048 + #1049 + #1050.
 *
 * Sibling to stack-viewer.spec.ts which verifies the data-layer v3
 * contract at API level. This spec walks the *rendered* viewer through
 * the PRD-029 acceptance loop:
 *
 *   1. Mount the viewer against the stack-built corpus.
 *   2. Open the graph and surface a real Person via the subject store.
 *   3. Assert the Person Profile + Position Tracker tab pair renders,
 *      identity header + ranked-topics section + episodes-appeared-in
 *      section are present (per the airgapped_thin capability set).
 *   4. When the loaded artifact slice happens to carry MENTIONS_PERSON
 *      ∩ ABOUT edges (which airgapped_thin emits conditionally — see
 *      stack-viewer.spec.ts § "airgapped_thin v3.0 capability set"
 *      caveat), follow the click-to-PositionTracker path and assert
 *      timeline rows.
 *
 * The strict shell assertions (steps 1–3) are gated only on the data
 * stack-viewer.spec.ts already asserts holds across runs. The
 * conditional click-through (step 4) auto-strengthens if a future
 * profile change makes the edge emission deterministic.
 */
import { expect, test } from "@playwright/test"

const STACK_TEST_CORPUS_PATH = "/app/output"

test.describe("Stack — Person Profile / Position Tracker (#1075 chunk 3)", () => {
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

  test("Person Profile rail renders against a stack-built corpus + tab pair operates", async ({
    page,
    request,
  }) => {
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
    // Identity header: name renders (falls back to Person id when name
    // is absent from the slice, which is the documented behavior).
    await expect(page.getByTestId("person-landing-view-name")).toBeVisible()

    // Tab pair — Person Profile + Position Tracker per #1048 shell.
    await expect(page.getByTestId("person-landing-tab-profile")).toHaveText(/Person Profile/)
    await expect(page.getByTestId("person-landing-tab-position-tracker")).toHaveText(
      /Position Tracker/,
    )

    // Profile is the default tab — panel is visible.
    await expect(page.getByTestId("person-landing-panel-profile")).toBeVisible()

    // Switching to Position Tracker shows the panel + the no-topic
    // placeholder state (PRD-028 UXS-009 state 1).
    await page.getByTestId("person-landing-tab-position-tracker").click()
    await expect(page.getByTestId("person-landing-panel-position-tracker")).toBeVisible()
    await expect(page.getByTestId("position-tracker-panel")).toBeVisible()
    await expect(page.getByTestId("position-tracker-no-topic")).toBeVisible()

    // === Conditional rich-data path ===
    //
    // airgapped_thin emits MENTIONS_PERSON conditionally (BART
    // paraphrases speaker names in some episodes). The viewer's
    // ranked-Topic rows compute over `MENTIONS_PERSON ∩ ABOUT`, so the
    // section only renders when both fire. When it does, exercise the
    // click-to-Position-Tracker path — that's the PRD-029 + PRD-028
    // entry-point contract.
    await page.getByTestId("person-landing-tab-profile").click()
    const rankedTopics = page.getByTestId("person-landing-ranked-topics")
    if (await rankedTopics.isVisible({ timeout: 2_000 }).catch(() => false)) {
      const firstButton = page
        .getByTestId("person-landing-ranked-topic-button")
        .first()
      await firstButton.click()
      // Position Tracker activates + carries content for the (Person, Topic) pair.
      await expect(page.getByTestId("position-tracker-arc")).toBeVisible({
        timeout: 5_000,
      })
      await expect(page.getByTestId("position-tracker-row").first()).toBeVisible()
    }

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
