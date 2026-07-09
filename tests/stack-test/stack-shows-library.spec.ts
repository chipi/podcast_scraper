import { expect, test } from "@playwright/test"

/**
 * UXS-015 / RFC-104 — the operator Library tab's shows-first browse, against the
 * Docker-served seeded corpus (served-corpus tier-3, companion to the mocked
 * web/gi-kg-viewer/e2e/shows-library.spec.ts). Drives Library → Shows → grid →
 * open a show → its episodes, exercising the real /api/corpus/feeds +
 * /api/corpus/episodes endpoints and PodcastCover art resolution.
 */

const STACK_TEST_CORPUS_PATH = "/app/output"

test.describe("Stack — operator Shows Library", () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript((corpus) => {
      try {
        window.localStorage.setItem("ps_corpus_path", corpus)
        window.localStorage.setItem("ps_graph_hints_seen", "1")
      } catch {
        /* private mode / quota — ignore */
      }
    }, STACK_TEST_CORPUS_PATH)
  })

  test("the seeded corpus exposes at least one show", async ({ request }) => {
    const res = await request.get(
      `/api/corpus/feeds?path=${encodeURIComponent(STACK_TEST_CORPUS_PATH)}`,
    )
    expect(res.status(), "GET /api/corpus/feeds must succeed").toBe(200)
    const body = (await res.json()) as { feeds?: Array<{ feed_id: string; episode_count: number }> }
    expect(Array.isArray(body.feeds) && body.feeds.length, "seeded corpus must have >=1 feed").toBeGreaterThan(0)
  })

  test("Shows grid renders and a show opens to its episodes", async ({ page }) => {
    await page.goto("/")
    await page
      .getByRole("navigation", { name: "Main views" })
      .getByRole("button", { name: "Library" })
      .click()

    // Opt in to Shows mode (Episodes is the default — PRD-044 OQ1).
    await page.getByTestId("library-mode-shows").click()
    await expect(page.getByTestId("shows-grid")).toBeVisible({ timeout: 10_000 })

    const cards = page.locator("[data-shows-card]")
    await expect(cards.first()).toBeVisible({ timeout: 10_000 })

    // Open the first show → it opens in the RIGHT RAIL (ShowRailPanel), not in-panel; the
    // grid stays put. Its episode list (or an explicit empty state) renders.
    await cards.first().click()
    await expect(page.getByTestId("show-rail-panel")).toBeVisible({ timeout: 10_000 })
    const firstEpisode = page.getByTestId("show-rail-episode-0")
    const empty = page.getByTestId("show-rail-empty")
    await expect(firstEpisode.or(empty).first()).toBeVisible({ timeout: 10_000 })
    await expect(page.getByTestId("shows-grid")).toBeVisible()

    // Closing the rail leaves the grid in place.
    await page.getByTestId("show-detail-rail").getByTestId("subject-rail-close").click()
    await expect(page.getByTestId("show-rail-panel")).toHaveCount(0)
    await expect(page.getByTestId("shows-grid")).toBeVisible()
  })

  test("an episode in a show opens in the subject rail with Back to the show", async ({ page }) => {
    await page.goto("/")
    await page
      .getByRole("navigation", { name: "Main views" })
      .getByRole("button", { name: "Library" })
      .click()
    await page.getByTestId("library-mode-shows").click()
    await page.locator("[data-shows-card]").first().click()

    const firstEpisode = page.getByTestId("show-rail-episode-0")
    // Only assert the rail handoff when the opened show actually has episodes.
    if (await firstEpisode.isVisible().catch(() => false)) {
      await firstEpisode.click()
      const episodeRegion = page.getByRole("region", { name: "Episode", exact: true })
      await expect(episodeRegion).toBeVisible({ timeout: 10_000 })
      // ‹ Back returns to the show rail (subject history).
      await episodeRegion.getByTestId("subject-rail-back").click()
      await expect(page.getByTestId("show-rail-panel")).toBeVisible({ timeout: 10_000 })
    }
  })
})
