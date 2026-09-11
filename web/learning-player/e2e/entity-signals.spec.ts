import { expect, test } from "@playwright/test"

import { signInIsolated } from "./helpers"

/**
 * Topic momentum badge (#1150 lineage) — the "↑ Rising" TrendMomentum badge that now LEADS the
 * topic card. It moved off EntitySignals (which is person-only now) and reads
 * `/api/app/trending?kind=topic`, showing only for genuinely-rising topics (≥1.5×). Drive it in a
 * real browser: capture the opened topic id, mock trending to return that topic rising, open the
 * card, and assert the badge. (Similar + storyline moved to the card's own chip rows / link.)
 */
test("topic entity card leads with the rising-momentum badge", async ({ page }, testInfo) => {
  await signInIsolated(page, "entity-signals", testInfo)

  let resolveId: (v: string) => void = () => {}
  const topicId = new Promise<string>((r) => {
    resolveId = r
  })

  // Capture the opened topic id from the topic-card request (not the /perspectives sub-path).
  await page.route("**/api/app/topics/*", async (route) => {
    const url = route.request().url()
    const m = url.match(/\/topics\/([^/?]+)(?:\?|$)/)
    if (m && !url.includes("/perspectives")) resolveId(decodeURIComponent(m[1]))
    await route.continue()
  })

  // The card's momentum reads /trending?kind=topic and matches the opened topic by id — mock it
  // rising (≥1.5×) with a weekly series so the badge + sparkline render, keyed to that topic.
  await page.route("**/api/app/trending*", async (route) => {
    const tid = await topicId
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        items: [
          { entity_id: tid, label: "x", velocity: 2.6, total: 40, series: [1, 2, 3, 5] },
        ],
      }),
    })
  })

  // Open an episode → Insights → click a topic chip → the topic entity card opens.
  await page.goto("/")
  await page.goto("/podcast/p05") // #1148: reach the episode via its show page (date-independent)
  await page.getByText("Index Investing Without the Myths").first().click()
  await page.getByRole("button", { name: "Insights" }).first().click()
  await page.getByTestId("kp-topic-chip").first().click()

  // The rising-momentum badge leads the card, rendered from the mocked trending row.
  await expect(page.getByTestId("ec-topic-momentum")).toBeVisible()
})
