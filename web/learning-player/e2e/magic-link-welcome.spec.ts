import { expect, test } from "@playwright/test"

/**
 * Email magic link → new account → welcome card → name saved (#2272), through the real UI.
 *
 * The link is read from the app's own outbox, the way the delivery worker reads it
 * (`GET /internal/outbox/pending`, token-gated), so no test-only backdoor exists. The e2e API is
 * started with INTERNAL_OUTBOX_TOKEN set to E2E_OUTBOX_TOKEN (playwright.config.ts and the
 * Makefile's app-e2e container); a 401 here means a reused server predates that and needs a
 * restart.
 *
 * Each run uses a fresh address: the server throttles one link per address per minute, and a
 * NEW account is the whole point. The returning-account path is covered by the server tests and
 * the device journey (MagicLinkJourneyTests M3).
 */

const E2E_API = "http://127.0.0.1:8011"
const E2E_OUTBOX_TOKEN = "e2e-outbox-token"

type Envelope = { recipient?: { email?: string }; payload?: { link?: string } }

test("a magic-link sign-up lands on the welcome card, and the name sticks", async ({
  page,
  request,
}, testInfo) => {
  const local = `welcome-${testInfo.project.name}-${Date.now()}`
    .toLowerCase()
    .replace(/[^a-z0-9-]/g, "")
  const address = `${local}@e2e.test`

  // 1. Ask for a link from the real sign-in page.
  await page.goto("/login")
  await page.getByTestId("magic-link-button").click()
  await page.getByTestId("magic-link-input").fill(address)
  await page.getByTestId("magic-link-submit").click()
  await expect(page.getByTestId("magic-link-sent")).toBeVisible()

  // 2. Take the link from the outbox, as the delivery worker would.
  const pending = await request.get(`${E2E_API}/internal/outbox/pending?channel=email&limit=100`, {
    headers: { "X-Internal-Token": E2E_OUTBOX_TOKEN },
  })
  expect(
    pending.status(),
    "outbox refused: restart the e2e API so it gets INTERNAL_OUTBOX_TOKEN"
  ).toBe(200)
  const envelopes = ((await pending.json()) as { envelopes: Envelope[] }).envelopes
  const mine = envelopes.find((e) => e.recipient?.email === address)
  expect(mine?.payload?.link, `no sign-in envelope for ${address}`).toBeTruthy()

  // 3. Open it. The link carries the API's own host; open the same path through the app origin so
  //    the session cookie lands where the app runs, as a deployed same-origin setup does.
  const link = new URL(mine!.payload!.link!)
  await page.goto(`${link.pathname}${link.search}`)
  await expect(page).toHaveURL(/\/profile\?welcome=1$/)

  // 4. The welcome card asks for a name, pre-filled with the address's local part.
  const card = page.getByTestId("profile-welcome")
  await expect(card).toBeVisible()
  await expect(page.getByTestId("profile-welcome-input")).toHaveValue(local)
  await page.getByTestId("profile-welcome-input").fill("Ada Lovelace")
  await page.getByTestId("profile-welcome-save").click()

  await expect(card).toBeHidden()
  await expect(page).toHaveURL(/\/profile$/)
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Ada Lovelace")

  // 5. Saved on the server, not only in the page: a reload shows the name and does not re-ask.
  await page.reload()
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Ada Lovelace")
  await expect(page.getByTestId("profile-welcome")).toHaveCount(0)

  // 6. Any account can rename itself from the header afterwards.
  await page.getByTestId("profile-name-edit").click()
  await page.getByTestId("profile-name-input").fill("Ada")
  await page.getByTestId("profile-name-save").click()
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Ada")
  const me = await page.request.get("/api/app/me")
  expect((await me.json()).name).toBe("Ada")
})
