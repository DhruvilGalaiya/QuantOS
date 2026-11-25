import { expect, type Page, test } from "@playwright/test";

const ROUTES = [
  "",
  "today/",
  "regimes/NIFTY_50/",
  "regimes/INDIA_VIX/",
  "states/NIFTY_BANK/",
  "events/NIFTY_50/",
  "volatility/NIFTY_50/",
  "surface/NIFTY_50/",
  "risk/",
  "us/",
  "us/AAPL/",
  "status/",
];

const inr = (x: number) => new Intl.NumberFormat("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(x);

/** Collect console errors and failed same-origin requests while a page loads. */
function watch(page: Page) {
  const problems: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "error") problems.push(`console: ${m.text()}`);
  });
  page.on("pageerror", (e) => problems.push(`exception: ${e.message}`));
  page.on("response", (r) => {
    if (r.status() >= 400 && r.url().startsWith("http://127.0.0.1")) problems.push(`${r.status()} ${r.url()}`);
  });
  return problems;
}

for (const route of ROUTES) {
  test(`/${route} renders cleanly @phone`, async ({ page }) => {
    const problems = watch(page);
    await page.goto(route);
    await expect(page.locator("h1").first()).toBeVisible();
    await expect(page.locator("h1").first()).not.toHaveText("");
    await expect(page).toHaveTitle(/QuantOS/);
    await expect(page.getByText(/could not be loaded|went wrong|No page here/i)).toHaveCount(0);
    // no sideways scrolling on any width
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    expect(overflow).toBeLessThanOrEqual(1);
    expect(problems).toEqual([]);
  });
}

test("Today shows the exported closing levels", async ({ page, request, baseURL }) => {
  const today = await (await request.get(`${baseURL}data/v1/today.json`)).json();
  await page.goto("today/");
  for (const i of today.indices) {
    await expect(page.getByText(inr(i.market.close)).first()).toBeVisible();
  }
});

test("Regimes headline matches the model's current regime", async ({ page, request, baseURL }) => {
  const r = await (await request.get(`${baseURL}data/v1/regimes/NIFTY_BANK.json`)).json();
  await page.goto("regimes/NIFTY_BANK/");
  await expect(page.locator("h1")).toContainText("NIFTY BANK");
  await expect(page.locator("h1")).toContainText(String(r.current.streak));
});

test("instrument tabs switch pages without a reload", async ({ page }) => {
  await page.goto("volatility/NIFTY_50/");
  await page.getByRole("link", { name: "NIFTY BANK" }).first().click();
  await expect(page).toHaveURL(/volatility\/NIFTY_BANK/);
  await expect(page.locator("h1")).toContainText("NIFTY BANK");
});

test("unknown pages show the not-found page", async ({ page }) => {
  await page.goto("no-such-page/");
  await expect(page.getByText("No page here")).toBeVisible();
});

test("lower-case symbols still load their data", async ({ page }) => {
  await page.goto("states/nifty_50/");
  await expect(page.locator("h1")).toContainText("NIFTY 50");
});
