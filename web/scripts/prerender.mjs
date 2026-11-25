#!/usr/bin/env node
/**
 * After `vite build`: give every public route its own HTML file so static hosts (GitHub Pages) answer 200 with the
 * right title, description and share tags before any JavaScript runs, then write 404.html, sitemap.xml and robots.txt.
 *
 *   SITE_URL=https://dhruvilgalaiya.github.io/QuantOS node scripts/prerender.mjs [dist]
 *
 * Routes come from the exported JSON (dist/data/v1), so a new instrument or US company needs no change here.
 */
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

const DIST = process.argv[2] ?? "dist";
const SITE = (process.env.SITE_URL ?? "https://dhruvilgalaiya.github.io/QuantOS").replace(/\/$/, "");
const DATA = join(DIST, "data", "v1");
const template = readFileSync(join(DIST, "index.html"), "utf8");

const read = (p) => JSON.parse(readFileSync(join(DATA, `${p}.json`), "utf8"));
const list = (dir) => (existsSync(join(DATA, dir)) ? readdirSync(join(DATA, dir)).filter((f) => f.endsWith(".json")).map((f) => f.slice(0, -5)) : []);
const esc = (s) => s.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/** path (no leading slash, "" = home) -> { title, description } */
const pages = new Map();
pages.set("", {
  title: "QuantOS: volatility and risk analytics for Indian index traders",
  description: "Volatility, option-implied ranges, market regimes and historical base rates for NIFTY 50, NIFTY BANK and SENSEX, updated after every session.",
});
pages.set("today", {
  title: "Today in Indian index markets · QuantOS",
  description: "The market in one screen: the option-implied range to expiry, the current regime, volatility and unusual moves for NIFTY 50 and NIFTY BANK.",
});
pages.set("risk", {
  title: "Risk monitor · QuantOS",
  description: "How much a portfolio could lose on a bad day, how often that estimate held, and how tightly Indian sectors move together.",
});
pages.set("status", {
  title: "Data status · QuantOS",
  description: "When QuantOS last refreshed its data, whether every job worked, and how fresh each dataset is.",
});
pages.set("us", {
  title: "Large US companies, explained · QuantOS",
  description: "Ratios, statements and price history for the largest US companies, written in plain language.",
});

const PER_SYMBOL = {
  regimes: [(n) => `${n} market regime · QuantOS`, (n) => `Bull, sideways or bear: the current ${n} regime from a three-state hidden Markov model, how long regimes last and how often the model was right.`],
  volatility: [(n) => `${n} volatility and the seller's edge · QuantOS`, (n) => `${n} realized against implied volatility, the volatility cone, a GARCH forecast and what an option seller earned per lot.`],
  states: [(n) => `${n} market state · QuantOS`, (n) => `Is ${n} calm, normal or stressed, how long each state tends to last and what usually comes next.`],
  events: [(n) => `${n} event lab · QuantOS`, (n) => `What ${n} did after past signals such as gaps, streaks and volatility spikes, with bootstrap confidence intervals.`],
  surface: [(n) => `${n} volatility surface · QuantOS`, (n) => `The ${n} implied volatility surface across strikes and expiries, fitted with SVI and SABR, with smiles and the term structure.`],
};
for (const [dir, [title, description]] of Object.entries(PER_SYMBOL)) {
  for (const sym of list(dir)) {
    const name = read(`${dir}/${sym}`).name ?? sym;
    pages.set(`${dir}/${sym}`, { title: title(name), description: description(name) });
  }
  // the bare section path (/regimes) opens NIFTY 50; give it a page too so nav links never meet a 404
  if (pages.has(`${dir}/NIFTY_50`)) pages.set(dir, pages.get(`${dir}/NIFTY_50`));
}
for (const t of list("us")) {
  const c = read(`us/${t}`);
  pages.set(`us/${t}`, {
    title: `${c.name} (${c.ticker}) explained · QuantOS`,
    description: `${c.name} in plain language: what the company does, its key ratios, statements and price history.`,
  });
}

const AUTHOR = { "@type": "Person", name: "Dhruvil Galaiya", url: "https://github.com/DhruvilGalaiya" };

/** schema.org data for search engines: the site on the home page, each other page as part of it. */
function jsonLd(path, url, title, description) {
  const site = { "@type": "WebSite", "@id": `${SITE}/#site`, name: "QuantOS", url: `${SITE}/`, description: pages.get("").description, author: AUTHOR, inLanguage: "en" };
  const data = path
    ? { "@context": "https://schema.org", "@type": "WebPage", name: title, description, url, isPartOf: site, author: AUTHOR }
    : { "@context": "https://schema.org", ...site };
  return `<script type="application/ld+json">${JSON.stringify(data).replace(/</g, "\\u003c")}</script>`;
}

function render(path, { title, description }) {
  const url = `${SITE}/${path ? `${path}/` : ""}`;
  const head = [
    `<title>${esc(title)}</title>`,
    `<meta name="description" content="${esc(description)}" />`,
    `<link rel="canonical" href="${url}" />`,
    `<meta property="og:type" content="website" />`,
    `<meta property="og:site_name" content="QuantOS" />`,
    `<meta property="og:title" content="${esc(title)}" />`,
    `<meta property="og:description" content="${esc(description)}" />`,
    `<meta property="og:url" content="${url}" />`,
    `<meta name="twitter:card" content="summary" />`,
    jsonLd(path, url, title, description),
  ].join("\n    ");
  return template
    .replace(/<title>[\s\S]*?<\/title>/, "")
    .replace(/<meta name="description"[^>]*>/, "")
    .replace("</head>", `  ${head}\n  </head>`);
}

for (const [path, meta] of pages) {
  const file = path ? join(DIST, path, "index.html") : join(DIST, "index.html");
  mkdirSync(dirname(file), { recursive: true });
  writeFileSync(file, render(path, meta));
}

// any other path: the app boots and its router shows the not-found page (or the right page for a link it knows)
writeFileSync(join(DIST, "404.html"), render("", { title: "Page not found · QuantOS", description: pages.get("").description }).replace(/<link rel="canonical"[^>]*>\n\s*/, ""));

const today = new Date().toISOString().slice(0, 10);
writeFileSync(
  join(DIST, "sitemap.xml"),
  `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${[...pages.keys()]
    .map((p) => `  <url><loc>${SITE}/${p ? `${p}/` : ""}</loc><lastmod>${today}</lastmod><changefreq>daily</changefreq></url>`)
    .join("\n")}\n</urlset>\n`,
);
writeFileSync(join(DIST, "robots.txt"), `User-agent: *\nAllow: /\n\nSitemap: ${SITE}/sitemap.xml\n`);
writeFileSync(join(DIST, ".nojekyll"), "");

console.log(`prerendered ${pages.size} routes for ${SITE}, plus 404.html, sitemap.xml, robots.txt`);
