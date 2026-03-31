/**
 * Injects CLI examples into [data-cli-hints] (text only, no HTML injection).
 */
(function (global) {
  "use strict";

  const BLOCK =
    "# Validate (strict schema)\n" +
    "python -m podcast_scraper.cli gi validate ./output/metadata --strict\n" +
    "python -m podcast_scraper.cli kg validate ./output/metadata --strict\n\n" +
    "# Export merged bundles for this viewer\n" +
    "python -m podcast_scraper.cli gi export --output-dir ./output " +
    "--format merged --out gi-bundle.json\n" +
    "python -m podcast_scraper.cli kg export --output-dir ./output " +
    "--format merged --out kg-bundle.json\n\n" +
    "# Inspect one episode\n" +
    "python -m podcast_scraper.cli gi inspect --episode-path ./output/metadata/ep1.gi.json\n" +
    "python -m podcast_scraper.cli kg inspect --episode-path ./output/metadata/ep1.kg.json\n\n" +
    "# Cross-episode explore (JSON)\n" +
    "python -m podcast_scraper.cli gi explore --output-dir ./output --format json\n";

  function inject(selector) {
    const el = document.querySelector(selector);
    if (!el) {
      return;
    }
    el.innerHTML = "";
    const p = document.createElement("p");
    p.className = "cli-intro";
    p.textContent =
      "Replace ./output with your run directory. See docs/api/CLI.md for full options.";
    el.appendChild(p);
    const pre = document.createElement("pre");
    pre.className = "cli-block";
    pre.textContent = BLOCK;
    el.appendChild(pre);
  }

  global.GiKgVizCli = { inject: inject };
})(window);
