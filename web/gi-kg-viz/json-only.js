/**
 * Metrics + Chart.js + raw JSON (offline-friendly except Chart CDN).
 */
(function () {
  "use strict";

  function run() {
    GiKgVizCli.inject("[data-cli-hints]");

    const metricsEl = document.getElementById("metrics-panel");
    const pillsEl = document.getElementById("node-type-pills");
    const rawEl = document.getElementById("raw-json");
    const chartCanvas = document.getElementById("metrics-chart");

    GiKgVizShell.init({
      onSelect: function (art) {
        GiKgViz.renderMetricsPanel(metricsEl, art);
        GiKgViz.renderNodeTypePills(pillsEl, art);
        GiKgVizCharts.renderNodeTypesBar(chartCanvas, art);
        if (art) {
          rawEl.textContent = JSON.stringify(art.data, null, 2);
          rawEl.classList.remove("hidden");
        } else {
          rawEl.textContent = "";
          rawEl.classList.add("hidden");
          GiKgVizCharts.destroyOn(chartCanvas);
        }
      },
      onClear: function () {
        GiKgVizCharts.destroyOn(chartCanvas);
      },
    });

    if (window.GiKgVizQuery) {
      window.GiKgVizQuery.applyVizNavLinks(document);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", run);
  } else {
    run();
  }
})();
