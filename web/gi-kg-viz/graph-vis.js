/**
 * vis-network graph + filters + Chart.js metrics bar.
 */
(function () {
  "use strict";

  let network = null;
  let fullArt = null;
  let filterState = null;

  const filterElId = "filter-panel";
  const metricsElId = "metrics-panel";
  const pillsElId = "node-type-pills";
  const rawElId = "raw-json";
  const graphElId = "graph-canvas";
  const chartCanvasId = "metrics-chart";

  function destroyNetwork() {
    if (network) {
      network.destroy();
      network = null;
    }
  }

  function setGraphPlaceholder(text) {
    const el = document.getElementById(graphElId);
    el.innerHTML =
      '<p class="hint graph-placeholder">' +
      GiKgViz.escapeHtml(text) +
      "</p>";
  }

  function visGroups() {
    return {
      Episode: {
        color: { background: "#4c6ef5", border: "#364fc7" },
        font: { color: "#ffffff" },
      },
      Insight: {
        color: { background: "#40c057", border: "#2f9e44" },
        font: { color: "#0d1117" },
      },
      Quote: {
        color: { background: "#fab005", border: "#e67700" },
        font: { color: "#0d1117" },
      },
      Speaker: {
        color: { background: "#69db7c", border: "#2b8a3e" },
        font: { color: "#0d1117" },
      },
      Topic: {
        color: { background: "#da77f2", border: "#862e9c" },
        font: { color: "#0d1117" },
      },
      Entity: {
        color: { background: "#9775fa", border: "#5f3dc4" },
        font: { color: "#ffffff" },
      },
      Podcast: {
        color: { background: "#748ffc", border: "#4263eb" },
        font: { color: "#ffffff" },
      },
    };
  }

  function redrawNetwork(displayArt) {
    destroyNetwork();
    if (!displayArt) {
      setGraphPlaceholder("Pick a file to render the graph.");
      return;
    }
    if (typeof vis === "undefined" || !vis.DataSet || !vis.Network) {
      setGraphPlaceholder("vis-network failed to load (check CDN / network).");
      return;
    }
    const pack = GiKgViz.toVisData(displayArt);
    if (pack.nodes.length === 0) {
      setGraphPlaceholder("No nodes in this view (adjust filters).");
      return;
    }
    const el = document.getElementById(graphElId);
    el.innerHTML = "";
    const edgeRows = pack.edges.map(function (e) {
      return {
        id: e.id,
        from: e.from,
        to: e.to,
        label: e.label || "",
        arrows: "to",
      };
    });
    const data = {
      nodes: new vis.DataSet(pack.nodes),
      edges: new vis.DataSet(edgeRows),
    };
    const options = {
      physics: {
        enabled: true,
        stabilization: { iterations: 150 },
      },
      nodes: {
        shape: "dot",
        size: 14,
        font: { size: 11, face: "system-ui, sans-serif" },
        borderWidth: 2,
      },
      edges: {
        font: { size: 9, align: "middle" },
        smooth: { type: "dynamic" },
      },
      groups: visGroups(),
    };
    network = new vis.Network(el, data, options);
  }

  function refreshView() {
    const metricsEl = document.getElementById(metricsElId);
    const pillsEl = document.getElementById(pillsElId);
    const rawEl = document.getElementById(rawElId);
    const chartCanvas = document.getElementById(chartCanvasId);

    if (!fullArt || !filterState) {
      GiKgViz.renderMetricsPanel(metricsEl, null);
      GiKgViz.renderNodeTypePills(pillsEl, null);
      GiKgVizCharts.destroyOn(chartCanvas);
      rawEl.textContent = "";
      rawEl.classList.add("hidden");
      return;
    }

    const display = GiKgViz.applyGraphFilters(fullArt, filterState);
    const fa = GiKgViz.filtersActive(fullArt, filterState);
    GiKgViz.renderMetricsPanel(metricsEl, display, {
      baseline: fullArt,
      filteredNote: fa,
    });
    GiKgViz.renderNodeTypePills(pillsEl, display);
    GiKgVizCharts.renderNodeTypesBar(chartCanvas, display);
    rawEl.textContent = JSON.stringify(fullArt.data, null, 2);
    rawEl.classList.remove("hidden");
    redrawNetwork(display);
  }

  function onFilterChange() {
    refreshView();
  }

  function run() {
    GiKgVizCli.inject("[data-cli-hints]");

    const filterEl = document.getElementById(filterElId);
    const btnFit = document.getElementById("btn-fit");
    const btnLayout = document.getElementById("btn-layout");

    GiKgVizShell.init({
      onSelect: function (art) {
        fullArt = art;
        filterState = art ? GiKgViz.defaultFilterState(art) : null;
        if (art) {
          GiKgVizFilters.mount(filterEl, fullArt, filterState, onFilterChange);
          refreshView();
        } else {
          filterEl.innerHTML =
            '<p class="hint">Load a file to use graph filters.</p>';
          destroyNetwork();
          setGraphPlaceholder("Pick a file to render the graph.");
          GiKgVizCharts.destroyOn(document.getElementById(chartCanvasId));
          GiKgViz.renderMetricsPanel(
            document.getElementById(metricsElId),
            null
          );
          GiKgViz.renderNodeTypePills(
            document.getElementById(pillsElId),
            null
          );
          const rawEl = document.getElementById(rawElId);
          rawEl.textContent = "";
          rawEl.classList.add("hidden");
        }
      },
      onClear: function () {},
    });

    if (btnFit) {
      btnFit.addEventListener("click", function () {
        if (network) {
          try {
            network.fit({
              animation: {
                duration: 280,
                easingFunction: "easeInOutQuad",
              },
            });
          } catch (_e) {
            network.fit();
          }
        }
      });
    }
    if (btnLayout) {
      btnLayout.addEventListener("click", function () {
        if (network) {
          network.stabilize(200);
        }
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", run);
  } else {
    run();
  }
})();
