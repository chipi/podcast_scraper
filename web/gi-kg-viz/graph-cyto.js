/**
 * Cytoscape.js graph + filters + Chart.js metrics bar.
 */
(function () {
  "use strict";

  let cy = null;
  let fullArt = null;
  let filterState = null;

  const filterElId = "filter-panel";
  const metricsElId = "metrics-panel";
  const pillsElId = "node-type-pills";
  const rawElId = "raw-json";
  const graphElId = "graph-canvas";
  const chartCanvasId = "metrics-chart";

  function destroyCy() {
    if (cy) {
      cy.destroy();
      cy = null;
    }
  }

  function setGraphPlaceholder(text) {
    const el = document.getElementById(graphElId);
    el.innerHTML =
      '<p class="hint graph-placeholder">' +
      GiKgViz.escapeHtml(text) +
      "</p>";
  }

  function nodeColor(type) {
    const map = {
      Episode: "#4c6ef5",
      Insight: "#40c057",
      Quote: "#fab005",
      Speaker: "#69db7c",
      Topic: "#da77f2",
      Entity: "#9775fa",
      Podcast: "#748ffc",
    };
    return map[type] || "#868e96";
  }

  function buildCyStyle() {
    const types = [
      "Episode",
      "Insight",
      "Quote",
      "Speaker",
      "Topic",
      "Entity",
      "Podcast",
    ];
    const style = [
      {
        selector: "node",
        style: {
          label: "data(label)",
          "font-size": "9px",
          "text-wrap": "wrap",
          "text-max-width": "100px",
          "background-color": "#868e96",
          color: "#0d1117",
          width: 18,
          height: 18,
        },
      },
      {
        selector: "edge",
        style: {
          width: 1.5,
          "curve-style": "bezier",
          "target-arrow-shape": "triangle",
          "target-arrow-color": "#adb5bd",
          "line-color": "#adb5bd",
          label: "data(label)",
          "font-size": "8px",
          color: "#495057",
        },
      },
    ];
    for (let i = 0; i < types.length; i++) {
      const t = types[i];
      style.push({
        selector: 'node[type = "' + t + '"]',
        style: {
          "background-color": nodeColor(t),
          color: t === "Episode" || t === "Entity" ? "#ffffff" : "#0d1117",
        },
      });
    }
    return style;
  }

  function redrawGraph(displayArt) {
    destroyCy();
    if (!displayArt) {
      setGraphPlaceholder("Pick a file to render the graph.");
      return;
    }
    if (typeof cytoscape === "undefined") {
      setGraphPlaceholder("Cytoscape failed to load (check CDN / network).");
      return;
    }
    const elements = GiKgViz.toCytoElements(displayArt);
    const nodeEls = elements.filter(function (x) {
      return !x.data.source;
    });
    if (nodeEls.length === 0) {
      setGraphPlaceholder("No nodes in this view (adjust filters).");
      return;
    }
    const el = document.getElementById(graphElId);
    el.innerHTML = "";
    cy = cytoscape({
      container: el,
      elements: elements,
      layout: {
        name: "cose",
        padding: 24,
        nodeRepulsion: function () {
          return 450000;
        },
      },
      style: buildCyStyle(),
      wheelSensitivity: 0.35,
    });
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
    redrawGraph(display);
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
          destroyCy();
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
        if (cy) {
          cy.fit(cy.elements(), 24);
        }
      });
    }
    if (btnLayout) {
      btnLayout.addEventListener("click", function () {
        if (cy) {
          cy.layout({
            name: "cose",
            padding: 24,
            nodeRepulsion: function () {
              return 450000;
            },
          }).run();
          cy.fit(cy.elements(), 24);
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
