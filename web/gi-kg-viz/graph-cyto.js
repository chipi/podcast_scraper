/**
 * Cytoscape.js graph + filters + Chart.js metrics bar.
 */
(function () {
  "use strict";

  let cy = null;
  let graphResizeObserver = null;
  let graphStageResizeObserver = null;
  let fullArt = null;
  let filterState = null;
  let lastDisplayArt = null;
  let focusNodeId = null;

  const filterElId = "filter-panel";
  const metricsElId = "metrics-panel";
  const pillsElId = "node-type-pills";
  const rawElId = "raw-json";
  const graphElId = "graph-canvas";
  const chartCanvasId = "metrics-chart";

  function mountGraphLegend() {
    if (typeof GiKgVizLegend === "undefined" || !GiKgVizLegend.mount) {
      return;
    }
    const leg = document.getElementById("graph-legend");
    if (!leg) {
      return;
    }
    GiKgVizLegend.mount(leg, {
      onLegendItemClick: applyLegendItemClick,
      activeVisualKey:
        fullArt && filterState ? filterState.legendSoloVisual : null,
    });
  }

  function applyLegendItemClick(visualKey) {
    if (!fullArt || !filterState) {
      return;
    }
    if (visualKey === "__reset__" || filterState.legendSoloVisual === visualKey) {
      const fresh = GiKgViz.defaultFilterState(fullArt);
      if (!fresh) {
        return;
      }
      filterState.allowedTypes = fresh.allowedTypes;
      filterState.hideUngroundedInsights = fresh.hideUngroundedInsights;
      filterState.legendSoloVisual = null;
      GiKgVizFilters.mount(
        document.getElementById(filterElId),
        fullArt,
        filterState,
        onFilterChange
      );
      refreshView();
      return;
    }
    const sem = GiKgViz.semanticTypeForLegendVisual(visualKey);
    if (!(sem in filterState.allowedTypes)) {
      return;
    }
    filterState.legendSoloVisual = visualKey;
    const allTypes = Object.keys(filterState.allowedTypes);
    for (let j = 0; j < allTypes.length; j++) {
      const t = allTypes[j];
      filterState.allowedTypes[t] = t === sem;
    }
    GiKgVizFilters.mount(
      document.getElementById(filterElId),
      fullArt,
      filterState,
      onFilterChange
    );
    refreshView();
  }

  function hideNodeDetailPanel() {
    const el = document.getElementById("graph-node-detail");
    if (el) {
      el.classList.add("hidden");
    }
  }

  function showNodeDetailPanelFromView(viewArt, nodeId) {
    const wrap = document.getElementById("graph-node-detail");
    const body = document.getElementById("graph-node-detail-body");
    if (!wrap || !body || !viewArt) {
      return;
    }
    const html = GiKgViz.buildNodeDetailHtml(viewArt, nodeId);
    if (!html) {
      hideNodeDetailPanel();
      return;
    }
    body.innerHTML = html;
    wrap.classList.remove("hidden");
  }

  function destroyCy() {
    if (graphResizeObserver) {
      graphResizeObserver.disconnect();
      graphResizeObserver = null;
    }
    if (graphStageResizeObserver) {
      graphStageResizeObserver.disconnect();
      graphStageResizeObserver = null;
    }
    if (cy) {
      cy.destroy();
      cy = null;
    }
  }

  function setGraphPlaceholder(text) {
    hideNodeDetailPanel();
    const el = document.getElementById(graphElId);
    el.innerHTML =
      '<p class="hint graph-placeholder">' +
      GiKgViz.escapeHtml(text) +
      "</p>";
  }

  function nodeColor(type) {
    return GiKgViz.graphNodeFill(type);
  }

  /**
   * Labels sit outside the small node shapes, mostly on the graph canvas — not on the
   * node fill. Light label text (meant for Episode/Entity blobs) was invisible on a light
   * canvas; keep one readable color per color-scheme.
   */
  function nodeLabelColor() {
    if (typeof window !== "undefined" && window.matchMedia) {
      try {
        if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
          return "#e8ecf1";
        }
      } catch (_e) {
        /* ignore */
      }
    }
    return "#1a2332";
  }

  /** Smooth fit like vis-network Fit (duration ~280ms); falls back to instant fit. */
  function cyAnimatedFit() {
    if (!cy) {
      return;
    }
    const els = cy.elements();
    if (els.length === 0) {
      return;
    }
    try {
      cy.animate({
        fit: { eles: els, padding: 24 },
        duration: 280,
      });
    } catch (_e) {
      cy.fit(els, 24);
    }
  }

  function buildCyStyle() {
    const types = [
      "Episode",
      "Insight",
      "Quote",
      "Speaker",
      "Topic",
      "Entity_person",
      "Entity_organization",
      "Podcast",
    ];
    const style = [
      {
        selector: "node",
        style: {
          label: "data(label)",
          "font-size": "9px",
          "text-wrap": "wrap",
          "text-max-width": "140px",
          "background-color": "#868e96",
          color: nodeLabelColor(),
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
        },
      });
    }
    return style;
  }

  function updateGraphFocusHint() {
    const el = document.getElementById("graph-interaction-hint");
    if (!el) {
      return;
    }
    if (focusNodeId) {
      el.textContent =
        "Showing this node and its neighbors only. Double-click the same node or empty canvas to restore the full graph. Click a node for details.";
    } else {
      el.textContent =
        "Click a node for details. Double-click a node to open its neighborhood; double-click it again or empty canvas to close.";
    }
  }

  function redrawGraph() {
    destroyCy();
    hideNodeDetailPanel();
    try {
      const displayArt = lastDisplayArt;
      if (!displayArt) {
        setGraphPlaceholder("Pick a file to render the graph.");
        updateGraphFocusHint();
        return;
      }
      if (typeof cytoscape === "undefined") {
        setGraphPlaceholder("Cytoscape failed to load (check CDN / network).");
        updateGraphFocusHint();
        return;
      }
      const viewArt = GiKgViz.filterArtifactEgoOneHop(displayArt, focusNodeId);
      const elements = GiKgViz.toCytoElements(viewArt);
      const nodeEls = elements.filter(function (x) {
        return !x.data.source;
      });
      if (nodeEls.length === 0) {
        setGraphPlaceholder("No nodes in this view (adjust filters).");
        updateGraphFocusHint();
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

      function syncCySize() {
        if (cy && el) {
          cy.resize();
        }
      }
      if (typeof ResizeObserver !== "undefined") {
        graphResizeObserver = new ResizeObserver(syncCySize);
        graphResizeObserver.observe(el);
        const stage = el.closest(".graph-main-stage");
        if (stage) {
          graphStageResizeObserver = new ResizeObserver(syncCySize);
          graphStageResizeObserver.observe(stage);
        }
      }
      requestAnimationFrame(syncCySize);
      requestAnimationFrame(function () {
        requestAnimationFrame(syncCySize);
      });

      cy.on("tap", function (evt) {
        if (!lastDisplayArt) {
          return;
        }
        const t = evt.target;
        if (t === cy) {
          hideNodeDetailPanel();
          return;
        }
        if (typeof t.isNode === "function" && t.isNode()) {
          const viewArt = GiKgViz.filterArtifactEgoOneHop(
            lastDisplayArt,
            focusNodeId
          );
          showNodeDetailPanelFromView(viewArt, t.id());
          return;
        }
        hideNodeDetailPanel();
      });

      cy.on("dblclick", function (evt) {
        if (!lastDisplayArt) {
          return;
        }
        const t = evt.target;
        if (typeof t.isNode === "function" && t.isNode()) {
          const id = t.id();
          focusNodeId = focusNodeId === id ? null : id;
          redrawGraph();
          return;
        }
        if (focusNodeId !== null) {
          focusNodeId = null;
          redrawGraph();
        }
      });

      updateGraphFocusHint();
    } finally {
      mountGraphLegend();
    }
  }

  function refreshView() {
    try {
      const metricsEl = document.getElementById(metricsElId);
      const pillsEl = document.getElementById(pillsElId);
      const rawEl = document.getElementById(rawElId);
      const chartCanvas = document.getElementById(chartCanvasId);

      if (!fullArt || !filterState) {
        lastDisplayArt = null;
        focusNodeId = null;
        GiKgViz.renderMetricsPanel(metricsEl, null);
        GiKgViz.renderNodeTypePills(pillsEl, null);
        GiKgVizCharts.destroyOn(chartCanvas);
        rawEl.textContent = "";
        rawEl.classList.add("hidden");
        updateGraphFocusHint();
        return;
      }

      const display = GiKgViz.applyGraphFilters(fullArt, filterState);
      lastDisplayArt = display;
      focusNodeId = null;
      const fa = GiKgViz.filtersActive(fullArt, filterState);
      GiKgViz.renderMetricsPanel(metricsEl, display, {
        baseline: fullArt,
        filteredNote: fa,
      });
      GiKgViz.renderNodeTypePills(pillsEl, display);
      GiKgVizCharts.renderNodeTypesBar(chartCanvas, display, {
        onTypeClick: applyChartBarFilter,
      });
      rawEl.textContent = JSON.stringify(fullArt.data, null, 2);
      rawEl.classList.remove("hidden");
      redrawGraph();
    } finally {
      mountGraphLegend();
    }
  }

  function onFilterChange() {
    refreshView();
  }

  function applyChartBarFilter(typeName) {
    if (!fullArt || !filterState) {
      return;
    }
    const semKey = GiKgViz.semanticTypeForLegendVisual(typeName);
    if (!(semKey in filterState.allowedTypes)) {
      return;
    }
    const keys = Object.keys(filterState.allowedTypes);
    const active = keys.filter(function (k) {
      return filterState.allowedTypes[k] !== false;
    });
    const alreadySolo =
      active.length === 1 &&
      active[0] === semKey &&
      (typeName === semKey || filterState.legendSoloVisual === typeName);
    if (alreadySolo) {
      filterState.legendSoloVisual = null;
      for (let i = 0; i < keys.length; i++) {
        filterState.allowedTypes[keys[i]] = true;
      }
    } else {
      filterState.legendSoloVisual = typeName !== semKey ? typeName : null;
      for (let i = 0; i < keys.length; i++) {
        const k = keys[i];
        filterState.allowedTypes[k] = k === semKey;
      }
    }
    GiKgVizFilters.mount(
      document.getElementById(filterElId),
      fullArt,
      filterState,
      onFilterChange
    );
    refreshView();
  }

  function run() {
    GiKgVizCli.inject("[data-cli-hints]");
    mountGraphLegend();

    const ndClose = document.querySelector(
      "#graph-node-detail .graph-node-detail-close"
    );
    if (ndClose && !ndClose.dataset.giKgBound) {
      ndClose.dataset.giKgBound = "1";
      ndClose.addEventListener("click", hideNodeDetailPanel);
    }

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
          lastDisplayArt = null;
          focusNodeId = null;
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
          updateGraphFocusHint();
        }
      },
      onClear: function () {},
    });

    if (btnFit) {
      btnFit.addEventListener("click", function () {
        cyAnimatedFit();
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
          cyAnimatedFit();
        }
      });
    }

    if (window.GiKgVizQuery) {
      window.GiKgVizQuery.applyVizNavLinks(document);
    }

    window.addEventListener("resize", function () {
      if (cy) {
        try {
          cy.resize();
        } catch (_e) {
          /* ignore */
        }
      }
    });

    updateGraphFocusHint();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", run);
  } else {
    run();
  }
})();
