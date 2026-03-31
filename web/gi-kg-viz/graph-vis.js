/**
 * vis-network graph + filters + Chart.js metrics bar.
 */
(function () {
  "use strict";

  let network = null;
  let graphResizeObserver = null;
  let fullArt = null;
  let filterState = null;
  /** Last filtered artifact used for the graph (before 1-hop focus). */
  let lastDisplayArt = null;
  /** When set, graph shows only this node and its direct neighbors. */
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

  function destroyNetwork() {
    if (graphResizeObserver) {
      graphResizeObserver.disconnect();
      graphResizeObserver = null;
    }
    if (network) {
      network.destroy();
      network = null;
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

  function visGroups() {
    return GiKgViz.toVisNetworkGroups();
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

  function redrawNetwork() {
    destroyNetwork();
    hideNodeDetailPanel();
    try {
      const displayArt = lastDisplayArt;
      if (!displayArt) {
        setGraphPlaceholder("Pick a file to render the graph.");
        updateGraphFocusHint();
        return;
      }
      if (typeof vis === "undefined" || !vis.DataSet || !vis.Network) {
        setGraphPlaceholder("vis-network failed to load (check CDN / network).");
        updateGraphFocusHint();
        return;
      }
      const viewArt = GiKgViz.filterArtifactEgoOneHop(displayArt, focusNodeId);
      const pack = GiKgViz.toVisData(viewArt);
      if (pack.nodes.length === 0) {
        setGraphPlaceholder("No nodes in this view (adjust filters).");
        updateGraphFocusHint();
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
        interaction: {
          zoomViewOnDoubleClick: false,
        },
      };
      network = new vis.Network(el, data, options);

      network.on("click", function (params) {
        if (!lastDisplayArt) {
          return;
        }
        if (params.nodes && params.nodes.length > 0) {
          const viewArt = GiKgViz.filterArtifactEgoOneHop(
            lastDisplayArt,
            focusNodeId
          );
          showNodeDetailPanelFromView(viewArt, params.nodes[0]);
          return;
        }
        hideNodeDetailPanel();
      });

      network.on("doubleClick", function (params) {
        if (!lastDisplayArt) {
          return;
        }
        if (params.nodes && params.nodes.length > 0) {
          const id = params.nodes[0];
          focusNodeId = focusNodeId === id ? null : id;
          redrawNetwork();
          return;
        }
        if (focusNodeId !== null) {
          focusNodeId = null;
          redrawNetwork();
        }
      });

      function syncVisSize() {
        if (network && el) {
          network.setSize(el.clientWidth + "px", el.clientHeight + "px");
        }
      }
      if (typeof ResizeObserver !== "undefined") {
        graphResizeObserver = new ResizeObserver(syncVisSize);
        graphResizeObserver.observe(el);
      }
      requestAnimationFrame(syncVisSize);
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
      redrawNetwork();
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
          updateGraphFocusHint();
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

    if (window.GiKgVizQuery) {
      window.GiKgVizQuery.applyVizNavLinks(document);
    }

    updateGraphFocusHint();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", run);
  } else {
    run();
  }
})();
