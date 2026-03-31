/**
 * Chart.js horizontal bar: node counts by type (requires Chart global from CDN).
 */
(function (global) {
  "use strict";

  function themeColors() {
    const dark = global.matchMedia("(prefers-color-scheme: dark)").matches;
    return {
      text: dark ? "#e7ecf3" : "#1a2332",
      grid: dark ? "#2d3a4d" : "#cfd8e6",
    };
  }

  /**
   * @param {string} hex
   * @param {number} a
   */
  function rgbaFromHex(hex, a) {
    const h = String(hex || "").replace("#", "");
    if (h.length !== 6) {
      return "rgba(108, 179, 247," + a + ")";
    }
    const r = parseInt(h.slice(0, 2), 16);
    const g = parseInt(h.slice(2, 4), 16);
    const b = parseInt(h.slice(4, 6), 16);
    return "rgba(" + r + "," + g + "," + b + "," + a + ")";
  }

  /**
   * @param {string[]} labels
   * @returns {{ backgroundColor: string[], borderColor: string[] }}
   */
  function barPaletteForLabels(labels) {
    const V = global.GiKgViz;
    const bg = [];
    const bd = [];
    for (let i = 0; i < labels.length; i++) {
      const fill =
        V && typeof V.graphNodeFill === "function"
          ? V.graphNodeFill(labels[i])
          : "#6cb3f7";
      bg.push(rgbaFromHex(fill, 0.45));
      bd.push(rgbaFromHex(fill, 0.92));
    }
    return { backgroundColor: bg, borderColor: bd };
  }

  /**
   * @param {HTMLCanvasElement|null} canvas
   * @param {object|null} art — ParsedArtifact with nodeTypes
   * @param {{ onTypeClick?: function(string): void }|undefined} options
   */
  function renderNodeTypesBar(canvas, art, options) {
    if (!canvas) {
      return;
    }
    if (canvas._giKgChart) {
      canvas._giKgChart.destroy();
      canvas._giKgChart = null;
    }
    if (!art) {
      return;
    }
    const V = global.GiKgViz;
    const vc =
      V && typeof V.visualNodeTypeCounts === "function"
        ? V.visualNodeTypeCounts(art.data.nodes || [])
        : {};
    if (!Object.keys(vc).length) {
      return;
    }
    if (typeof Chart === "undefined") {
      return;
    }
    options = options || {};
    const onTypeClick = options.onTypeClick;
    const c = themeColors();
    const visualKeys = Object.keys(vc).sort(function (a, b) {
      return vc[b] - vc[a];
    });
    const data = visualKeys.map(function (k) {
      return vc[k];
    });
    const displayLabels = visualKeys.map(function (k) {
      return V && typeof V.graphNodeLegendLabel === "function"
        ? V.graphNodeLegendLabel(k)
        : k;
    });
    const pal = barPaletteForLabels(visualKeys);
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    const chartTitle = onTypeClick
      ? "Nodes by type — click a bar to show only that type (again to show all)"
      : "Nodes by type (current view)";
    canvas._giKgChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: displayLabels,
        datasets: [
          {
            label: "Nodes",
            data: data,
            backgroundColor: pal.backgroundColor,
            borderColor: pal.borderColor,
            borderWidth: 1,
          },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        onClick: onTypeClick
          ? function (_evt, elements, chart) {
              if (!elements || elements.length === 0 || !chart) {
                return;
              }
              const idx = elements[0].index;
              if (idx >= 0 && idx < visualKeys.length) {
                onTypeClick(String(visualKeys[idx]));
              }
            }
          : undefined,
        onHover: onTypeClick
          ? function (_evt, elements) {
              canvas.style.cursor =
                elements && elements.length > 0 ? "pointer" : "default";
            }
          : undefined,
        plugins: {
          legend: { display: false },
          title: {
            display: true,
            text: chartTitle,
            color: c.text,
            font: { size: 13 },
          },
        },
        scales: {
          x: {
            beginAtZero: true,
            ticks: { color: c.text, precision: 0 },
            grid: { color: c.grid },
          },
          y: {
            ticks: { color: c.text },
            grid: { color: c.grid },
          },
        },
      },
    });
  }

  /**
   * @param {HTMLCanvasElement|null} canvas
   */
  function destroyOn(canvas) {
    if (canvas && canvas._giKgChart) {
      canvas._giKgChart.destroy();
      canvas._giKgChart = null;
    }
  }

  global.GiKgVizCharts = {
    renderNodeTypesBar: renderNodeTypesBar,
    destroyOn: destroyOn,
  };
})(window);
