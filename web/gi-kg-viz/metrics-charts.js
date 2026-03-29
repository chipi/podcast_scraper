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
   * @param {HTMLCanvasElement|null} canvas
   * @param {object|null} art — ParsedArtifact with nodeTypes
   */
  function renderNodeTypesBar(canvas, art) {
    if (!canvas) {
      return;
    }
    if (canvas._giKgChart) {
      canvas._giKgChart.destroy();
      canvas._giKgChart = null;
    }
    if (!art || !Object.keys(art.nodeTypes).length) {
      return;
    }
    if (typeof Chart === "undefined") {
      return;
    }
    const c = themeColors();
    const labels = Object.keys(art.nodeTypes).sort(function (a, b) {
      return art.nodeTypes[b] - art.nodeTypes[a];
    });
    const data = labels.map(function (l) {
      return art.nodeTypes[l];
    });
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    canvas._giKgChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Nodes",
            data: data,
            backgroundColor: "rgba(108, 179, 247, 0.5)",
            borderColor: "rgba(108, 179, 247, 0.95)",
            borderWidth: 1,
          },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          title: {
            display: true,
            text: "Nodes by type (current view)",
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
