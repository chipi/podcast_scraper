/**
 * Color key for node types (palette from GiKgViz.graphNodeTypeStyles).
 * Optional: rows are clickable to drive graph filters (graph pages only).
 */
(function (global) {
  "use strict";

  /**
   * @param {HTMLElement | null} container
   * @param {{ onLegendItemClick?: function(string): void, activeVisualKey?: string|null|undefined }} [options]
   */
  function mount(container, options) {
    options = options || {};
    const onItemClick = options.onLegendItemClick;
    const activeKey = options.activeVisualKey;

    if (!container || !global.GiKgViz) {
      return;
    }
    const V = global.GiKgViz;
    const styles = V.graphNodeTypeStyles;
    const order = V.graphNodeTypesOrdered;
    if (!styles || !order) {
      container.innerHTML = "";
      const err = document.createElement("p");
      err.className = "hint";
      err.textContent =
        "Node color key unavailable (GiKgViz palette missing). Reload with shared.js.";
      container.appendChild(err);
      return;
    }
    container.innerHTML = "";
    const head = document.createElement("h3");
    head.className = "graph-legend-heading";
    head.textContent = "Node colors";
    container.appendChild(head);

    const ul = document.createElement("ul");
    ul.className = "graph-legend-list";

    /**
     * @param {string} labelText
     * @param {string} background
     * @param {string} border
     * @param {string|null} dataKey
     */
    function addRow(labelText, background, border, dataKey) {
      const li = document.createElement("li");
      li.className = "graph-legend-item";
      const clickable = typeof onItemClick === "function" && dataKey != null;
      if (clickable) {
        li.classList.add("graph-legend-item--clickable");
        if (activeKey === dataKey) {
          li.classList.add("graph-legend-item--active");
        }
        li.setAttribute("role", "button");
        li.tabIndex = 0;
        li.setAttribute("data-legend-key", dataKey);
        const activate = function () {
          onItemClick(dataKey);
        };
        li.addEventListener("click", activate);
        li.addEventListener("keydown", function (e) {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            activate();
          }
        });
      }
      const sw = document.createElement("span");
      sw.className = "graph-legend-swatch";
      sw.style.backgroundColor = background;
      sw.style.borderColor = border;
      sw.setAttribute("aria-hidden", "true");
      const lab = document.createElement("span");
      lab.className = "graph-legend-label";
      lab.textContent = labelText;
      li.appendChild(sw);
      li.appendChild(lab);
      ul.appendChild(li);
    }

    const labelFn =
      typeof V.graphNodeLegendLabel === "function"
        ? V.graphNodeLegendLabel
        : function (k) {
            return k;
          };
    for (let i = 0; i < order.length; i++) {
      const t = order[i];
      const s = styles[t];
      if (!s) {
        continue;
      }
      addRow(labelFn(t), s.background, s.border, t);
    }
    const unk = V.graphNodeUnknownFill || "#868e96";
    addRow(
      "Other / unknown",
      unk,
      "#5c636a",
      typeof onItemClick === "function" ? "__reset__" : null
    );

    container.appendChild(ul);

    const note = document.createElement("p");
    note.className = "hint graph-legend-note";
    note.textContent = onItemClick
      ? "Click a row to show only that color (click again to show all). " +
        "Other / unknown resets filters. " +
        "Edge labels show relation types (e.g. SUPPORTED_BY, HAS_INSIGHT, MENTIONS)."
      : "Edge labels show relation types (e.g. SUPPORTED_BY, HAS_INSIGHT, MENTIONS).";
    container.appendChild(note);
  }

  global.GiKgVizLegend = { mount: mount };
})(window);
