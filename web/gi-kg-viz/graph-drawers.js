/**
 * Collapsible left/right panels on graph viewer pages (vis + Cytoscape).
 */
(function () {
  "use strict";

  /**
   * @param {HTMLElement} drawer
   * @param {'left'|'right'} side
   */
  function updateEdgeUi(drawer, side) {
    const open = drawer.classList.contains("is-open");
    const btn = drawer.querySelector(".graph-drawer-edge");
    if (!btn) {
      return;
    }
    btn.setAttribute("aria-expanded", open ? "true" : "false");
    if (side === "left") {
      btn.textContent = open ? "\u2039" : "\u203a";
      btn.setAttribute(
        "aria-label",
        open ? "Collapse files panel" : "Expand files panel"
      );
    } else {
      btn.textContent = open ? "\u203a" : "\u2039";
      btn.setAttribute(
        "aria-label",
        open
          ? "Collapse right panel (overview, CLI, raw JSON)"
          : "Expand right panel (overview, CLI, raw JSON)"
      );
    }
  }

  /**
   * @param {HTMLElement} drawer
   * @param {'left'|'right'} side
   */
  function wireDrawer(drawer, side) {
    const btn = drawer.querySelector(".graph-drawer-edge");
    if (!btn) {
      return;
    }
    btn.addEventListener("click", function () {
      drawer.classList.toggle("is-open");
      updateEdgeUi(drawer, side);
      window.dispatchEvent(new Event("resize"));
    });
    updateEdgeUi(drawer, side);
  }

  function run() {
    const left = document.getElementById("graph-drawer-left");
    const right = document.getElementById("graph-drawer-right");
    if (left) {
      wireDrawer(left, "left");
    }
    if (right) {
      wireDrawer(right, "right");
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", run);
  } else {
    run();
  }
})();
