/**
 * Graph filter UI: node types + GIL "hide ungrounded insights".
 * Mutates the passed state object; parent calls onChange(state) to refresh graph.
 */
(function (global) {
  "use strict";

  /**
   * @param {HTMLElement} container
   * @param {object|null} fullArt
   * @param {object|null} state
   * @param {function(object): void} onChange
   */
  function mount(container, fullArt, state, onChange) {
    container.innerHTML = "";
    if (!fullArt || !state) {
      container.innerHTML =
        '<p class="hint">Load a file to use graph filters.</p>';
      return;
    }

    const h = document.createElement("h3");
    h.className = "filter-heading";
    h.textContent = "Graph filters";
    container.appendChild(h);

    const types = Object.keys(state.allowedTypes).sort();
    for (let i = 0; i < types.length; i++) {
      const t = types[i];
      const safe = t.replace(/\W/g, "_");
      const id = "flt-type-" + safe + "-" + String(i);
      const wrap = document.createElement("div");
      wrap.className = "filter-row";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.id = id;
      cb.checked = state.allowedTypes[t] !== false;
      cb.addEventListener("change", function () {
        state.allowedTypes[t] = cb.checked;
        onChange(state);
      });
      const lab = document.createElement("label");
      lab.htmlFor = id;
      lab.textContent = "Show " + t;
      wrap.appendChild(cb);
      wrap.appendChild(lab);
      container.appendChild(wrap);
    }

    if (fullArt.kind === "gi") {
      const id = "flt-hide-ungrounded";
      const wrap = document.createElement("div");
      wrap.className = "filter-row filter-row-gi";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.id = id;
      cb.checked = state.hideUngroundedInsights;
      cb.addEventListener("change", function () {
        state.hideUngroundedInsights = cb.checked;
        onChange(state);
      });
      const lab = document.createElement("label");
      lab.htmlFor = id;
      lab.textContent = "Hide ungrounded insights";
      wrap.appendChild(cb);
      wrap.appendChild(lab);
      container.appendChild(wrap);
    }

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "secondary filter-reset";
    btn.textContent = "Reset filters";
    btn.addEventListener("click", function () {
      const fresh = global.GiKgViz.defaultFilterState(fullArt);
      if (!fresh) {
        return;
      }
      const next = state.allowedTypes;
      const keys = Object.keys(next);
      for (let j = 0; j < keys.length; j++) {
        delete next[keys[j]];
      }
      const fk = Object.keys(fresh.allowedTypes);
      for (let k = 0; k < fk.length; k++) {
        next[fk[k]] = fresh.allowedTypes[fk[k]];
      }
      state.hideUngroundedInsights = false;
      onChange(state);
      mount(container, fullArt, state, onChange);
    });
    container.appendChild(btn);
  }

  global.GiKgVizFilters = { mount: mount };
})(window);
