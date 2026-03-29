/**
 * File picker, artifact store, and file list. Calls back on select / clear.
 * Depends on GiKgViz (shared.js).
 */
(function () {
  "use strict";

  /**
   * @param {{ onSelect: (art: object|null) => void, onClear?: () => void }} options
   */
  function init(options) {
    const onSelect = options.onSelect;
    const onClear = options.onClear || function () {};

    const picker = document.getElementById("file-picker");
    const clearBtn = document.getElementById("clear-btn");
    const fileListEl = document.getElementById("file-list");
    const emptyHint = document.getElementById("empty-hint");

    /** @type {Map<string, object>} */
    const store = new Map();
    let activeKey = null;

    function renderList() {
      fileListEl.innerHTML = "";
      if (store.size === 0) {
        emptyHint.classList.remove("hidden");
        fileListEl.classList.add("hidden");
        return;
      }
      emptyHint.classList.add("hidden");
      fileListEl.classList.remove("hidden");

      const sorted = [...store.entries()].sort(function (a, b) {
        return a[1].name.localeCompare(b[1].name);
      });
      for (let i = 0; i < sorted.length; i++) {
        const key = sorted[i][0];
        const art = sorted[i][1];
        const li = document.createElement("li");
        const btn = document.createElement("button");
        btn.type = "button";
        btn.dataset.key = key;
        const badge = document.createElement("span");
        badge.className =
          "badge badge-" +
          (art.kind === "gi" || art.kind === "kg" ? art.kind : "unknown");
        badge.textContent = art.kind === "unknown" ? "?" : art.kind.toUpperCase();
        btn.appendChild(badge);
        btn.appendChild(document.createTextNode(art.name));
        if (key === activeKey) {
          btn.classList.add("active");
        }
        btn.addEventListener("click", function () {
          selectKey(key);
        });
        li.appendChild(btn);
        fileListEl.appendChild(li);
      }
    }

    function selectKey(key) {
      activeKey = key;
      renderList();
      const art = key ? store.get(key) : null;
      onSelect(art || null);
    }

    picker.addEventListener("change", async function () {
      const files = picker.files;
      if (!files || files.length === 0) {
        return;
      }
      let lastName = null;
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const text = await file.text();
        let data;
        try {
          data = JSON.parse(text);
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e);
          window.alert("Invalid JSON in " + file.name + ": " + msg);
          continue;
        }
        if (data === null || typeof data !== "object") {
          window.alert("Expected a JSON object in " + file.name);
          continue;
        }
        const parsed = window.GiKgViz.parseArtifact(file.name, data);
        store.set(file.name, parsed);
        lastName = file.name;
      }
      picker.value = "";
      if (lastName) {
        activeKey = lastName;
      }
      renderList();
      selectKey(activeKey);
    });

    clearBtn.addEventListener("click", function () {
      store.clear();
      activeKey = null;
      renderList();
      onClear();
      onSelect(null);
    });
  }

  window.GiKgVizShell = { init: init };
})();
