/**
 * File picker, folder picker, artifact store, file list, URL prefs.
 * Depends on GiKgViz (shared.js). Optional GiKgVizQuery (viz-query.js) for ?layer=&merged=.
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

    const urlPrefs =
      window.GiKgVizQuery != null
        ? window.GiKgVizQuery.getVizQueryPrefs()
        : { layer: "both", merged: false };

    /** @type {Map<string, object>} */
    const store = new Map();
    let activeKey = null;
    /** @type {'gi'|'kg'|'both'|null} */
    let mergedMode = null;
    const layerNorm =
      urlPrefs.layer === "gi" || urlPrefs.layer === "kg" ? urlPrefs.layer : "both";
    /** @type {'both'|'gi'|'kg'} */
    let layerMode = layerNorm;

    const loadSection = document.getElementById("viz-load-section");
    const panelSection =
      loadSection || picker.closest("section") || picker.parentElement;

    const urlDataBanner = document.createElement("div");
    urlDataBanner.id = "viz-url-data-banner";
    urlDataBanner.className = "viz-url-data-banner hidden";
    urlDataBanner.setAttribute("role", "status");
    panelSection.insertBefore(urlDataBanner, panelSection.firstChild);

    /**
     * @param {'ok'|'warn'} kind
     * @param {string} message
     */
    function showUrlDataBanner(kind, message) {
      urlDataBanner.className =
        "viz-url-data-banner viz-url-data-banner--" + kind;
      urlDataBanner.textContent = message;
      urlDataBanner.classList.remove("hidden");
    }

    function hideUrlDataBanner() {
      urlDataBanner.className = "viz-url-data-banner hidden";
      urlDataBanner.textContent = "";
    }

    const folderPanel = document.createElement("div");
    folderPanel.className = "viz-folder-panel";
    const titleRow = document.createElement("div");
    titleRow.className = "viz-folder-title-row";
    const folderHeading = document.createElement("span");
    folderHeading.className = "file-label";
    folderHeading.textContent = "Load from folder (recursive)";
    titleRow.appendChild(folderHeading);
    const folderHelpAnchor = document.createElement("div");
    folderHelpAnchor.className = "viz-help-anchor";
    const folderHelpBtn = document.createElement("button");
    folderHelpBtn.type = "button";
    folderHelpBtn.className = "viz-help-trigger";
    folderHelpBtn.textContent = "?";
    folderHelpBtn.setAttribute(
      "aria-label",
      "Help: URL ?data= loads JSON from a repo path via the dev server; it does not open " +
        "a disk folder. Use Choose folder to pick files on disk."
    );
    const folderHelpPop = document.createElement("div");
    folderHelpPop.className = "viz-help-popover";
    folderHelpPop.id = "viz-folder-help-popover";
    folderHelpPop.setAttribute("role", "tooltip");
    folderHelpPop.innerHTML =
      "<strong>?data= in the URL</strong> tells the dev server to <em>fetch</em> JSON from a " +
      "repo path — it does <strong>not</strong> open Finder or the disk folder picker. " +
      "Use <strong>Choose folder…</strong> to pick a folder on disk. " +
      "With <code>make serve-gi-kg-viz</code>, <code>?data=.test_outputs/…/metadata</code> " +
      "auto-loads matching <code>*.gi.json</code> / <code>*.kg.json</code> into the list below.";
    folderHelpAnchor.appendChild(folderHelpBtn);
    titleRow.appendChild(folderHelpAnchor);
    document.body.appendChild(folderHelpPop);

    let folderHelpHideTimer = null;
    let folderHelpVisible = false;

    function positionFolderHelpPop() {
      if (!folderHelpVisible) {
        return;
      }
      const rect = folderHelpBtn.getBoundingClientRect();
      const gap = 4;
      const margin = 8;
      const w = folderHelpPop.offsetWidth;
      const h = folderHelpPop.offsetHeight;
      let left = rect.left;
      let top = rect.bottom + gap;
      if (top + h > window.innerHeight - margin && rect.top > h + gap + margin) {
        top = rect.top - h - gap;
      }
      if (left + w > window.innerWidth - margin) {
        left = window.innerWidth - w - margin;
      }
      if (left < margin) {
        left = margin;
      }
      folderHelpPop.style.left = left + "px";
      folderHelpPop.style.top = top + "px";
    }

    function showFolderHelpPop() {
      folderHelpVisible = true;
      folderHelpPop.style.visibility = "hidden";
      folderHelpPop.classList.add("is-open");
      requestAnimationFrame(function () {
        positionFolderHelpPop();
        folderHelpPop.style.visibility = "";
      });
    }

    function hideFolderHelpPop() {
      folderHelpVisible = false;
      folderHelpPop.classList.remove("is-open");
      folderHelpPop.style.left = "";
      folderHelpPop.style.top = "";
      folderHelpPop.style.visibility = "";
    }

    function cancelFolderHelpHide() {
      if (folderHelpHideTimer != null) {
        clearTimeout(folderHelpHideTimer);
        folderHelpHideTimer = null;
      }
    }

    function scheduleFolderHelpHide() {
      cancelFolderHelpHide();
      folderHelpHideTimer = setTimeout(function () {
        folderHelpHideTimer = null;
        hideFolderHelpPop();
      }, 100);
    }

    function onFolderHelpScrollOrResize() {
      if (folderHelpVisible) {
        positionFolderHelpPop();
      }
    }

    folderHelpBtn.addEventListener("mouseenter", function () {
      cancelFolderHelpHide();
      showFolderHelpPop();
    });
    folderHelpBtn.addEventListener("mouseleave", function () {
      scheduleFolderHelpHide();
    });
    folderHelpPop.addEventListener("mouseenter", cancelFolderHelpHide);
    folderHelpPop.addEventListener("mouseleave", scheduleFolderHelpHide);
    folderHelpBtn.addEventListener("focus", function () {
      cancelFolderHelpHide();
      showFolderHelpPop();
    });
    folderHelpBtn.addEventListener("blur", function () {
      scheduleFolderHelpHide();
    });
    window.addEventListener("scroll", onFolderHelpScrollOrResize, true);
    window.addEventListener("resize", onFolderHelpScrollOrResize);
    folderPanel.appendChild(titleRow);

    const layerRow = document.createElement("div");
    layerRow.className = "viz-folder-layer";
    const layerLabel = document.createElement("span");
    layerLabel.className = "file-label viz-folder-layer-label";
    layerLabel.textContent = "GI / KG:";
    layerRow.appendChild(layerLabel);
    const layerBtnRow = document.createElement("div");
    layerBtnRow.className = "viz-list-filter-btns";
    layerBtnRow.setAttribute("role", "group");
    layerBtnRow.setAttribute(
      "aria-label",
      "GI only, KG only, or both — applies to folder load, URL load, file list, and merge"
    );
    /** @type {Record<string, HTMLButtonElement>} */
    const layerModeBtns = {};
    const layerChoices = [
      { v: "both", label: "Both" },
      { v: "gi", label: "GI only" },
      { v: "kg", label: "KG only" },
    ];
    for (let li = 0; li < layerChoices.length; li++) {
      const lc = layerChoices[li];
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "secondary";
      btn.textContent = lc.label;
      btn.addEventListener("click", function () {
        layerMode = lc.v === "gi" || lc.v === "kg" ? lc.v : "both";
        syncLayerModeButtons();
        syncPrefsToUrlAndNav();
        onLayerOrNameFilterChange();
      });
      layerModeBtns[lc.v] = btn;
      layerBtnRow.appendChild(btn);
    }
    layerRow.appendChild(layerBtnRow);
    folderPanel.appendChild(layerRow);

    function syncLayerModeButtons() {
      const lk = Object.keys(layerModeBtns);
      for (let i = 0; i < lk.length; i++) {
        const k = lk[i];
        const b = layerModeBtns[k];
        const on = layerMode === k;
        b.classList.toggle("active", on);
        b.setAttribute("aria-pressed", on ? "true" : "false");
      }
    }
    syncLayerModeButtons();

    const mergePrefLbl = document.createElement("label");
    mergePrefLbl.className = "viz-folder-merge";
    const mergePrefCb = document.createElement("input");
    mergePrefCb.type = "checkbox";
    mergePrefCb.id = "viz-pref-merged";
    mergePrefCb.checked = !!urlPrefs.merged;
    mergePrefLbl.appendChild(mergePrefCb);
    mergePrefLbl.appendChild(
      document.createTextNode(
        " After load, auto-merge when possible (GI+KG in Both mode if ≥1 each; else all GI or all KG)"
      )
    );
    folderPanel.appendChild(mergePrefLbl);

    const folderBtnRow = document.createElement("div");
    folderBtnRow.className = "viz-folder-actions";
    const folderBtn = document.createElement("button");
    folderBtn.type = "button";
    folderBtn.className = "secondary";
    folderBtn.id = "viz-folder-btn";
    folderBtn.textContent = "Choose folder…";
    const folderInput = document.createElement("input");
    folderInput.type = "file";
    folderInput.id = "viz-folder-input";
    folderInput.multiple = true;
    folderInput.setAttribute("webkitdirectory", "");
    folderInput.setAttribute("directory", "");
    folderInput.className = "viz-folder-input-hidden";
    folderBtnRow.appendChild(folderBtn);
    folderBtnRow.appendChild(folderInput);
    folderPanel.appendChild(folderBtnRow);

    panelSection.appendChild(folderPanel);

    function syncPrefsToUrlAndNav() {
      if (window.GiKgVizQuery == null) {
        return;
      }
      window.GiKgVizQuery.replaceUrlQuery({
        layer: layerMode,
        merged: mergePrefCb.checked,
      });
      window.GiKgVizQuery.applyVizNavLinks(document);
    }

    mergePrefCb.addEventListener("change", syncPrefsToUrlAndNav);

    function filterArtifactPaths(paths, layer) {
      const out = [];
      for (let i = 0; i < paths.length; i++) {
        const p = paths[i];
        const n = (p || "").toLowerCase();
        if (!n.endsWith(".json")) {
          continue;
        }
        if (layer === "gi") {
          if (n.endsWith(".gi.json")) {
            out.push(p);
          }
        } else if (layer === "kg") {
          if (n.endsWith(".kg.json")) {
            out.push(p);
          }
        } else if (n.endsWith(".gi.json") || n.endsWith(".kg.json")) {
          out.push(p);
        }
      }
      out.sort();
      return out;
    }

    function repoFileUrl(relPath) {
      const segs = relPath.split("/").map(function (s) {
        return encodeURIComponent(s);
      });
      return "/_repo/" + segs.join("/");
    }

    function filterArtifactFiles(files, layer) {
      const out = [];
      for (let i = 0; i < files.length; i++) {
        const f = files[i];
        const n = (f.name || "").toLowerCase();
        if (!n.endsWith(".json")) {
          continue;
        }
        if (layer === "gi") {
          if (n.endsWith(".gi.json")) {
            out.push(f);
          }
        } else if (layer === "kg") {
          if (n.endsWith(".kg.json")) {
            out.push(f);
          }
        } else if (n.endsWith(".gi.json") || n.endsWith(".kg.json")) {
          out.push(f);
        }
      }
      out.sort(function (a, b) {
        const pa = a.webkitRelativePath || a.name;
        const pb = b.webkitRelativePath || b.name;
        return pa.localeCompare(pb);
      });
      return out;
    }

    /**
     * @param {FileSystemDirectoryHandle} dirHandle
     * @param {File[]} out
     */
    async function collectFromDirHandle(dirHandle, out) {
      for await (const entry of dirHandle.values()) {
        if (entry.kind === "file") {
          out.push(await entry.getFile());
        } else if (entry.kind === "directory") {
          await collectFromDirHandle(entry, out);
        }
      }
    }

    async function processFolderFileList(files, layer) {
      const filtered = filterArtifactFiles(files, layer);
      if (filtered.length === 0) {
        window.alert(
          "No matching .gi.json / .kg.json files in that folder (including subfolders)."
        );
        return;
      }
      store.clear();
      mergedMode = null;
      activeKey = null;
      let lastKey = null;
      for (let i = 0; i < filtered.length; i++) {
        const file = filtered[i];
        const text = await file.text();
        let data;
        try {
          data = JSON.parse(text);
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e);
          window.alert(
            "Invalid JSON in " + (file.webkitRelativePath || file.name) + ": " + msg
          );
          continue;
        }
        if (data === null || typeof data !== "object") {
          window.alert(
            "Expected a JSON object in " + (file.webkitRelativePath || file.name)
          );
          continue;
        }
        const parsed = window.GiKgViz.parseArtifact(file.name, data);
        const key = file.webkitRelativePath || file.name;
        store.set(key, parsed);
        lastKey = key;
      }
      if (!lastKey) {
        renderList();
        onSelect(null);
        return;
      }
      renderList();
      const doMerge = mergePrefCb.checked;
      if (doMerge) {
        if (
          layer === "both" &&
          countByKind("gi") >= 1 &&
          countByKind("kg") >= 1
        ) {
          selectMergedBoth();
          return;
        }
        if (
          countByKind("gi") >= 2 &&
          (layer === "gi" || layer === "both")
        ) {
          selectMerged("gi");
          return;
        }
        if (
          countByKind("kg") >= 2 &&
          (layer === "kg" || layer === "both")
        ) {
          selectMerged("kg");
          return;
        }
      }
      selectKey(lastKey);
    }

    folderBtn.addEventListener("click", async function () {
      if (typeof window.showDirectoryPicker === "function") {
        try {
          const dirHandle = await window.showDirectoryPicker();
          const acc = [];
          await collectFromDirHandle(dirHandle, acc);
          await processFolderFileList(acc, layerMode);
        } catch (e) {
          if (e && e.name === "AbortError") {
            return;
          }
          folderInput.click();
        }
      } else {
        folderInput.click();
      }
    });

    folderInput.addEventListener("change", async function () {
      const fl = folderInput.files;
      if (!fl || fl.length === 0) {
        return;
      }
      await processFolderFileList(Array.from(fl), layerMode);
      folderInput.value = "";
    });

    const multiWrap = document.createElement("div");
    multiWrap.id = "viz-multi-controls";
    multiWrap.className = "viz-multi-controls hidden";
    multiWrap.setAttribute("aria-label", "Multi-file merge");
    const multiHint = document.createElement("p");
    multiHint.className = "hint viz-multi-hint";
    multiHint.innerHTML =
      "The graph uses <strong>one</strong> highlighted file. Use <strong>GI / KG</strong> above " +
      "to show GI only, KG only, or <strong>Both</strong> (default). " +
      "<strong>Name filter</strong> narrows the list. " +
      "The merge button combines <em>visible</em> files (GI+KG when both kinds exist, else all GI or all KG).";
    const listFilterRow = document.createElement("div");
    listFilterRow.className = "viz-list-filter-row";
    const nameFilterLbl = document.createElement("label");
    nameFilterLbl.className = "viz-list-name-filter";
    nameFilterLbl.appendChild(document.createTextNode("Name filter "));
    const listNameInput = document.createElement("input");
    listNameInput.type = "search";
    listNameInput.id = "viz-list-name-filter";
    listNameInput.placeholder = "substring…";
    listNameInput.setAttribute("aria-label", "Filter file list by name substring");
    nameFilterLbl.appendChild(listNameInput);
    listFilterRow.appendChild(nameFilterLbl);

    const btnRow = document.createElement("div");
    btnRow.className = "viz-merge-btns";
    const mergeVisibleBtn = document.createElement("button");
    mergeVisibleBtn.type = "button";
    mergeVisibleBtn.className = "secondary hidden";
    mergeVisibleBtn.setAttribute("aria-pressed", "false");
    mergeVisibleBtn.setAttribute(
      "aria-label",
      "Merge visible files into one graph (GI+KG, all GI, or all KG)"
    );
    btnRow.appendChild(mergeVisibleBtn);
    multiWrap.appendChild(multiHint);
    multiWrap.appendChild(listFilterRow);
    multiWrap.appendChild(btnRow);
    fileListEl.parentNode.insertBefore(multiWrap, fileListEl.nextSibling);

    function entryPassesFilters(key, art) {
      const layer = layerMode;
      if (layer === "gi" && art.kind !== "gi") {
        return false;
      }
      if (layer === "kg" && art.kind !== "kg") {
        return false;
      }
      const q = listNameInput.value.trim().toLowerCase();
      if (q) {
        const displayName =
          key.indexOf("/") >= 0 || key.indexOf("\\") >= 0 ? key : art.name;
        if (displayName.toLowerCase().indexOf(q) === -1) {
          return false;
        }
      }
      return true;
    }

    function firstStoreKeyForKind(kind) {
      const sorted = [...store.entries()].sort(function (a, b) {
        return a[1].name.localeCompare(b[1].name);
      });
      for (let i = 0; i < sorted.length; i++) {
        const key = sorted[i][0];
        const art = sorted[i][1];
        if (art.kind === kind && entryPassesFilters(key, art)) {
          return key;
        }
      }
      return null;
    }

    function firstVisibleKeyAny() {
      const sorted = [...store.entries()].sort(function (a, b) {
        return a[1].name.localeCompare(b[1].name);
      });
      for (let i = 0; i < sorted.length; i++) {
        const key = sorted[i][0];
        const art = sorted[i][1];
        if (entryPassesFilters(key, art)) {
          return key;
        }
      }
      return null;
    }

    function countByKind(kind) {
      let n = 0;
      for (const [key, a] of store.entries()) {
        if (a.kind === kind && entryPassesFilters(key, a)) {
          n += 1;
        }
      }
      return n;
    }

    /**
     * @param {'gi'|'kg'} kind
     * @returns {object[]}
     */
    function collectArts(kind) {
      const arr = [];
      for (const [key, a] of store.entries()) {
        if (a.kind === kind && entryPassesFilters(key, a)) {
          arr.push(a);
        }
      }
      arr.sort(function (x, y) {
        return x.name.localeCompare(y.name);
      });
      return arr;
    }

    function computeMergeAction() {
      const nGi = countByKind("gi");
      const nKg = countByKind("kg");
      if (layerMode === "both" && nGi >= 1 && nKg >= 1) {
        return "both";
      }
      if (nGi >= 2 && (layerMode === "gi" || layerMode === "both")) {
        return "gi";
      }
      if (nKg >= 2 && (layerMode === "kg" || layerMode === "both")) {
        return "kg";
      }
      return null;
    }

    function updateMergeUi() {
      const nGi = countByKind("gi");
      const nKg = countByKind("kg");
      const show = store.size >= 1;
      multiWrap.classList.toggle("hidden", !show);
      const action = computeMergeAction();
      if (action == null) {
        mergeVisibleBtn.classList.add("hidden");
      } else {
        mergeVisibleBtn.classList.remove("hidden");
        if (action === "both") {
          mergeVisibleBtn.textContent =
            "Merge GI + KG (" + nGi + " GI · " + nKg + " KG)";
        } else if (action === "gi") {
          mergeVisibleBtn.textContent = "Merge all GI (" + nGi + " files)";
        } else {
          mergeVisibleBtn.textContent = "Merge all KG (" + nKg + " files)";
        }
        mergeVisibleBtn.classList.toggle("active", mergedMode === action);
        mergeVisibleBtn.setAttribute(
          "aria-pressed",
          mergedMode === action ? "true" : "false"
        );
      }
    }

    function selectMerged(kind) {
      if (mergedMode === kind) {
        const k = firstStoreKeyForKind(kind);
        if (k) {
          selectKey(k);
        }
        return;
      }
      const arr = collectArts(kind);
      if (arr.length < 2) {
        return;
      }
      const merged = window.GiKgViz.mergeParsedArtifacts(arr);
      if (!merged) {
        return;
      }
      mergedMode = kind;
      activeKey = "__merged_" + kind + "__";
      renderList();
      updateMergeUi();
      onSelect(merged);
    }

    function selectMergedBoth() {
      if (mergedMode === "both") {
        const k = firstVisibleKeyAny();
        if (k) {
          selectKey(k);
        } else {
          mergedMode = null;
          activeKey = null;
          renderList();
          updateMergeUi();
          onSelect(null);
        }
        return;
      }
      const gis = collectArts("gi");
      const kgs = collectArts("kg");
      if (gis.length < 1 || kgs.length < 1) {
        return;
      }
      const merged = window.GiKgViz.mergeGiKgFromArtifactArrays(gis, kgs);
      if (!merged) {
        return;
      }
      mergedMode = "both";
      activeKey = "__merged_both__";
      renderList();
      updateMergeUi();
      onSelect(merged);
    }

    function onLayerOrNameFilterChange() {
      if (mergedMode === "gi" && countByKind("gi") < 2) {
        mergedMode = null;
        activeKey = firstVisibleKeyAny();
      } else if (mergedMode === "kg" && countByKind("kg") < 2) {
        mergedMode = null;
        activeKey = firstVisibleKeyAny();
      } else if (
        mergedMode === "both" &&
        (countByKind("gi") < 1 || countByKind("kg") < 1)
      ) {
        mergedMode = null;
        activeKey = firstVisibleKeyAny();
      } else if (
        activeKey &&
        !activeKey.startsWith("__merged_") &&
        store.has(activeKey)
      ) {
        const cur = store.get(activeKey);
        if (!entryPassesFilters(activeKey, cur)) {
          mergedMode = null;
          activeKey = firstVisibleKeyAny();
        }
      }

      renderList();
      updateMergeUi();

      if (mergedMode === "gi") {
        const arr = collectArts("gi");
        if (arr.length >= 2) {
          const m = window.GiKgViz.mergeParsedArtifacts(arr);
          if (m) {
            onSelect(m);
          }
        } else {
          mergedMode = null;
          updateMergeUi();
          if (activeKey && store.has(activeKey)) {
            onSelect(store.get(activeKey));
          } else {
            onSelect(null);
          }
        }
      } else if (mergedMode === "kg") {
        const arr = collectArts("kg");
        if (arr.length >= 2) {
          const m = window.GiKgViz.mergeParsedArtifacts(arr);
          if (m) {
            onSelect(m);
          }
        } else {
          mergedMode = null;
          updateMergeUi();
          if (activeKey && store.has(activeKey)) {
            onSelect(store.get(activeKey));
          } else {
            onSelect(null);
          }
        }
      } else if (mergedMode === "both") {
        const gis = collectArts("gi");
        const kgs = collectArts("kg");
        if (gis.length >= 1 && kgs.length >= 1) {
          const m = window.GiKgViz.mergeGiKgFromArtifactArrays(gis, kgs);
          if (m) {
            onSelect(m);
          }
        } else {
          mergedMode = null;
          updateMergeUi();
          if (activeKey && store.has(activeKey)) {
            onSelect(store.get(activeKey));
          } else {
            onSelect(null);
          }
        }
      } else if (activeKey && store.has(activeKey)) {
        onSelect(store.get(activeKey));
      } else {
        onSelect(null);
      }
    }

    listNameInput.addEventListener("input", onLayerOrNameFilterChange);

    mergeVisibleBtn.addEventListener("click", function () {
      const action = computeMergeAction();
      if (!action) {
        return;
      }
      if (action === "both") {
        selectMergedBoth();
      } else {
        selectMerged(action);
      }
    });

    function renderList() {
      fileListEl.innerHTML = "";
      if (store.size === 0) {
        emptyHint.classList.remove("hidden");
        fileListEl.classList.add("hidden");
        updateMergeUi();
        return;
      }
      emptyHint.classList.add("hidden");
      fileListEl.classList.remove("hidden");

      const sorted = [...store.entries()].sort(function (a, b) {
        return a[1].name.localeCompare(b[1].name);
      });
      let visibleCount = 0;
      for (let i = 0; i < sorted.length; i++) {
        const key = sorted[i][0];
        const art = sorted[i][1];
        if (!entryPassesFilters(key, art)) {
          continue;
        }
        visibleCount += 1;
        const li = document.createElement("li");
        const btn = document.createElement("button");
        btn.type = "button";
        btn.dataset.key = key;
        const badge = document.createElement("span");
        let badgeKind = "unknown";
        let badgeText = "?";
        if (art.kind === "gi" || art.kind === "kg") {
          badgeKind = art.kind;
          badgeText = art.kind.toUpperCase();
        } else if (art.kind === "both") {
          badgeKind = "both";
          badgeText = "GI+KG";
        }
        badge.className = "badge badge-" + badgeKind;
        badge.textContent = badgeText;
        btn.appendChild(badge);
        const displayName =
          key.indexOf("/") >= 0 || key.indexOf("\\") >= 0 ? key : art.name;
        btn.appendChild(document.createTextNode(displayName));
        if (key === activeKey) {
          btn.classList.add("active");
        }
        btn.addEventListener("click", function () {
          selectKey(key);
        });
        li.appendChild(btn);
        fileListEl.appendChild(li);
      }
      if (visibleCount === 0) {
        const emptyLi = document.createElement("li");
        emptyLi.className = "viz-list-filter-empty hint";
        emptyLi.textContent =
          "No files match the current filters (GI / KG mode or name).";
        fileListEl.appendChild(emptyLi);
      }
      updateMergeUi();
    }

    function selectKey(key) {
      mergedMode = null;
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
      mergedMode = null;
      selectKey(activeKey);
    });

    clearBtn.addEventListener("click", function () {
      hideUrlDataBanner();
      listNameInput.value = "";
      layerMode = "both";
      syncLayerModeButtons();
      syncPrefsToUrlAndNav();
      store.clear();
      activeKey = null;
      mergedMode = null;
      renderList();
      onClear();
      onSelect(null);
    });

    async function tryLoadFromUrlData() {
      if (window.GiKgVizQuery == null) {
        return;
      }
      const dataPath = window.GiKgVizQuery.getDataPathFromUrl();
      if (!dataPath) {
        return;
      }
      let listRes;
      try {
        listRes = await fetch(
          "/_api/gi-kg-list?path=" + encodeURIComponent(dataPath)
        );
      } catch (_e) {
        showUrlDataBanner(
          "warn",
          "Could not reach /_api/gi-kg-list. Run `make serve-gi-kg-viz` from the repo " +
            "root (not plain python -m http.server). ?data= does not open a disk folder."
        );
        return;
      }
      if (listRes.status === 404) {
        showUrlDataBanner(
          "warn",
          "Server has no /_api/gi-kg-list (404). Use `make serve-gi-kg-viz` so ?data= can " +
            "load files. Links with ?data= never open Finder — they only fetch JSON via HTTP."
        );
        return;
      }
      if (!listRes.ok) {
        if (listRes.status === 400) {
          let msg = "Invalid ?data= directory (must exist under repo root).";
          try {
            const err = await listRes.json();
            if (err && err.error) {
              msg = String(err.error);
            }
          } catch (_e2) {
            /* ignore */
          }
          window.alert(msg);
          showUrlDataBanner(
            "warn",
            msg + " ?data= loads via the server; it does not open a folder on disk."
          );
        }
        return;
      }
      let listJson;
      try {
        listJson = await listRes.json();
      } catch (_e) {
        showUrlDataBanner(
          "warn",
          "Bad response from server (not JSON). Use `make serve-gi-kg-viz` — ?data= requires " +
            "scripts/gi_kg_viz_server.py. This does not open Finder or a disk folder."
        );
        return;
      }
      const files = listJson.files || [];
      const filtered = filterArtifactPaths(files, layerMode);
      if (filtered.length === 0) {
        window.alert(
          "No .gi.json / .kg.json found under ?data= path (check path and server; " +
            "python -m http.server alone has no /_api/)."
        );
        showUrlDataBanner(
          "warn",
          "No matching GI/KG files under: " +
            dataPath +
            ". Path must be relative to repo root. ?data= does not open Finder."
        );
        return;
      }
      store.clear();
      mergedMode = null;
      activeKey = null;
      let lastKey = null;
      for (let i = 0; i < filtered.length; i++) {
        const rel = filtered[i];
        let tr;
        try {
          tr = await fetch(repoFileUrl(rel));
        } catch (_e) {
          continue;
        }
        if (!tr.ok) {
          continue;
        }
        const text = await tr.text();
        let data;
        try {
          data = JSON.parse(text);
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e);
          window.alert("Invalid JSON in " + rel + ": " + msg);
          continue;
        }
        if (data === null || typeof data !== "object") {
          window.alert("Expected a JSON object in " + rel);
          continue;
        }
        const base = rel.split("/").pop() || rel;
        const parsed = window.GiKgViz.parseArtifact(base, data);
        store.set(rel, parsed);
        lastKey = rel;
      }
      if (!lastKey) {
        renderList();
        onSelect(null);
        showUrlDataBanner(
          "warn",
          "Could not read any JSON files from ?data= path (fetch errors). Check server logs."
        );
        return;
      }
      showUrlDataBanner(
        "ok",
        "Loaded " +
          String(filtered.length) +
          " file(s) from repo path: " +
          dataPath +
          " (HTTP via make serve-gi-kg-viz). This did not open Finder — use Choose folder… " +
          "to browse disk."
      );
      renderList();
      const doMerge = mergePrefCb.checked;
      if (doMerge) {
        if (
          layerMode === "both" &&
          countByKind("gi") >= 1 &&
          countByKind("kg") >= 1
        ) {
          selectMergedBoth();
          return;
        }
        if (
          countByKind("gi") >= 2 &&
          (layerMode === "gi" || layerMode === "both")
        ) {
          selectMerged("gi");
          return;
        }
        if (
          countByKind("kg") >= 2 &&
          (layerMode === "kg" || layerMode === "both")
        ) {
          selectMerged("kg");
          return;
        }
      }
      selectKey(lastKey);
    }

    tryLoadFromUrlData().catch(function (e) {
      if (typeof console !== "undefined" && console.warn) {
        console.warn("tryLoadFromUrlData", e);
      }
    });
  }

  window.GiKgVizShell = { init: init };
})();
