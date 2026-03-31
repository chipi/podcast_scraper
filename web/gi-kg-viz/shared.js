/**
 * Shared parsing, overview metrics, and graph adapters for GI/KG artifacts.
 * Exposes window.GiKgViz (no build step).
 */
(function (global) {
  "use strict";

  /** @typedef {{ name: string, kind: 'gi'|'kg'|'both'|'unknown', episodeId: string|null, nodes: number, edges: number, nodeTypes: Record<string, number>, data: object }} ParsedArtifact */

  /**
   * @param {string} filename
   * @param {object} data
   * @returns {ParsedArtifact}
   */
  function parseArtifact(filename, data) {
    let nodes = Array.isArray(data.nodes) ? data.nodes.slice() : [];
    let edges = Array.isArray(data.edges) ? data.edges.slice() : [];
    const episodeId =
      typeof data.episode_id === "string" ? data.episode_id : null;

    let kind = "unknown";
    const lower = filename.toLowerCase();
    if (lower.endsWith(".gi.json")) {
      kind = "gi";
    } else if (lower.endsWith(".kg.json")) {
      kind = "kg";
    } else if (
      data.extraction &&
      typeof data.extraction === "object" &&
      !Object.prototype.hasOwnProperty.call(data, "prompt_version")
    ) {
      kind = "kg";
    } else if (
      typeof data.model_version === "string" &&
      typeof data.prompt_version === "string"
    ) {
      kind = "gi";
    }

    if (kind === "gi") {
      const aug = ensureEpisodeToInsightEdges(nodes, edges);
      nodes = aug.nodes;
      edges = aug.edges;
    }

    /** @type {Record<string, number>} */
    const nodeTypes = {};
    for (let ni = 0; ni < nodes.length; ni++) {
      const n = nodes[ni];
      const t = n && typeof n.type === "string" ? n.type : "?";
      nodeTypes[t] = (nodeTypes[t] || 0) + 1;
    }

    const dataOut =
      kind === "gi"
        ? Object.assign({}, data, { nodes: nodes, edges: edges })
        : data;

    return {
      name: filename,
      kind,
      episodeId,
      nodes: nodes.length,
      edges: edges.length,
      nodeTypes,
      data: dataOut,
    };
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  /**
   * @param {string} s
   * @param {number} max
   */
  function truncate(s, max) {
    s = String(s);
    if (s.length <= max) {
      return s;
    }
    return s.slice(0, max - 1) + "…";
  }

  /**
   * @param {string} slug
   */
  function humanizeSlug(slug) {
    return String(slug)
      .split("-")
      .filter(Boolean)
      .map(function (w) {
        return w.length ? w.charAt(0).toUpperCase() + w.slice(1) : "";
      })
      .join(" ");
  }

  /**
   * Display name from global entity id `entity:{kind}:{slug}` when properties omit name.
   * Strips optional `g:` / `k:` prefix from combined GI+KG graphs.
   * @param {string} idStr
   */
  function entityDisplayNameFromId(idStr) {
    let s = String(idStr);
    if (s.indexOf("k:") === 0 || s.indexOf("g:") === 0) {
      s = s.slice(2);
    }
    const m = s.match(/^entity:(?:person|organization):(.+)$/);
    if (!m || !m[1]) {
      return "";
    }
    return humanizeSlug(m[1]);
  }

  /**
   * @param {object} n
   */
  function nodeLabel(n) {
    const tRaw = n.type != null ? String(n.type) : "?";
    const tLower = tRaw.toLowerCase();
    const p = n.properties || {};
    let idShort = n.id != null ? String(n.id) : "";
    if (idShort.length > 24) {
      idShort = idShort.slice(0, 22) + "…";
    }
    if (tLower === "insight" && p.text) {
      return truncate(p.text, 36);
    }
    if (tLower === "quote" && p.text) {
      return truncate(p.text, 36);
    }
    if (tLower === "topic" && p.label) {
      return truncate(String(p.label), 80);
    }
    if (tLower === "entity") {
      const nm =
        (p.label && String(p.label).trim()) ||
        (p.name && String(p.name).trim()) ||
        entityDisplayNameFromId(n.id) ||
        "";
      if (nm) {
        let out = truncate(nm, 52);
        const role = p.role && String(p.role).trim();
        if (role && role !== "mentioned") {
          out = truncate(out + " (" + role + ")", 58);
        }
        return out;
      }
    }
    if (
      tLower === "speaker" &&
      ((p.name && String(p.name).trim()) || (p.label && String(p.label).trim()))
    ) {
      return truncate(String(p.name || p.label).trim(), 48);
    }
    if (tLower === "episode" && p.title) {
      return truncate(String(p.title), 40);
    }
    return tRaw + (idShort ? ": " + idShort : "");
  }

  /**
   * @param {object} n
   */
  function buildNodeTitle(n) {
    try {
      const disp = nodeLabel(n);
      const head =
        (n.type || "?") +
        "\n" +
        disp +
        "\nid: " +
        String(n.id || "") +
        "\n\nproperties:\n" +
        JSON.stringify(n.properties || {}, null, 2);
      return truncate(head, 900);
    } catch (_e) {
      return String(n.id || "");
    }
  }

  /**
   * @param {ParsedArtifact|null} art
   * @param {string|number} nodeId
   * @returns {object|null}
   */
  function findRawNodeInArtifact(art, nodeId) {
    if (!art || !art.data) {
      return null;
    }
    const nodes = Array.isArray(art.data.nodes) ? art.data.nodes : [];
    const sid = String(nodeId);
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i];
      if (n && n.id != null && String(n.id) === sid) {
        return n;
      }
    }
    return null;
  }

  /**
   * HTML snippet for the node detail popover (escaped). Empty string if not found.
   * @param {ParsedArtifact|null} art
   * @param {string|number} nodeId
   * @returns {string}
   */
  function buildNodeDetailHtml(art, nodeId) {
    const n = findRawNodeInArtifact(art, nodeId);
    if (!n) {
      return "";
    }
    const type = escapeHtml(String(n.type != null ? n.type : "?"));
    const id = escapeHtml(String(n.id != null ? n.id : ""));
    const disp = escapeHtml(nodeLabel(n));
    let html = '<dl class="graph-node-detail-dl">';
    html += '<dt>id</dt><dd class="graph-node-detail-mono">' + id + "</dd>";
    html += "<dt>type</dt><dd>" + type + "</dd>";
    html += "<dt>display</dt><dd>" + disp + "</dd>";
    const p = n.properties;
    if (p && typeof p === "object") {
      const keys = Object.keys(p).sort();
      for (let i = 0; i < keys.length; i++) {
        const k = keys[i];
        let v = p[k];
        if (v === null || v === undefined) {
          v = "";
        } else if (typeof v === "object") {
          try {
            v = JSON.stringify(v);
          } catch (_e) {
            v = String(v);
          }
        } else {
          v = String(v);
        }
        v = truncate(v, 400);
        html += "<dt>" + escapeHtml(k) + "</dt><dd>" + escapeHtml(v) + "</dd>";
      }
    }
    html += "</dl>";
    return html;
  }

  /**
   * @param {ParsedArtifact} art
   */
  function edgeTypeCounts(data) {
    const edges = Array.isArray(data.edges) ? data.edges : [];
    /** @type {Record<string, number>} */
    const et = {};
    for (const e of edges) {
      const t = e && typeof e.type === "string" ? e.type : "?";
      et[t] = (et[t] || 0) + 1;
    }
    return et;
  }

  /**
   * Overview metrics for the panel (GI / KG specific rows).
   * @param {ParsedArtifact} art
   */
  function computeMetrics(art) {
    /** @type {{ rows: { k: string, v: string }[], edgeTypes: Record<string, number> }} */
    const out = {
      rows: [],
      edgeTypes: edgeTypeCounts(art.data),
    };

    out.rows.push({ k: "File", v: art.name });
    out.rows.push({
      k: "Layer",
      v:
        art.kind === "gi"
          ? "Grounded insights (GIL)"
          : art.kind === "kg"
            ? "Knowledge graph (KG)"
            : art.kind === "both"
              ? "GI + KG (combined)"
              : "Unknown",
    });
    out.rows.push({ k: "Episode", v: art.episodeId || "—" });
    out.rows.push({ k: "Nodes", v: String(art.nodes) });
    out.rows.push({ k: "Edges", v: String(art.edges) });

    const d = art.data;
    if (art.kind === "gi") {
      out.rows.push({ k: "Model", v: d.model_version || "—" });
      out.rows.push({ k: "Prompt", v: d.prompt_version || "—" });
      const nodes = Array.isArray(d.nodes) ? d.nodes : [];
      let insights = 0;
      let groundedTrue = 0;
      let quotes = 0;
      let speakers = 0;
      for (const n of nodes) {
        if (n.type === "Insight") {
          insights += 1;
          if (n.properties && n.properties.grounded === true) {
            groundedTrue += 1;
          }
        }
        if (n.type === "Quote") {
          quotes += 1;
        }
        if (n.type === "Speaker") {
          speakers += 1;
        }
      }
      out.rows.push({ k: "Insights", v: String(insights) });
      out.rows.push({ k: "Grounded (true)", v: String(groundedTrue) });
      out.rows.push({
        k: "Not grounded",
        v: String(Math.max(0, insights - groundedTrue)),
      });
      if (insights > 0) {
        const pct = ((100 * groundedTrue) / insights).toFixed(1);
        out.rows.push({ k: "% grounded", v: pct + "%" });
      }
      out.rows.push({ k: "Quotes", v: String(quotes) });
      if (speakers > 0) {
        out.rows.push({ k: "Speakers", v: String(speakers) });
      }
      const vcGi = visualNodeTypeCounts(nodes);
      if (vcGi.Entity_person) {
        out.rows.push({
          k: graphNodeLegendLabel("Entity_person"),
          v: String(vcGi.Entity_person),
        });
      }
      if (vcGi.Entity_organization) {
        out.rows.push({
          k: graphNodeLegendLabel("Entity_organization"),
          v: String(vcGi.Entity_organization),
        });
      }
    } else if (art.kind === "kg") {
      const ex = d.extraction && typeof d.extraction === "object" ? d.extraction : {};
      out.rows.push({ k: "Extraction", v: ex.model_version || "—" });
      out.rows.push({ k: "Extracted at", v: ex.extracted_at || "—" });
      const nt = art.nodeTypes;
      const kgNodes = Array.isArray(d.nodes) ? d.nodes : [];
      const vcKg = visualNodeTypeCounts(kgNodes);
      if (nt.Topic != null) {
        out.rows.push({ k: "Topics", v: String(nt.Topic) });
      }
      const ep = vcKg.Entity_person || 0;
      const eo = vcKg.Entity_organization || 0;
      if (ep > 0 || eo > 0) {
        if (ep > 0) {
          out.rows.push({
            k: graphNodeLegendLabel("Entity_person"),
            v: String(ep),
          });
        }
        if (eo > 0) {
          out.rows.push({
            k: graphNodeLegendLabel("Entity_organization"),
            v: String(eo),
          });
        }
      } else if (nt.Entity != null) {
        out.rows.push({ k: "Entities", v: String(nt.Entity) });
      }
    } else if (art.kind === "both") {
      if (typeof d.model_version === "string") {
        out.rows.push({ k: "GI model", v: d.model_version });
      }
      if (typeof d.prompt_version === "string") {
        out.rows.push({ k: "GI prompt", v: d.prompt_version });
      }
      const ex = d.extraction && typeof d.extraction === "object" ? d.extraction : {};
      if (ex.model_version || ex.extracted_at) {
        out.rows.push({ k: "KG extraction", v: ex.model_version || "—" });
        if (ex.extracted_at) {
          out.rows.push({ k: "KG extracted at", v: ex.extracted_at });
        }
      }
      const bothNodes = Array.isArray(d.nodes) ? d.nodes : [];
      const vcBoth = visualNodeTypeCounts(bothNodes);
      if (vcBoth.Entity_person) {
        out.rows.push({
          k: graphNodeLegendLabel("Entity_person"),
          v: String(vcBoth.Entity_person),
        });
      }
      if (vcBoth.Entity_organization) {
        out.rows.push({
          k: graphNodeLegendLabel("Entity_organization"),
          v: String(vcBoth.Entity_organization),
        });
      }
    }

    return out;
  }

  /**
   * @param {object[]} nodes
   * @returns {Record<string, number>}
   */
  function nodeTypesFromNodes(nodes) {
    /** @type {Record<string, number>} */
    const nt = {};
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i];
      const t = n && typeof n.type === "string" ? n.type : "?";
      nt[t] = (nt[t] || 0) + 1;
    }
    return nt;
  }

  /**
   * @param {ParsedArtifact|null} art
   * @returns {{ allowedTypes: Record<string, boolean>, hideUngroundedInsights: boolean }|null}
   */
  function defaultFilterState(art) {
    if (!art) {
      return null;
    }
    /** @type {Record<string, boolean>} */
    const allowedTypes = {};
    const rawNodes = Array.isArray(art.data.nodes) ? art.data.nodes : [];
    const seen = new Set();
    for (let i = 0; i < rawNodes.length; i++) {
      const n = rawNodes[i];
      if (!n || typeof n !== "object") {
        continue;
      }
      const t = n.type || "?";
      if (!seen.has(t)) {
        seen.add(t);
        allowedTypes[t] = true;
      }
    }
    return {
      allowedTypes: allowedTypes,
      hideUngroundedInsights: false,
      legendSoloVisual: null,
    };
  }

  /**
   * @param {ParsedArtifact|null} fullArt
   * @param {{ allowedTypes: Record<string, boolean>, hideUngroundedInsights: boolean }|null} state
   */
  function filtersActive(fullArt, state) {
    if (!fullArt || !state) {
      return false;
    }
    const keys = Object.keys(state.allowedTypes);
    for (let i = 0; i < keys.length; i++) {
      if (state.allowedTypes[keys[i]] === false) {
        return true;
      }
    }
    if (
      (fullArt.kind === "gi" || fullArt.kind === "both") &&
      state.hideUngroundedInsights
    ) {
      return true;
    }
    if (
      typeof state.legendSoloVisual === "string" &&
      state.legendSoloVisual.length > 0
    ) {
      return true;
    }
    return false;
  }

  /**
   * @param {ParsedArtifact} fullArt
   * @param {{ allowedTypes: Record<string, boolean>, hideUngroundedInsights: boolean }} state
   * @returns {ParsedArtifact}
   */
  function applyGraphFilters(fullArt, state) {
    let nodes = (fullArt.data.nodes || []).slice();
    if (
      (fullArt.kind === "gi" || fullArt.kind === "both") &&
      state.hideUngroundedInsights
    ) {
      nodes = nodes.filter(function (n) {
        if (!n || typeof n !== "object") {
          return false;
        }
        if (n.type !== "Insight") {
          return true;
        }
        return n.properties && n.properties.grounded === true;
      });
    }
    const allowed = state.allowedTypes;
    nodes = nodes.filter(function (n) {
      if (!n || typeof n !== "object") {
        return false;
      }
      const t = n.type || "?";
      return allowed[t] !== false;
    });
    const soloV = state.legendSoloVisual;
    if (typeof soloV === "string" && soloV.length > 0) {
      nodes = nodes.filter(function (n) {
        return visualGroupForNode(n) === soloV;
      });
    }
    const ids = new Set();
    for (let i = 0; i < nodes.length; i++) {
      if (nodes[i].id != null) {
        ids.add(String(nodes[i].id));
      }
    }
    const edges = (fullArt.data.edges || []).filter(function (e) {
      return ids.has(String(e.from)) && ids.has(String(e.to));
    });
    const nodeTypes = nodeTypesFromNodes(nodes);
    return {
      name: fullArt.name,
      kind: fullArt.kind,
      episodeId: fullArt.episodeId,
      nodes: nodes.length,
      edges: edges.length,
      nodeTypes: nodeTypes,
      data: Object.assign({}, fullArt.data, {
        nodes: nodes,
        edges: edges,
      }),
    };
  }

  /**
   * Restrict the graph to one node and its direct neighbors (induced 1-hop ego network).
   * Uses the same node/edge lists as the current filtered artifact (no new edges).
   *
   * @param {ParsedArtifact} art
   * @param {string|null|undefined} focusId
   * @returns {ParsedArtifact}
   */
  function filterArtifactEgoOneHop(art, focusId) {
    if (focusId == null || focusId === "") {
      return art;
    }
    const f = String(focusId);
    const nodes = Array.isArray(art.data.nodes) ? art.data.nodes : [];
    const edges = Array.isArray(art.data.edges) ? art.data.edges : [];
    const idSet = new Set();
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i];
      if (n && n.id != null) {
        idSet.add(String(n.id));
      }
    }
    if (!idSet.has(f)) {
      return art;
    }
    const keep = new Set([f]);
    for (let i = 0; i < edges.length; i++) {
      const e = edges[i];
      if (!e) {
        continue;
      }
      const fr = String(e.from);
      const to = String(e.to);
      if (fr === f && idSet.has(to)) {
        keep.add(to);
      } else if (to === f && idSet.has(fr)) {
        keep.add(fr);
      }
    }
    const nodesOut = nodes.filter(function (n) {
      return n && n.id != null && keep.has(String(n.id));
    });
    const outIds = new Set(
      nodesOut.map(function (n) {
        return String(n.id);
      })
    );
    const edgesOut = edges.filter(function (e) {
      if (!e) {
        return false;
      }
      return outIds.has(String(e.from)) && outIds.has(String(e.to));
    });
    const nodeTypes = nodeTypesFromNodes(nodesOut);
    return {
      name: art.name,
      kind: art.kind,
      episodeId: art.episodeId,
      nodes: nodesOut.length,
      edges: edgesOut.length,
      nodeTypes: nodeTypes,
      data: Object.assign({}, art.data, {
        nodes: nodesOut,
        edges: edgesOut,
      }),
    };
  }

  /**
   * @param {HTMLElement} el
   * @param {ParsedArtifact|null} art
   * @param {{ baseline?: ParsedArtifact|null, filteredNote?: boolean }|undefined} options
   */
  function renderMetricsPanel(el, art, options) {
    options = options || {};
    if (!art) {
      el.innerHTML = '<p class="hint">Select a file from the list.</p>';
      return;
    }
    const baseline = options.baseline;
    const showNote = options.filteredNote && baseline;
    let note = "";
    if (
      showNote &&
      (art.nodes !== baseline.nodes || art.edges !== baseline.edges)
    ) {
      note =
        '<p class="filter-note">After filters: <strong>' +
        String(art.nodes) +
        "</strong> / " +
        String(baseline.nodes) +
        " nodes, <strong>" +
        String(art.edges) +
        "</strong> / " +
        String(baseline.edges) +
        " edges.</p>";
    } else if (showNote) {
      note = '<p class="filter-note">Filters active.</p>';
    }
    const m = computeMetrics(art);
    let html = note + '<dl class="metrics-dl">';
    for (const row of m.rows) {
      html +=
        "<dt>" +
        escapeHtml(row.k) +
        "</dt><dd>" +
        escapeHtml(row.v) +
        "</dd>";
    }
    html += "</dl>";
    const etKeys = Object.keys(m.edgeTypes);
    if (etKeys.length > 0) {
      etKeys.sort(function (a, b) {
        return m.edgeTypes[b] - m.edgeTypes[a];
      });
      html += '<p class="metrics-subh">Edges by type</p><ul class="metrics-ul">';
      for (let i = 0; i < etKeys.length; i++) {
        const t = etKeys[i];
        html +=
          "<li>" +
          escapeHtml(t) +
          ": <strong>" +
          String(m.edgeTypes[t]) +
          "</strong></li>";
      }
      html += "</ul>";
    }
    el.innerHTML = html;
  }

  /**
   * @param {HTMLElement} el
   * @param {ParsedArtifact|null} art
   */
  function renderNodeTypePills(el, art) {
    if (!art) {
      el.innerHTML = '<p class="hint">—</p>';
      return;
    }
    const vc = visualNodeTypeCounts(art.data.nodes || []);
    if (Object.keys(vc).length === 0) {
      el.innerHTML = '<p class="hint">No nodes.</p>';
      return;
    }
    const parts = Object.entries(vc)
      .sort((a, b) => b[1] - a[1])
      .map(function (ent) {
        const t = ent[0];
        const c = ent[1];
        const lab = graphNodeLegendLabel(t);
        return "<span>" + escapeHtml(lab) + " · " + String(c) + "</span>";
      });
    el.innerHTML = '<div class="type-pills">' + parts.join("") + "</div>";
  }

  /**
   * Combine several parsed artifacts of the same layer (all GI or all KG) into one graph.
   * Node IDs are assumed unique across episodes (pipeline embeds episode id). Duplicate IDs
   * keep the first node; edges are unioned and de-duplicated by (from, to, type).
   *
   * @param {ParsedArtifact[]} arts
   * @returns {ParsedArtifact|null}
   */
  function mergeParsedArtifacts(arts) {
    if (!arts || arts.length < 2) {
      return null;
    }
    const kind = arts[0].kind;
    if (kind !== "gi" && kind !== "kg") {
      return null;
    }
    for (let i = 1; i < arts.length; i++) {
      if (arts[i].kind !== kind) {
        return null;
      }
    }
    const nodeById = new Map();
    for (let ai = 0; ai < arts.length; ai++) {
      const rawNodes = Array.isArray(arts[ai].data.nodes)
        ? arts[ai].data.nodes
        : [];
      for (let ni = 0; ni < rawNodes.length; ni++) {
        const n = rawNodes[ni];
        if (!n || n.id == null) {
          continue;
        }
        const id = String(n.id);
        if (!nodeById.has(id)) {
          nodeById.set(id, JSON.parse(JSON.stringify(n)));
        }
      }
    }
    const idSet = new Set(nodeById.keys());
    const edgeList = [];
    const edgeSeen = new Set();
    for (let ai = 0; ai < arts.length; ai++) {
      const rawEdges = Array.isArray(arts[ai].data.edges)
        ? arts[ai].data.edges
        : [];
      for (let ei = 0; ei < rawEdges.length; ei++) {
        const e = rawEdges[ei];
        if (!e || e.from == null || e.to == null) {
          continue;
        }
        const from = String(e.from);
        const to = String(e.to);
        if (!idSet.has(from) || !idSet.has(to)) {
          continue;
        }
        const ek = from + "\0" + to + "\0" + String(e.type || "");
        if (edgeSeen.has(ek)) {
          continue;
        }
        edgeSeen.add(ek);
        edgeList.push(JSON.parse(JSON.stringify(e)));
      }
    }
    const mergedData = JSON.parse(JSON.stringify(arts[0].data));
    mergedData.nodes = Array.from(nodeById.values());
    mergedData.edges = edgeList;
    mergedData.episode_id = "merged:" + String(arts.length) + "-artifacts";
    const nodeTypes = {};
    for (let i = 0; i < mergedData.nodes.length; i++) {
      const t = mergedData.nodes[i].type || "?";
      nodeTypes[t] = (nodeTypes[t] || 0) + 1;
    }
    return {
      name: "Merged " + kind.toUpperCase() + " (" + arts.length + " files)",
      kind: kind,
      episodeId: mergedData.episode_id,
      nodes: mergedData.nodes.length,
      edges: mergedData.edges.length,
      nodeTypes: nodeTypes,
      data: mergedData,
    };
  }

  /**
   * Union one GI and one KG parsed artifact into a single graph (disjunct id prefixes).
   * @param {ParsedArtifact} giArt
   * @param {ParsedArtifact} kgArt
   * @returns {ParsedArtifact|null}
   */
  function combineGiKgParsedArtifacts(giArt, kgArt) {
    if (!giArt || !kgArt || giArt.kind !== "gi" || kgArt.kind !== "kg") {
      return null;
    }
    /**
     * @param {object} artData
     * @param {string} pfx
     */
    function remapData(artData, pfx) {
      const data = JSON.parse(JSON.stringify(artData));
      const nodes = Array.isArray(data.nodes) ? data.nodes : [];
      const idMap = new Map();
      for (let i = 0; i < nodes.length; i++) {
        const n = nodes[i];
        if (!n || n.id == null) {
          continue;
        }
        const old = String(n.id);
        const neu = pfx + old;
        idMap.set(old, neu);
        n.id = neu;
      }
      const edges = Array.isArray(data.edges) ? data.edges : [];
      for (let e = 0; e < edges.length; e++) {
        const ed = edges[e];
        if (!ed) {
          continue;
        }
        if (ed.from != null) {
          const f = String(ed.from);
          ed.from = idMap.has(f) ? idMap.get(f) : pfx + f;
        }
        if (ed.to != null) {
          const t = String(ed.to);
          ed.to = idMap.has(t) ? idMap.get(t) : pfx + t;
        }
      }
      return data;
    }
    const gd = remapData(giArt.data, "g:");
    const kd = remapData(kgArt.data, "k:");

    /**
     * Remap edge endpoints still using g:episode:KEY / k:episode:KEY to __unified_ep__:KEY
     * when that unified Episode exists. Safety net after rewriteEdges for HAS_INSIGHT / MENTIONS.
     * @param {object[]} nodes
     * @param {object[]|undefined} edges
     * @returns {object[]}
     */
    function repairStalePrefixedEpisodeRefs(nodes, edges) {
      const unified = new Map();
      const nArr = Array.isArray(nodes) ? nodes : [];
      for (let i = 0; i < nArr.length; i++) {
        const n = nArr[i];
        if (!n || n.type !== "Episode" || n.id == null) {
          continue;
        }
        const id = String(n.id);
        const p = "__unified_ep__:";
        if (id.indexOf(p) === 0) {
          const key = id.slice(p.length);
          if (key) {
            unified.set(key, id);
          }
        }
      }
      if (unified.size === 0) {
        return Array.isArray(edges) ? edges : [];
      }
      const eArr = Array.isArray(edges) ? edges : [];
      return eArr.map(function (ed) {
        if (!ed || typeof ed !== "object") {
          return ed;
        }
        const o = Object.assign({}, ed);
        function fix(v) {
          if (v == null) {
            return v;
          }
          const s = String(v);
          if (s.indexOf("g:episode:") === 0) {
            const k = s.slice("g:episode:".length);
            if (unified.has(k)) {
              return unified.get(k);
            }
          }
          if (s.indexOf("k:episode:") === 0) {
            const k = s.slice("k:episode:".length);
            if (unified.has(k)) {
              return unified.get(k);
            }
          }
          return v;
        }
        o.from = fix(o.from);
        o.to = fix(o.to);
        return o;
      });
    }

    /**
     * Merge one Episode per shared anchor id (episode:… UUID) from GI and KG layers.
     * Uses node ids (g:episode:KEY / k:episode:KEY), not artifact root episode_id, so
     * multi-file merged GI/KG (root episode_id like merged:N-artifacts) still dedupes.
     * @returns {{ nodes: object[], edges: object[], episode_id: string }|null}
     */
    function unifyGiKgEpisodeAnchors(gdIn, kdIn) {
      const PGI = "g:episode:";
      const PKG = "k:episode:";
      const nodesGi = (gdIn.nodes || []).slice();
      const nodesKg = (kdIn.nodes || []).slice();
      /** @type {Map<string, string>} */
      const mapGi = new Map();
      /** @type {Map<string, string>} */
      const mapKg = new Map();
      for (let i = 0; i < nodesGi.length; i++) {
        const n = nodesGi[i];
        if (!n || n.type !== "Episode" || n.id == null) {
          continue;
        }
        const sid = String(n.id);
        if (sid.indexOf(PGI) !== 0) {
          continue;
        }
        const key = sid.slice(PGI.length);
        if (key) {
          mapGi.set(key, sid);
        }
      }
      for (let i = 0; i < nodesKg.length; i++) {
        const n = nodesKg[i];
        if (!n || n.type !== "Episode" || n.id == null) {
          continue;
        }
        const sid = String(n.id);
        if (sid.indexOf(PKG) !== 0) {
          continue;
        }
        const key = sid.slice(PKG.length);
        if (key) {
          mapKg.set(key, sid);
        }
      }
      const keys = [];
      mapGi.forEach(function (_gid, key) {
        if (mapKg.has(key)) {
          keys.push(key);
        }
      });
      keys.sort();
      if (keys.length === 0) {
        return null;
      }
      const giRemove = new Set();
      const kgRemove = new Set();
      const unifiedList = [];
      for (let ki = 0; ki < keys.length; ki++) {
        const key = keys[ki];
        const giEpId = mapGi.get(key);
        const kgEpId = mapKg.get(key);
        giRemove.add(giEpId);
        kgRemove.add(kgEpId);
        let giNode = null;
        let kgNode = null;
        for (let i = 0; i < nodesGi.length; i++) {
          if (nodesGi[i] && String(nodesGi[i].id) === giEpId) {
            giNode = nodesGi[i];
            break;
          }
        }
        for (let i = 0; i < nodesKg.length; i++) {
          if (nodesKg[i] && String(nodesKg[i].id) === kgEpId) {
            kgNode = nodesKg[i];
            break;
          }
        }
        const unifiedId = "__unified_ep__:" + key;
        unifiedList.push({
          id: unifiedId,
          type: "Episode",
          properties: Object.assign(
            {},
            giNode && giNode.properties ? giNode.properties : {},
            kgNode && kgNode.properties ? kgNode.properties : {}
          ),
        });
      }
      const restGi = nodesGi.filter(function (n) {
        return n && !giRemove.has(String(n.id));
      });
      const restKg = nodesKg.filter(function (n) {
        return n && !kgRemove.has(String(n.id));
      });
      /** @type {Map<string, string>} */
      const repl = new Map();
      for (let ri = 0; ri < keys.length; ri++) {
        const key = keys[ri];
        const u = "__unified_ep__:" + key;
        repl.set(PGI + key, u);
        repl.set(PKG + key, u);
      }
      function rewriteEdges(edges) {
        const arr = Array.isArray(edges) ? edges : [];
        return arr.map(function (ed) {
          if (!ed || typeof ed !== "object") {
            return ed;
          }
          const o = Object.assign({}, ed);
          const from = o.from != null ? String(o.from) : "";
          const to = o.to != null ? String(o.to) : "";
          if (from && repl.has(from)) {
            o.from = repl.get(from);
          }
          if (to && repl.has(to)) {
            o.to = repl.get(to);
          }
          return o;
        });
      }
      const edgesGi = rewriteEdges(gdIn.edges);
      const edgesKg = rewriteEdges(kdIn.edges);
      const nodesOut = restGi.concat(restKg).concat(unifiedList);
      const edgesOut = repairStalePrefixedEpisodeRefs(nodesOut, edgesGi.concat(edgesKg));
      let epRoot;
      if (keys.length === 1) {
        epRoot = "merged:gi+kg:" + keys[0];
      } else {
        epRoot = "merged:gi+kg:multi";
      }
      return {
        nodes: nodesOut,
        edges: edgesOut,
        episode_id: epRoot,
      };
    }

    const unified = unifyGiKgEpisodeAnchors(gd, kd);
    let mergedData;
    if (unified) {
      mergedData = Object.assign({}, gd, {
        nodes: unified.nodes,
        edges: unified.edges,
        episode_id: unified.episode_id,
      });
    } else {
      mergedData = Object.assign({}, gd, {
        nodes: (gd.nodes || []).concat(kd.nodes || []),
        edges: (gd.edges || []).concat(kd.edges || []),
        episode_id:
          "merged:gi+kg:" +
          String(giArt.episodeId || giArt.name) +
          "+" +
          String(kgArt.episodeId || kgArt.name),
      });
    }
    if (kd.extraction && typeof kd.extraction === "object") {
      mergedData.extraction = kd.extraction;
    }
    const epAug = ensureEpisodeToInsightEdges(
      mergedData.nodes || [],
      mergedData.edges || []
    );
    mergedData.nodes = epAug.nodes;
    mergedData.edges = epAug.edges;
    const nodeTypes = nodeTypesFromNodes(mergedData.nodes);
    return {
      name: "Merged GI + KG",
      kind: "both",
      episodeId: mergedData.episode_id,
      nodes: mergedData.nodes.length,
      edges: mergedData.edges.length,
      nodeTypes: nodeTypes,
      data: mergedData,
    };
  }

  /**
   * Coalesce same-layer artifacts (merge if 2+, else single).
   * @param {ParsedArtifact[]} giArts
   * @param {ParsedArtifact[]} kgArts
   * @returns {ParsedArtifact|null}
   */
  function mergeGiKgFromArtifactArrays(giArts, kgArts) {
    if (!giArts || giArts.length < 1 || !kgArts || kgArts.length < 1) {
      return null;
    }
    const giMerged =
      giArts.length >= 2 ? mergeParsedArtifacts(giArts) : giArts[0];
    const kgMerged =
      kgArts.length >= 2 ? mergeParsedArtifacts(kgArts) : kgArts[0];
    if (!giMerged || !kgMerged) {
      return null;
    }
    const combined = combineGiKgParsedArtifacts(giMerged, kgMerged);
    if (!combined) {
      return null;
    }
    combined.name =
      "Merged GI + KG (" +
      String(giArts.length) +
      " GI · " +
      String(kgArts.length) +
      " KG)";
    return combined;
  }

  /**
   * Add HAS_INSIGHT (Episode → Insight) when missing so Episode is not isolated in the viz.
   * Matches insights to episodes via properties.episode_id and Episode node id conventions.
   * @param {object[]} nodes
   * @param {object[]} edges
   * @returns {{ nodes: object[], edges: object[] }}
   */
  function ensureEpisodeToInsightEdges(nodes, edges) {
    const nList = Array.isArray(nodes) ? nodes.slice() : [];
    const eList = Array.isArray(edges)
      ? edges.map(function (e) {
          return Object.assign({}, e);
        })
      : [];
    const episodes = nList.filter(function (n) {
      return n && n.type === "Episode";
    });
    const insights = nList.filter(function (n) {
      return n && n.type === "Insight";
    });
    if (episodes.length === 0 || insights.length === 0) {
      return { nodes: nList, edges: eList };
    }
    /**
     * @param {object} ep
     * @returns {string|null}
     */
    function episodeKeyFromNode(ep) {
      const id = String(ep.id);
      if (id.indexOf("g:episode:") === 0) {
        return id.slice("g:episode:".length);
      }
      if (id.indexOf("k:episode:") === 0) {
        return id.slice("k:episode:".length);
      }
      if (id.indexOf("episode:") === 0) {
        return id.slice("episode:".length);
      }
      if (id.indexOf("__unified_ep__:") === 0) {
        return id.slice("__unified_ep__:".length);
      }
      return null;
    }
    const seen = new Set();
    for (let i = 0; i < eList.length; i++) {
      const e = eList[i];
      if (!e) {
        continue;
      }
      seen.add(
        String(e.from) +
          "\0" +
          String(e.to) +
          "\0" +
          String(e.type || "")
      );
    }
    for (let ei = 0; ei < episodes.length; ei++) {
      const ep = episodes[ei];
      const eid = episodeKeyFromNode(ep);
      if (eid == null) {
        continue;
      }
      const epId = String(ep.id);
      for (let ii = 0; ii < insights.length; ii++) {
        const ins = insights[ii];
        const p = ins.properties || {};
        if (String(p.episode_id || "") !== eid) {
          continue;
        }
        const iid = String(ins.id);
        const k = epId + "\0" + iid + "\0HAS_INSIGHT";
        if (seen.has(k)) {
          continue;
        }
        eList.push({ type: "HAS_INSIGHT", from: epId, to: iid });
        seen.add(k);
      }
    }
    /**
     * If the graph is clearly single-episode but edges or id conventions are incomplete
     * (e.g. hand-edited JSON, missing properties.episode_id), still anchor insights to the
     * sole Episode node. Skip when multiple episodes — would be ambiguous.
     */
    if (episodes.length === 1) {
      const soleEp = episodes[0];
      const epId = String(soleEp.id);
      for (let ii = 0; ii < insights.length; ii++) {
        const ins = insights[ii];
        const iid = String(ins.id);
        let hasIncoming = false;
        for (let j = 0; j < eList.length; j++) {
          const e = eList[j];
          if (
            e &&
            String(e.type || "") === "HAS_INSIGHT" &&
            String(e.to) === iid
          ) {
            hasIncoming = true;
            break;
          }
        }
        if (hasIncoming) {
          continue;
        }
        const k = epId + "\0" + iid + "\0HAS_INSIGHT";
        if (seen.has(k)) {
          continue;
        }
        eList.push({ type: "HAS_INSIGHT", from: epId, to: iid });
        seen.add(k);
      }
    }
    return { nodes: nList, edges: eList };
  }

  /**
   * @param {ParsedArtifact} art
   */
  function toGraphElements(art) {
    const d = art.data;
    let rawNodes = Array.isArray(d.nodes) ? d.nodes.slice() : [];
    let rawEdges = Array.isArray(d.edges)
      ? d.edges.map(function (e) {
          return Object.assign({}, e);
        })
      : [];
    if (art.kind === "gi" || art.kind === "both") {
      const aug = ensureEpisodeToInsightEdges(rawNodes, rawEdges);
      rawNodes = aug.nodes;
      rawEdges = aug.edges;
    }
    const visNodes = rawNodes.map(function (n, i) {
      if (!n || typeof n !== "object") {
        return {
          id: "n" + i,
          label: "?",
          group: "?",
          title: "",
        };
      }
      const id = n.id != null ? String(n.id) : "n" + i;
      return {
        id: id,
        label: nodeLabel(n),
        group: visualGroupForNode(n),
        title: buildNodeTitle(n),
      };
    });
    const idSet = new Set(visNodes.map(function (x) {
      return x.id;
    }));
    const visEdges = rawEdges.map(function (e, i) {
      if (!e || typeof e !== "object") {
        return { id: "e" + i, from: "", to: "", label: "" };
      }
      return {
        id: "e" + i,
        from: e.from != null ? String(e.from) : "",
        to: e.to != null ? String(e.to) : "",
        label: e.type ? String(e.type) : "",
      };
    });
    return { visNodes: visNodes, visEdges: visEdges, idSet: idSet };
  }

  /**
   * vis-network group / Cytoscape data.type for styling. Entity nodes split by entity_kind.
   * @param {object} n
   * @returns {string}
   */
  function visualGroupForNode(n) {
    if (!n || typeof n !== "object") {
      return "?";
    }
    const t = typeof n.type === "string" ? n.type : "?";
    if (t !== "Entity") {
      return t;
    }
    const p = n.properties || {};
    const raw = p.entity_kind;
    if (typeof raw !== "string" || !raw.trim()) {
      return "Entity_person";
    }
    const k = raw.trim().toLowerCase();
    const isOrg =
      k === "organization" ||
      k === "org" ||
      k === "company" ||
      k === "corporation" ||
      k === "institution";
    return isOrg ? "Entity_organization" : "Entity_person";
  }

  /**
   * Count nodes by legend / graph visual group (Entity split into person vs organization).
   * @param {object[]} nodes
   * @returns {Record<string, number>}
   */
  function visualNodeTypeCounts(nodes) {
    /** @type {Record<string, number>} */
    const nt = {};
    const arr = Array.isArray(nodes) ? nodes : [];
    for (let i = 0; i < arr.length; i++) {
      const g = visualGroupForNode(arr[i]);
      nt[g] = (nt[g] || 0) + 1;
    }
    return nt;
  }

  /**
   * @param {ParsedArtifact} art
   */
  function toVisData(art) {
    const g = toGraphElements(art);
    const nodes = g.visNodes.map(function (n) {
      return {
        id: n.id,
        label: n.label,
        group: n.group,
        title: n.title,
      };
    });
    const edges = g.visEdges.filter(function (e) {
      return g.idSet.has(e.from) && g.idSet.has(e.to);
    });
    return { nodes: nodes, edges: edges };
  }

  /**
   * @param {ParsedArtifact} art
   */
  function toCytoElements(art) {
    const g = toGraphElements(art);
    const nodes = g.visNodes.map(function (n) {
      return {
        data: {
          id: n.id,
          label: n.label,
          type: n.group,
        },
      };
    });
    const edges = g.visEdges
      .filter(function (e) {
        return g.idSet.has(e.from) && g.idSet.has(e.to);
      })
      .map(function (e) {
        return {
          data: {
            id: e.id,
            source: e.from,
            target: e.to,
            label: e.label,
          },
        };
      });
    return nodes.concat(edges);
  }

  /** Fallback node fill when type is missing or unknown (matches Cytoscape default). */
  const GRAPH_NODE_UNKNOWN_FILL = "#868e96";

  /**
   * Single source of truth for graph node colors (vis-network + Cytoscape + legend).
   * @type {Readonly<Record<string, { background: string, border: string, labelColor: string }>>}
   */
  const graphNodeTypeStyles = Object.freeze({
    Episode: {
      background: "#4c6ef5",
      border: "#364fc7",
      labelColor: "#ffffff",
    },
    Insight: {
      background: "#40c057",
      border: "#2f9e44",
      labelColor: "#0d1117",
    },
    Quote: {
      background: "#fab005",
      border: "#e67700",
      labelColor: "#0d1117",
    },
    Speaker: {
      background: "#69db7c",
      border: "#2b8a3e",
      labelColor: "#0d1117",
    },
    Topic: {
      background: "#da77f2",
      border: "#862e9c",
      labelColor: "#0d1117",
    },
    /** KG entities: person vs organization (visual group only; filters use type Entity). */
    Entity_person: {
      background: "#9775fa",
      border: "#5f3dc4",
      labelColor: "#ffffff",
    },
    Entity_organization: {
      background: "#12b886",
      border: "#087f5b",
      labelColor: "#ffffff",
    },
    /** Fallback when consumers pass semantic type "Entity" (e.g. chart). */
    Entity: {
      background: "#9775fa",
      border: "#5f3dc4",
      labelColor: "#ffffff",
    },
    Podcast: {
      background: "#748ffc",
      border: "#4263eb",
      labelColor: "#ffffff",
    },
  });

  const graphNodeTypesOrdered = Object.freeze([
    "Episode",
    "Insight",
    "Quote",
    "Speaker",
    "Topic",
    "Entity_person",
    "Entity_organization",
    "Podcast",
  ]);

  /**
   * Legend row text for graphNodeTypesOrdered keys.
   * @param {string} key
   * @returns {string}
   */
  function graphNodeLegendLabel(key) {
    if (key === "Entity_person") {
      return "Entity (person)";
    }
    if (key === "Entity_organization") {
      return "Entity (organization)";
    }
    return key;
  }

  /**
   * Map legend visual key (vis group) to semantic node.type for filter checkboxes.
   * @param {string} visualKey
   * @returns {string}
   */
  function semanticTypeForLegendVisual(visualKey) {
    if (visualKey === "Entity_person" || visualKey === "Entity_organization") {
      return "Entity";
    }
    return visualKey;
  }

  /**
   * @param {string} type — visual group (Entity_person) or semantic chart label (Entity).
   * @returns {string}
   */
  function graphNodeFill(type) {
    const s =
      graphNodeTypeStyles[type] ||
      (type === "Entity" ? graphNodeTypeStyles.Entity_person : undefined);
    return s ? s.background : GRAPH_NODE_UNKNOWN_FILL;
  }

  /**
   * vis-network `groups` option keyed by node type.
   * @returns {Record<string, object>}
   */
  function toVisNetworkGroups() {
    /** @type {Record<string, object>} */
    const out = {};
    for (let i = 0; i < graphNodeTypesOrdered.length; i++) {
      const k = graphNodeTypesOrdered[i];
      const s = graphNodeTypeStyles[k];
      if (!s) {
        continue;
      }
      out[k] = {
        color: { background: s.background, border: s.border },
        font: { color: s.labelColor },
      };
    }
    return out;
  }

  global.GiKgViz = {
    parseArtifact: parseArtifact,
    computeMetrics: computeMetrics,
    renderMetricsPanel: renderMetricsPanel,
    renderNodeTypePills: renderNodeTypePills,
    toVisData: toVisData,
    toCytoElements: toCytoElements,
    escapeHtml: escapeHtml,
    defaultFilterState: defaultFilterState,
    filtersActive: filtersActive,
    applyGraphFilters: applyGraphFilters,
    nodeTypesFromNodes: nodeTypesFromNodes,
    mergeParsedArtifacts: mergeParsedArtifacts,
    mergeGiKgFromArtifactArrays: mergeGiKgFromArtifactArrays,
    combineGiKgParsedArtifacts: combineGiKgParsedArtifacts,
    graphNodeTypeStyles: graphNodeTypeStyles,
    graphNodeTypesOrdered: graphNodeTypesOrdered,
    graphNodeUnknownFill: GRAPH_NODE_UNKNOWN_FILL,
    graphNodeFill: graphNodeFill,
    graphNodeLegendLabel: graphNodeLegendLabel,
    semanticTypeForLegendVisual: semanticTypeForLegendVisual,
    visualGroupForNode: visualGroupForNode,
    visualNodeTypeCounts: visualNodeTypeCounts,
    toVisNetworkGroups: toVisNetworkGroups,
    filterArtifactEgoOneHop: filterArtifactEgoOneHop,
    buildNodeDetailHtml: buildNodeDetailHtml,
  };
})(window);
