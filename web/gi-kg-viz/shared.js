/**
 * Shared parsing, overview metrics, and graph adapters for GI/KG artifacts.
 * Exposes window.GiKgViz (no build step).
 */
(function (global) {
  "use strict";

  /** @typedef {{ name: string, kind: 'gi'|'kg'|'unknown', episodeId: string|null, nodes: number, edges: number, nodeTypes: Record<string, number>, data: object }} ParsedArtifact */

  /**
   * @param {string} filename
   * @param {object} data
   * @returns {ParsedArtifact}
   */
  function parseArtifact(filename, data) {
    const nodes = Array.isArray(data.nodes) ? data.nodes : [];
    const edges = Array.isArray(data.edges) ? data.edges : [];
    const episodeId =
      typeof data.episode_id === "string" ? data.episode_id : null;

    /** @type {Record<string, number>} */
    const nodeTypes = {};
    for (const n of nodes) {
      const t = n && typeof n.type === "string" ? n.type : "?";
      nodeTypes[t] = (nodeTypes[t] || 0) + 1;
    }

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

    return {
      name: filename,
      kind,
      episodeId,
      nodes: nodes.length,
      edges: edges.length,
      nodeTypes,
      data,
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
   * @param {object} n
   */
  function nodeLabel(n) {
    const t = n.type || "?";
    const p = n.properties || {};
    let idShort = n.id != null ? String(n.id) : "";
    if (idShort.length > 24) {
      idShort = idShort.slice(0, 22) + "…";
    }
    if (t === "Insight" && p.text) {
      return truncate(p.text, 36);
    }
    if (t === "Quote" && p.text) {
      return truncate(p.text, 36);
    }
    if (t === "Topic" && p.label) {
      return String(p.label);
    }
    if (t === "Entity" && p.name) {
      return String(p.name);
    }
    if (t === "Episode" && p.title) {
      return truncate(String(p.title), 40);
    }
    return t + (idShort ? ": " + idShort : "");
  }

  /**
   * @param {object} n
   */
  function buildNodeTitle(n) {
    try {
      const head = (n.type || "?") + " · " + String(n.id || "");
      const body = JSON.stringify(n.properties || {}, null, 2);
      return truncate(head + "\n" + body, 900);
    } catch (_e) {
      return String(n.id || "");
    }
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
    } else if (art.kind === "kg") {
      const ex = d.extraction && typeof d.extraction === "object" ? d.extraction : {};
      out.rows.push({ k: "Extraction", v: ex.model_version || "—" });
      out.rows.push({ k: "Extracted at", v: ex.extracted_at || "—" });
      const nt = art.nodeTypes;
      if (nt.Topic != null) {
        out.rows.push({ k: "Topics", v: String(nt.Topic) });
      }
      if (nt.Entity != null) {
        out.rows.push({ k: "Entities", v: String(nt.Entity) });
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
      const t = rawNodes[i].type || "?";
      if (!seen.has(t)) {
        seen.add(t);
        allowedTypes[t] = true;
      }
    }
    return {
      allowedTypes: allowedTypes,
      hideUngroundedInsights: false,
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
    if (fullArt.kind === "gi" && state.hideUngroundedInsights) {
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
    if (fullArt.kind === "gi" && state.hideUngroundedInsights) {
      nodes = nodes.filter(function (n) {
        if (n.type !== "Insight") {
          return true;
        }
        return n.properties && n.properties.grounded === true;
      });
    }
    const allowed = state.allowedTypes;
    nodes = nodes.filter(function (n) {
      const t = n.type || "?";
      return allowed[t] !== false;
    });
    const ids = new Set();
    for (let i = 0; i < nodes.length; i++) {
      ids.add(String(nodes[i].id));
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
    if (Object.keys(art.nodeTypes).length === 0) {
      el.innerHTML = '<p class="hint">No nodes.</p>';
      return;
    }
    const parts = Object.entries(art.nodeTypes)
      .sort((a, b) => b[1] - a[1])
      .map(
        ([t, c]) =>
          "<span>" + escapeHtml(t) + " · " + String(c) + "</span>"
      );
    el.innerHTML = '<div class="type-pills">' + parts.join("") + "</div>";
  }

  /**
   * @param {ParsedArtifact} art
   */
  function toGraphElements(art) {
    const d = art.data;
    const rawNodes = Array.isArray(d.nodes) ? d.nodes : [];
    const rawEdges = Array.isArray(d.edges) ? d.edges : [];
    const visNodes = rawNodes.map(function (n, i) {
      const id = n.id != null ? String(n.id) : "n" + i;
      return {
        id: id,
        label: nodeLabel(n),
        group: n.type || "?",
        title: buildNodeTitle(n),
      };
    });
    const idSet = new Set(visNodes.map(function (x) {
      return x.id;
    }));
    const visEdges = rawEdges.map(function (e, i) {
      return {
        id: "e" + i,
        from: String(e.from),
        to: String(e.to),
        label: e.type ? String(e.type) : "",
      };
    });
    return { visNodes: visNodes, visEdges: visEdges, idSet: idSet };
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
  };
})(window);
