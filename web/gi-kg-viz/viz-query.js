/**
 * URL query helpers for GI/KG viewer pages (layer filter, auto-merge, nav links).
 * Exposes window.GiKgVizQuery (no build step).
 */
(function (global) {
  "use strict";

  const LAYER_VALUES = ["gi", "kg", "both"];

  /**
   * @param {string|null|undefined} raw
   * @returns {'gi'|'kg'|'both'}
   */
  function normalizeLayer(raw) {
    const s = (raw || "both").toString().toLowerCase();
    return LAYER_VALUES.indexOf(s) >= 0 ? s : "both";
  }

  /**
   * Repo-relative directory for auto-load (requires make serve-gi-kg-viz dev server).
   * @returns {string|null}
   */
  function getDataPathFromUrl() {
    try {
      const p = new URLSearchParams(global.location.search);
      const d = p.get("data");
      if (d == null) {
        return null;
      }
      const t = String(d).trim();
      return t ? t : null;
    } catch (_e) {
      return null;
    }
  }

  /**
   * @returns {{ layer: 'gi'|'kg'|'both', merged: boolean, data: string|null }}
   */
  function getVizQueryPrefs() {
    let p;
    try {
      p = new URLSearchParams(global.location.search);
    } catch (_e) {
      p = new URLSearchParams();
    }
    return {
      layer: normalizeLayer(p.get("layer")),
      merged:
        p.get("merged") === "1" ||
        p.get("merged") === "true" ||
        p.get("merged") === "yes",
      data: getDataPathFromUrl(),
    };
  }

  /**
   * Serialize prefs for cross-page viewer links (home ↔ graph pages).
   * @param {{ layer?: string, merged?: boolean, data?: string|null }|undefined} override
   * @returns {string} leading "?" or ""
   */
  function vizNavQueryString(override) {
    const base = getVizQueryPrefs();
    const layer = normalizeLayer(
      override && override.layer != null ? override.layer : base.layer
    );
    const merged =
      override && override.merged != null ? !!override.merged : base.merged;
    let data;
    if (override && Object.prototype.hasOwnProperty.call(override, "data")) {
      data = override.data;
    } else {
      data = base.data;
    }
    const out = new URLSearchParams();
    if (layer !== "both") {
      out.set("layer", layer);
    }
    if (merged) {
      out.set("merged", "1");
    }
    if (data) {
      out.set("data", data);
    }
    const s = out.toString();
    return s ? "?" + s : "";
  }

  /**
   * @param {ParentNode} root
   */
  function applyVizNavLinks(root) {
    const el = root || global.document;
    if (!el || !el.querySelectorAll) {
      return;
    }
    const q = vizNavQueryString();
    const nodes = el.querySelectorAll("a[data-viz-nav]");
    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i];
      const href = a.getAttribute("href");
      if (!href) {
        continue;
      }
      const base = href.split("?")[0];
      a.setAttribute("href", base + q);
    }
  }

  /**
   * @param {{ layer?: string, merged?: boolean, data?: string|null }} prefs
   */
  function replaceUrlQuery(prefs) {
    if (!global.history || !global.history.replaceState || !global.URL) {
      return;
    }
    try {
      const u = new URL(global.location.href);
      const prevData = u.searchParams.get("data");
      const layerRaw =
        prefs.layer != null ? prefs.layer : u.searchParams.get("layer");
      const layer = normalizeLayer(layerRaw || "both");
      if (layer === "both") {
        u.searchParams.delete("layer");
      } else {
        u.searchParams.set("layer", layer);
      }
      const merged =
        prefs.merged != null
          ? !!prefs.merged
          : u.searchParams.get("merged") === "1";
      if (merged) {
        u.searchParams.set("merged", "1");
      } else {
        u.searchParams.delete("merged");
      }
      if (Object.prototype.hasOwnProperty.call(prefs, "data")) {
        if (prefs.data) {
          u.searchParams.set("data", prefs.data);
        } else {
          u.searchParams.delete("data");
        }
      } else if (prevData) {
        u.searchParams.set("data", prevData);
      }
      const search = u.searchParams.toString();
      const path = u.pathname + (search ? "?" + search : "");
      global.history.replaceState(null, "", path);
    } catch (_e) {
      /* ignore */
    }
  }

  global.GiKgVizQuery = {
    getVizQueryPrefs: getVizQueryPrefs,
    getDataPathFromUrl: getDataPathFromUrl,
    vizNavQueryString: vizNavQueryString,
    applyVizNavLinks: applyVizNavLinks,
    replaceUrlQuery: replaceUrlQuery,
    normalizeLayer: normalizeLayer,
  };
})(typeof window !== "undefined" ? window : globalThis);
