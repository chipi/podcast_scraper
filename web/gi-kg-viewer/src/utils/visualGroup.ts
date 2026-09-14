import type { RawGraphNode } from "../types/artifact"

/**
 * vis-network group / Cytoscape data.type for styling. Entity split by kind / entity_kind.
 */
export function visualGroupForNode(n: RawGraphNode | null | undefined): string {
  if (!n || typeof n !== "object") return "?"
  const t = typeof n.type === "string" ? n.type : "?"
  if (t === "TopicCluster") return "TopicCluster"
  // RFC-097 v3.0: Person + Organization are first-class typed nodes; both
  // map to their existing visual groups so legend / metrics / stylesheet
  // surfaces treat them uniformly with legacy v1.x Entity(kind=...) shapes.
  if (t === "Person") return "Entity_person"
  if (t === "Organization") return "Entity_organization"
  // KG schema 2.1 (#2057). Without this, Object fell through the `t !== 'Entity'` return below
  // as the raw string 'Object', which has no entry in graphNodeTypeStyles — so every Object
  // rendered in the unknown-node grey with no legend entry.
  if (t === "Object") return "Entity_object"
  if (t !== "Entity") return t
  const p = n.properties || {}
  const kindRaw = typeof p.kind === "string" ? p.kind.trim().toLowerCase() : ""
  if (kindRaw === "org") return "Entity_organization"
  if (kindRaw === "person") return "Entity_person"
  const raw = p.entity_kind
  if (typeof raw !== "string" || !raw.trim()) {
    // #2057: an absent kind is NOT evidence of personhood. The backend classifier stopped
    // defaulting to person for exactly this reason; the viewer had the same bug, so a legacy
    // Entity with no kind was drawn as a human.
    return "Entity_object"
  }
  const k = raw.trim().toLowerCase()
  const isOrg =
    k === "organization" ||
    k === "org" ||
    k === "company" ||
    k === "corporation" ||
    k === "institution"
  if (isOrg) return "Entity_organization"
  if (k === "person") return "Entity_person"
  // Anything else the extractor emitted is a named thing we cannot place — the `object`
  // catch-all, not a person (#2057).
  return "Entity_object"
}

export function visualNodeTypeCounts(nodes: RawGraphNode[]): Record<string, number> {
  const nt: Record<string, number> = {}
  const arr = Array.isArray(nodes) ? nodes : []
  for (const n of arr) {
    const g = visualGroupForNode(n)
    nt[g] = (nt[g] || 0) + 1
  }
  return nt
}
