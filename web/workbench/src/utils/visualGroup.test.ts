import { describe, expect, it } from "vitest"
import { visualGroupForNode, visualNodeTypeCounts } from "./visualGroup"

describe("visualGroupForNode", () => {
  it("returns ? for null/undefined", () => {
    expect(visualGroupForNode(null)).toBe("?")
    expect(visualGroupForNode(undefined)).toBe("?")
  })

  it("passes through non-Entity types", () => {
    expect(visualGroupForNode({ type: "Topic" })).toBe("Topic")
    expect(visualGroupForNode({ type: "Insight" })).toBe("Insight")
  })

  it("maps GIL Person to Entity_person styling group", () => {
    expect(visualGroupForNode({ type: "Person", properties: { name: "Ada" } })).toBe(
      "Entity_person"
    )
  })

  it("does NOT default Entity to a person when no entity_kind (#2057)", () => {
    // This test used to assert 'Entity_person'. The viewer had the same defect as the backend
    // classifier: an ABSENT kind was treated as evidence of personhood, so a legacy Entity with
    // no kind was drawn as a human. `object` is the catch-all for "a named thing we cannot
    // place" — see ENTITY_KINDS in kg/llm_extract.py.
    expect(visualGroupForNode({ type: "Entity" })).toBe("Entity_object")
  })

  it("maps a first-class Object node to its own visual group (#2057)", () => {
    // Before this, Object fell through as the raw string 'Object', which has no entry in
    // graphNodeTypeStyles — every Object rendered in the unknown-node grey with no legend.
    expect(visualGroupForNode({ type: "Object" })).toBe("Entity_object")
  })

  it("maps CIL kind person|org on Entity", () => {
    expect(visualGroupForNode({ type: "Entity", properties: { kind: "person" } })).toBe(
      "Entity_person"
    )
    expect(visualGroupForNode({ type: "Entity", properties: { kind: "org" } })).toBe(
      "Entity_organization"
    )
  })

  it("maps organization variants to Entity_organization", () => {
    for (const kind of ["organization", "org", "company", "corporation", "institution"]) {
      expect(visualGroupForNode({ type: "Entity", properties: { entity_kind: kind } })).toBe(
        "Entity_organization"
      )
    }
  })

  it("maps person to Entity_person", () => {
    expect(visualGroupForNode({ type: "Entity", properties: { entity_kind: "person" } })).toBe(
      "Entity_person"
    )
  })

  it("handles case-insensitive entity_kind", () => {
    expect(
      visualGroupForNode({ type: "Entity", properties: { entity_kind: "Organization" } })
    ).toBe("Entity_organization")
  })

  it("treats a blank entity_kind as unplaceable, not as a person (#2057)", () => {
    expect(visualGroupForNode({ type: "Entity", properties: { entity_kind: "  " } })).toBe(
      "Entity_object"
    )
  })

  it("treats an unrecognised entity_kind as an object, not a person (#2057)", () => {
    expect(visualGroupForNode({ type: "Entity", properties: { entity_kind: "event" } })).toBe(
      "Entity_object"
    )
  })
})

describe("visualNodeTypeCounts", () => {
  it("counts visual groups", () => {
    const nodes = [
      { type: "Topic" },
      { type: "Topic" },
      { type: "Entity", properties: { entity_kind: "org" } },
      { type: "Entity" },
    ]
    expect(visualNodeTypeCounts(nodes)).toEqual({
      Topic: 2,
      Entity_organization: 1,
      // The kindless Entity counts as an object, not a person (#2057).
      Entity_object: 1,
    })
  })

  it("returns empty for empty array", () => {
    expect(visualNodeTypeCounts([])).toEqual({})
  })

  it("handles non-array gracefully", () => {
    expect(visualNodeTypeCounts(null as unknown as [])).toEqual({})
  })
})
