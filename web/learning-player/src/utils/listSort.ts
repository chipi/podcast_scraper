import type { useI18n } from "vue-i18n"

/**
 * The ONE sort model for every browsable list (operator 2026-09-14): Newest / Oldest / A–Z / Z–A,
 * identical on Browse › Episodes and Browse › Shows so the two tabs cannot drift. Episodes sort by
 * publish date + title; shows by feed `last_updated` + title — same four values, same labels.
 */
export type ListSortValue = "newest" | "oldest" | "az" | "za"

type Translate = ReturnType<typeof useI18n>["t"]

export function listSortOptions(t: Translate): { value: ListSortValue; label: string }[] {
  return [
    { value: "newest", label: t("list.sortNewest") },
    { value: "oldest", label: t("list.sortOldest") },
    { value: "az", label: t("list.sortAz") },
    { value: "za", label: t("list.sortZa") },
  ]
}
