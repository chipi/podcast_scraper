/**
 * Pure-function label renderer for ``CilTopicPillsRow`` (pre-#656 foundation).
 *
 * Lives outside the ``.vue`` file so we can unit-test it without a Vue
 * mount — the component re-exports nothing usable via ``<script setup>``.
 */

export type CilPillTruncation = 'ellipsis' | 'wrap' | 'none'

/**
 * Apply the configured truncation strategy to a topic-pill label.
 *
 * - ``none`` returns the trimmed label verbatim (CSS decides overflow).
 * - ``wrap`` also returns the trimmed label; caller is responsible for
 *   using ``whitespace-normal break-words`` in the template so the pill
 *   grows to accommodate.
 * - ``ellipsis`` clips at ``maxChars - 1`` and appends a U+2026 horizontal
 *   ellipsis (legacy pre-#656 behaviour).
 */
export function renderPillLabel(
  label: string,
  maxChars: number,
  strategy: CilPillTruncation,
): string {
  const s = label.trim()
  if (strategy === 'none' || strategy === 'wrap') {
    return s
  }
  if (s.length <= maxChars) {
    return s
  }
  return `${s.slice(0, maxChars - 1)}…`
}
