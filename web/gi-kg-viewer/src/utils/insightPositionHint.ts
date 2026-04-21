/**
 * UXS-009 / node detail: present `position_hint` (0–1) without implying false
 * precision. Prefer time-into-episode when duration is known; otherwise tier labels.
 */
export function formatInsightPositionHintLine(
  positionHint: number,
  episodeDurationMs?: number | null,
): string {
  const ph = Math.max(0, Math.min(1, positionHint))
  if (
    typeof episodeDurationMs === 'number' &&
    Number.isFinite(episodeDurationMs) &&
    episodeDurationMs > 0
  ) {
    const secondsInto = (episodeDurationMs / 1000) * ph
    const totalS = Math.max(0, Math.round(secondsInto))
    const m = Math.floor(totalS / 60)
    const s = totalS % 60
    if (m > 0) {
      const secPart = s > 0 ? ` ${s}s` : ''
      return `Position in episode: about ${m}m${secPart} into the episode`
    }
    return `Position in episode: about ${s}s into the episode`
  }
  if (ph < 0.34) {
    return 'Position in episode: early segment'
  }
  if (ph < 0.67) {
    return 'Position in episode: middle segment'
  }
  return 'Position in episode: late segment'
}
