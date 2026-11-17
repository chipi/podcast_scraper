/**
 * Monotonic counter for overlapping async work (watchers, double-submit, tab churn).
 *
 * Convention:
 * - At the start of each logical run: ``const seq = gate.bump()``.
 * - After each ``await``: if ``gate.isStale(seq)``, return without writing UI state.
 * - In ``finally``: clear loading flags only when ``gate.isCurrent(seq)``.
 * - To cancel in-flight work without starting a new run: ``gate.invalidate()``.
 */
export class StaleGeneration {
  private gen = 0

  bump(): number {
    this.gen += 1
    return this.gen
  }

  /** Increment generation so older runs become stale (no new sequence for a caller). */
  invalidate(): void {
    this.gen += 1
  }

  isCurrent(seq: number): boolean {
    return seq === this.gen
  }

  isStale(seq: number): boolean {
    return seq !== this.gen
  }
}
