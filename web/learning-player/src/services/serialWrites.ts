/**
 * One behaviour for every on/off control that writes to the server (operator 2026-09-30:
 * "consistency above all").
 *
 * The follow, heart, follow-show, played, queue and save-insight toggles each flip on screen at
 * once and then write. Each used to fire its request immediately and adopt whichever response
 * arrived, so a quick tap-tap sent both writes in parallel: when the first one's response landed
 * last it overwrote the second tap's state (the button showed the wrong thing), and the server,
 * taking the two requests in either order, could keep the wrong one too.
 *
 * A serializer fixes both halves for everything that uses it:
 *
 * - writes go out ONE AT A TIME, in the order the user tapped, so the server applies them in that
 *   order;
 * - each write is told whether it is still the LATEST one. A response to a superseded tap must not
 *   be adopted, because the newer tap is already showing its own state and its own response will
 *   settle it.
 *
 * One serializer per store, not per item: a store's server responses are whole lists, so a
 * response for item A is as capable of overwriting item B's fresh optimistic state as one for A.
 */
export interface SerialWrites {
  /**
   * Queue `write` behind every earlier write of this serializer. `isLatest()` is true while no
   * later write has been queued; check it before adopting a server response or reverting.
   */
  run<T>(write: (isLatest: () => boolean) => Promise<T>): Promise<T>
}

export function serialWrites(): SerialWrites {
  let chain: Promise<unknown> = Promise.resolve()
  let latest = 0
  return {
    run<T>(write: (isLatest: () => boolean) => Promise<T>): Promise<T> {
      const seq = ++latest
      const isLatest = (): boolean => seq === latest
      const next = (): Promise<T> => write(isLatest)
      const run = chain.then(next, next)
      chain = run.catch(() => {})
      return run
    },
  }
}
