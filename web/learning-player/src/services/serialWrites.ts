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
  /**
   * Fetch the store's list so that a tap made WHILE the fetch was in flight is not undone by it.
   *
   * A load that started before a tap returns the server's state from before that tap. Adopting it
   * wiped the tap's on-screen flip, so the NEXT tap read the wrong state and sent the wrong write —
   * caught in a trace on 2026-10-01: GET /interests landed after the follow's POST had gone out,
   * reset the button, and the "unfollow" tap sent a second follow. So: if any write was queued
   * during the fetch, wait for the writes to finish and fetch again; return only a list fetched
   * with no tap in between. Bounded by the user's own taps.
   */
  fresh<T>(fetch: () => Promise<T>): Promise<T>
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
    async fresh<T>(fetch: () => Promise<T>): Promise<T> {
      for (;;) {
        const before = latest
        const value = await fetch()
        if (latest === before) return value
        await chain
      }
    },
  }
}
