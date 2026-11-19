/** Fire-and-forget POST of a batch of graph-analytics events. Never throws; ``keepalive`` lets it
 *  survive a tab close so the flush-on-hide batch still lands. */
export function postGraphEvents(events: Array<Record<string, unknown>>): void {
  if (!events.length) {
    return
  }
  void fetch('/api/app/graph-events', {
    method: 'POST',
    credentials: 'include',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ events }),
    keepalive: true,
  }).catch(() => {
    /* analytics is best-effort — never surface a failure */
  })
}
