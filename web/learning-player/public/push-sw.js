/*
 * Web Push handler, injected into the generated service worker via
 * `workbox.importScripts` (vite.config.ts). Shows the resurfacing nudge and deep-links
 * back into the player on click. Payload is the rendered nudge from the delivery worker
 * (#1415): { title, body, url }.
 */
self.addEventListener('push', (event) => {
  let data = {}
  try {
    data = event.data ? event.data.json() : {}
  } catch (_e) {
    data = {}
  }
  const title = data.title || 'Time to revisit'
  const body = data.body || 'You have highlights worth revisiting.'
  const url = data.url || '/'
  event.waitUntil(
    self.registration.showNotification(title, {
      body,
      icon: '/icon-192.png',
      badge: '/icon-192.png',
      data: { url },
    }),
  )
})

self.addEventListener('notificationclick', (event) => {
  event.notification.close()
  const url = (event.notification.data && event.notification.data.url) || '/'
  event.waitUntil(
    self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then((list) => {
      for (const client of list) {
        if ('focus' in client) return client.focus()
      }
      return self.clients.openWindow(url)
    }),
  )
})
