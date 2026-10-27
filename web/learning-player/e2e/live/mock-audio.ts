import { fileURLToPath } from 'node:url'
import type { BrowserContext, Request } from '@playwright/test'

/**
 * Keep the live smoke off PUBLISHERS' audio: every audio request the page makes outside the app's
 * own origin is answered with one of our own committed fixture MP3s.
 *
 * Opening an episode page mounts the player, which loads the episode's media at once (PlayerView
 * calls player.load() on mount, preload=metadata). On prod that media is the publisher's real file,
 * so the smoke's result depended on whichever publisher was behind the episode it opened, and when
 * that host's audio would not load from the CI runner the player filed it as a prod GlitchTip error
 * (PLAYER-1J on 10-19, PLAYER-1L on 10-27, both during the smoke). The operator's rule: tests play
 * our own mock episodes, never real shows. The real-host chain (API -> /audio proxy -> mock podcast
 * host -> playable bytes) is asserted before deploy by e2e/fixture-audio.spec.ts against the mock
 * server; here the page, the routing and the player run on the deployed app, with our audio.
 *
 * Returns the hosts it intercepted, so a spec can assert the publisher was never contacted.
 */
const FIXTURE_MP3 = fileURLToPath(
  new URL('../../../../tests/fixtures/audio/v3/p06_e05.mp3', import.meta.url),
)

function isAudio(req: Request, url: URL): boolean {
  return (
    req.resourceType() === 'media' ||
    // finalStreamUrl() resolves a publisher redirect with a 1-byte Range fetch.
    req.headers()['range'] !== undefined ||
    /\.(mp3|m4a|aac|ogg|opus|wav)$/i.test(url.pathname)
  )
}

export async function mockEpisodeAudio(context: BrowserContext, appOrigin: string): Promise<string[]> {
  const own = new URL(appOrigin).origin
  const intercepted: string[] = []
  await context.route('**/*', async (route) => {
    const req = route.request()
    const url = new URL(req.url())
    if (url.origin === own || !/^https?:$/.test(url.protocol) || !isAudio(req, url)) {
      return route.fallback()
    }
    intercepted.push(url.host)
    return route.fulfill({
      path: FIXTURE_MP3,
      contentType: 'audio/mpeg',
      headers: { 'access-control-allow-origin': '*', 'accept-ranges': 'none' },
    })
  })
  return intercepted
}
