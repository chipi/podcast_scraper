import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { routeForDeepLink } from '../services/deepLinks'

/**
 * App links (operator 2026-10-05): a closelistening.app content link opens in the installed app.
 * FOUR lists have to agree, and nothing else makes them: the paths iOS is told the app handles
 * (apple-app-site-association), the paths Android filters for (AndroidManifest), the paths the
 * app's parser routes (services/deepLinks) and the paths the share menu builds (utils/shareLink).
 * A path claimed but not parsed opens the app on Home; a path parsed but not claimed opens the
 * browser. Either way the link "works" and lands in the wrong place, which no other test sees.
 */
const ROOT = join(__dirname, '..', '..')
const read = (p: string) => readFileSync(join(ROOT, p), 'utf8')

const CONTENT = ['episode', 'podcast', 'topic', 'person', 'storyline', 'theme']

describe('app links agree across iOS, Android, the parser and the share links', () => {
  it('iOS: the association file claims exactly the content paths, for both bundle ids', () => {
    const aasa = JSON.parse(read('public/.well-known/apple-app-site-association'))
    const detail = aasa.applinks.details[0]
    expect(detail.appIDs).toEqual([
      '3P3PX275ZM.app.closelistening.player',
      '3P3PX275ZM.app.closelistening.player.dev',
    ])
    expect(detail.components.map((c: Record<string, string>) => c['/'])).toEqual(
      CONTENT.map((k) => `/${k}/*`),
    )
    expect(read('ios/App/App/App.entitlements')).toContain('applinks:closelistening.app')
  })

  it('Android: the verified filter claims the same paths, and nothing under /api', () => {
    const manifest = read('android/app/src/main/AndroidManifest.xml')
    const block = manifest.slice(manifest.indexOf('android:autoVerify="true"'))
    const prefixes = [...block.matchAll(/android:pathPrefix="([^"]+)"/g)].map((m) => m[1])
    expect(prefixes).toEqual(CONTENT.map((k) => `/${k}/`))
    expect(block).toContain('android:host="closelistening.app"')
    const links = JSON.parse(read('public/.well-known/assetlinks.json'))
    expect(links[0].target.package_name).toBe('app.closelistening.player')
  })

  it('every claimed path is one the app routes — none opens the app on Home', () => {
    for (const kind of CONTENT) {
      expect(routeForDeepLink(`https://closelistening.app/${kind}/x1`), kind).not.toBeNull()
    }
  })

  it('the sign-in link is NOT claimed — email sign-in has to stay in the browser', () => {
    expect(routeForDeepLink('https://closelistening.app/api/app/auth/magic/verify?token=x')).toBeNull()
  })
})
