import { describe, expect, it } from 'vitest'

import { postLinkSignInRoute } from './redirect'

describe('postLinkSignInRoute', () => {
  it('sends a just-created account to its profile, from anywhere', () => {
    const profile = { name: 'profile', query: { welcome: '1' } }
    expect(postLinkSignInRoute(true, { name: 'landing' })).toEqual(profile)
    expect(postLinkSignInRoute(true, { name: 'episode' })).toEqual(profile)
  })

  it('takes a returning account off the signed-out landing — home, or its safe redirect', () => {
    expect(postLinkSignInRoute(false, { name: 'landing' })).toEqual({ name: 'home' })
    expect(postLinkSignInRoute(false, { name: 'login', query: { redirect: '/library' } })).toEqual({
      path: '/library',
    })
    // An unsafe redirect is ignored, never followed off-origin.
    expect(postLinkSignInRoute(false, { name: 'landing', query: { redirect: '//evil.test' } })).toEqual(
      { name: 'home' },
    )
  })

  it('leaves a returning account where it already was', () => {
    expect(postLinkSignInRoute(false, { name: 'episode' })).toBeNull()
    expect(postLinkSignInRoute(false, {})).toBeNull()
  })
})
