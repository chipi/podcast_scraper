import { describe, expect, it } from 'vitest'

import { authedInfoFromCallback, tokenFromCallback } from './native'

// The native sign-in callback (#1310) and the email magic link (#2272) share one URL shape:
// `closelistening://auth#token=<signed>[&new=1]`. Only the magic link sets `new`, and it is the
// one thing that decides whether a fresh account lands on its profile or on home.
describe('sign-in callback parsing', () => {
  it('reads the token from the fragment', () => {
    expect(tokenFromCallback('closelistening://auth#token=abc.def&new=1')).toBe('abc.def')
    expect(tokenFromCallback('closelistening://auth')).toBeNull()
  })

  it('reports a created account only for new=1', () => {
    expect(authedInfoFromCallback('closelistening://auth#token=t&new=1')).toEqual({ isNew: true })
    expect(authedInfoFromCallback('closelistening://auth#token=t&new=0')).toEqual({ isNew: false })
    // OAuth callbacks never carry the flag: a returning OAuth user must not be sent to the profile.
    expect(authedInfoFromCallback('closelistening://auth#token=t')).toEqual({ isNew: false })
  })
})
