import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import ProfileAvatar from './ProfileAvatar.vue'

describe('ProfileAvatar', () => {
  it('renders the photo when src is supplied (Area E OAuth/upload avatar)', () => {
    const w = mount(ProfileAvatar, { props: { name: 'Jane Doe', src: 'https://cdn/pic.jpg' } })
    const img = w.find('img')
    expect(img.exists()).toBe(true)
    expect(img.attributes('src')).toBe('https://cdn/pic.jpg')
    expect(w.text()).toBe('') // no initials fallback when a photo is shown
  })

  it('falls back to initials when there is no src', () => {
    const w = mount(ProfileAvatar, { props: { name: 'Jane Doe', src: null } })
    expect(w.find('img').exists()).toBe(false)
    expect(w.text()).toBe('JD')
  })

  it('falls back to initials when the photo fails to load (advisor M5)', async () => {
    const w = mount(ProfileAvatar, { props: { name: 'Jane Doe', src: 'https://cdn/gone.jpg' } })
    await w.find('img').trigger('error')
    expect(w.find('img').exists()).toBe(false) // broken photo hidden
    expect(w.text()).toBe('JD') // initials shown instead of the browser glyph
  })
})
