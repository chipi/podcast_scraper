// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import PodcastCover from './PodcastCover.vue'

// PodcastCover is a pure prop-driven image with a candidate-source fallback
// chain. No store / no emits. We exercise the candidate ordering, the binary
// URL builder, the dedupe, the @error fallback stepping, and the ♪ placeholder.

function mountCover(props: Record<string, unknown>) {
  return mount(PodcastCover, {
    props: { alt: 'Cover', ...props },
    attachTo: document.body,
  })
}

describe('PodcastCover', () => {
  it('always renders the wrapper with the default size class and the data-testid', () => {
    const w = mountCover({})
    const root = w.get('[data-testid="podcast-cover"]')
    expect(root.classes()).toContain('h-10')
    expect(root.classes()).toContain('w-10')
  })

  it('applies a custom sizeClass', () => {
    const w = mountCover({ sizeClass: 'h-20 w-20' })
    const root = w.get('[data-testid="podcast-cover"]')
    expect(root.classes()).toContain('h-20')
    expect(root.classes()).toContain('w-20')
  })

  it('renders the ♪ placeholder (no img) when there are no candidate sources', () => {
    const w = mountCover({})
    expect(w.find('img').exists()).toBe(false)
    const placeholder = w.get('[aria-hidden="true"]')
    expect(placeholder.text()).toContain('♪')
  })

  it('renders an img with the alt text when a remote episode image URL is provided', () => {
    const w = mountCover({
      episodeImageUrl: 'https://cdn.example.com/ep.jpg',
      alt: 'Episode artwork',
    })
    const img = w.get('img')
    expect(img.attributes('src')).toBe('https://cdn.example.com/ep.jpg')
    expect(img.attributes('alt')).toBe('Episode artwork')
    expect(img.attributes('loading')).toBe('lazy')
    expect(img.attributes('referrerpolicy')).toBe('no-referrer')
  })

  it('prefers the local episode binary URL over a remote episode URL when corpusPath is set', () => {
    const w = mountCover({
      corpusPath: '/corpus',
      episodeImageLocalRelpath: 'feed/ep/cover.jpg',
      episodeImageUrl: 'https://cdn.example.com/ep.jpg',
    })
    const src = w.get('img').attributes('src')!
    expect(src.startsWith('/api/corpus/binary?')).toBe(true)
    expect(src).toContain('path=%2Fcorpus')
    expect(src).toContain('relpath=feed%2Fep%2Fcover.jpg')
  })

  it('ignores a local relpath when corpusPath is missing and uses the remote URL', () => {
    const w = mountCover({
      episodeImageLocalRelpath: 'feed/ep/cover.jpg',
      episodeImageUrl: 'https://cdn.example.com/ep.jpg',
    })
    expect(w.get('img').attributes('src')).toBe('https://cdn.example.com/ep.jpg')
  })

  it('falls back to the next candidate on @error and finally to the ♪ placeholder', async () => {
    const w = mountCover({
      episodeImageUrl: 'https://cdn.example.com/ep.jpg',
      feedImageUrl: 'https://cdn.example.com/feed.jpg',
    })
    // First candidate: episode URL.
    expect(w.get('img').attributes('src')).toBe('https://cdn.example.com/ep.jpg')

    // First error -> step to feed URL.
    await w.get('img').trigger('error')
    expect(w.get('img').attributes('src')).toBe('https://cdn.example.com/feed.jpg')

    // Second error -> past the end -> placeholder, no img.
    await w.get('img').trigger('error')
    expect(w.find('img').exists()).toBe(false)
    expect(w.get('[aria-hidden="true"]').text()).toContain('♪')
  })

  it('orders all four candidate sources: local episode, remote episode, local feed, remote feed', async () => {
    const w = mountCover({
      corpusPath: '/c',
      episodeImageLocalRelpath: 'ep-local.jpg',
      episodeImageUrl: 'https://x/ep-remote.jpg',
      feedImageLocalRelpath: 'feed-local.jpg',
      feedImageUrl: 'https://x/feed-remote.jpg',
    })
    expect(w.get('img').attributes('src')).toContain('relpath=ep-local.jpg')
    await w.get('img').trigger('error')
    expect(w.get('img').attributes('src')).toBe('https://x/ep-remote.jpg')
    await w.get('img').trigger('error')
    expect(w.get('img').attributes('src')).toContain('relpath=feed-local.jpg')
    await w.get('img').trigger('error')
    expect(w.get('img').attributes('src')).toBe('https://x/feed-remote.jpg')
  })

  it('dedupes identical candidate URLs so the same src is not retried twice', async () => {
    // episode + feed remote URLs identical -> only one effective candidate.
    const w = mountCover({
      episodeImageUrl: 'https://same/cover.jpg',
      feedImageUrl: 'https://same/cover.jpg',
    })
    expect(w.get('img').attributes('src')).toBe('https://same/cover.jpg')
    await w.get('img').trigger('error')
    // Deduped -> no second candidate -> placeholder.
    expect(w.find('img').exists()).toBe(false)
  })

  it('resets the fallback step when the source props change', async () => {
    const w = mountCover({
      episodeImageUrl: 'https://a/one.jpg',
      feedImageUrl: 'https://a/two.jpg',
    })
    await w.get('img').trigger('error')
    expect(w.get('img').attributes('src')).toBe('https://a/two.jpg')

    // Changing a source prop resets fallbackStep to 0.
    await w.setProps({ episodeImageUrl: 'https://b/new.jpg' })
    expect(w.get('img').attributes('src')).toBe('https://b/new.jpg')
  })

  it('trims whitespace-only props and treats them as absent', () => {
    const w = mountCover({
      episodeImageUrl: '   ',
      feedImageUrl: '',
    })
    expect(w.find('img').exists()).toBe(false)
    expect(w.get('[aria-hidden="true"]').text()).toContain('♪')
  })
})
