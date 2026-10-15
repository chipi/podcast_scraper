import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'

import en from '../i18n/locales/en.json'
import MemberTrendBadge from './MemberTrendBadge.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

const mountBadge = (props: Record<string, unknown>) =>
  mount(MemberTrendBadge, { props, global: { plugins: [i18n] } })

describe('MemberTrendBadge', () => {
  it('renders nothing for steady — a badge on every row is a badge that says nothing', () => {
    expect(mountBadge({ trend: 'steady' }).find('[data-testid="member-trend"]').exists()).toBe(false)
    expect(mountBadge({}).find('[data-testid="member-trend"]').exists()).toBe(false)
  })

  it('renders the four trends that are not steady', () => {
    for (const trend of ['new', 'growing', 'fading', 'gone']) {
      expect(mountBadge({ trend }).find('[data-testid="member-trend"]').exists(), trend).toBe(true)
    }
  })

  it('accents what is arriving and mutes what is leaving', () => {
    expect(mountBadge({ trend: 'new' }).find('[data-testid="member-trend"]').classes()).toContain(
      'text-accent',
    )
    expect(mountBadge({ trend: 'gone' }).find('[data-testid="member-trend"]').classes()).toContain(
      'text-muted',
    )
  })

  it('marks fading apart from gone: an arrow and the cooling colour (operator 2026-10-15)', () => {
    const fading = mountBadge({ trend: 'fading' }).find('[data-testid="member-trend"]')
    const gone = mountBadge({ trend: 'gone' }).find('[data-testid="member-trend"]')
    expect(fading.text()).toBe('↘fading') // the space is a margin on the arrow
    expect(fading.attributes('style')).toContain('var(--lp-trend-cooling)')
    expect(gone.text()).toBe('gone')
    expect(gone.classes()).toContain('text-muted')
    expect(mountBadge({ trend: 'growing' }).find('[data-testid="member-trend"]').text()).toBe('↗growing')
    expect(mountBadge({ trend: 'new' }).find('[data-testid="member-trend"]').text()).toBe('new')
  })

  /**
   * The regression that shipped: the title asked vue-i18n for `d(date, 'monthYear')` and the app
   * registers no `datetimeFormats` at all, so every tooltip fell through the catch to a raw ISO
   * string — visible only as a console warning during a design shot.
   */
  it('dates the badge in words, not as an ISO string', () => {
    const since = mountBadge({ trend: 'new', firstSeen: '2026-03-14' })
    expect(since.find('[data-testid="member-trend"]').attributes('title')).toContain('Mar 2026')

    const until = mountBadge({ trend: 'gone', lastSeen: '2026-07-02' })
    expect(until.find('[data-testid="member-trend"]').attributes('title')).toContain('Jul 2026')
  })

  it('falls back to the plain label when the grouping carries no dates', () => {
    const title = mountBadge({ trend: 'new' }).find('[data-testid="member-trend"]').attributes('title')
    expect(title).toBe(mountBadge({ trend: 'new' }).find('[data-testid="member-trend"]').text())
  })
})
