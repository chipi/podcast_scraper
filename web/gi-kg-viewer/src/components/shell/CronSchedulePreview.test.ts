// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import CronSchedulePreview from './CronSchedulePreview.vue'

const YAML = `scheduled_jobs:
  - name: nightly
    cron: "0 2 * * *"
    enabled: true
  - name: broken
    cron: not a cron
    enabled: true
`

describe('CronSchedulePreview', () => {
  it('renders nothing when there are no scheduled jobs', () => {
    const w = mount(CronSchedulePreview, { props: { yaml: 'max_episodes: 5\n' } })
    expect(w.find('[data-testid="cron-schedule-preview"]').exists()).toBe(false)
  })

  it('previews valid jobs and flags invalid crons', () => {
    const w = mount(CronSchedulePreview, { props: { yaml: YAML } })
    expect(w.get('[data-testid="cron-schedule-preview-row-0"]').text()).toContain('nightly')
    expect(w.get('[data-testid="cron-schedule-preview-row-0"]').text()).toContain('next:')
    expect(w.find('[data-testid="cron-schedule-preview-invalid-1"]').exists()).toBe(true)
    expect(w.get('[data-testid="cron-schedule-preview-invalid-summary"]').text()).toContain('1 invalid')
  })

  it('marks a disabled job as disabled (no next-run line)', () => {
    const w = mount(CronSchedulePreview, {
      props: { yaml: 'scheduled_jobs:\n  - name: off\n    cron: "0 2 * * *"\n    enabled: false\n' },
    })
    expect(w.get('[data-testid="cron-schedule-preview-row-0"]').text()).toContain('(disabled)')
  })
})
