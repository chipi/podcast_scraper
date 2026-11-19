import { describe, expect, it } from 'vitest'
import { parseScheduledJobsFromYaml, toggleScheduledJobEnabled } from './scheduledJobsYaml'

const YAML = `# operator config
profile: cloud_balanced
scheduled_jobs:
  - name: nightly
    cron: "0 2 * * *"
    enabled: true  # keep this comment
  - name: weekly
    cron: "0 3 * * 0"
    enabled: false
max_episodes: 5
`

describe('toggleScheduledJobEnabled', () => {
  it('flips an entry and preserves the trailing comment + other lines', () => {
    const out = toggleScheduledJobEnabled(YAML, 'nightly', false)
    expect(out).not.toBeNull()
    expect(out).toContain('enabled: false  # keep this comment')
    // Other entry + surrounding keys untouched.
    expect(out).toContain('  - name: weekly')
    expect(out).toContain('max_episodes: 5')
    expect(out).toContain('# operator config')
  })

  it('flips the second entry independently', () => {
    const out = toggleScheduledJobEnabled(YAML, 'weekly', true)
    const weeklyBlock = out!.slice(out!.indexOf('- name: weekly'))
    expect(weeklyBlock).toContain('enabled: true')
    // nightly stays enabled (its own comment intact).
    expect(out).toContain('enabled: true  # keep this comment')
  })

  it('inserts an enabled line when the entry has none', () => {
    const yaml = `scheduled_jobs:\n  - name: adhoc\n    cron: "0 1 * * *"\n`
    const out = toggleScheduledJobEnabled(yaml, 'adhoc', false)
    expect(out).toBe(
      `scheduled_jobs:\n  - name: adhoc\n    enabled: false\n    cron: "0 1 * * *"\n`,
    )
  })

  it('returns null for an unknown job name', () => {
    expect(toggleScheduledJobEnabled(YAML, 'missing', true)).toBeNull()
  })

  it('returns null when there is no scheduled_jobs block', () => {
    expect(toggleScheduledJobEnabled('profile: x\nmax_episodes: 5\n', 'nightly', true)).toBeNull()
  })

  it('handles quoted names', () => {
    const yaml = `scheduled_jobs:\n  - name: "nightly sweep"\n    enabled: true\n`
    const out = toggleScheduledJobEnabled(yaml, 'nightly sweep', false)
    expect(out).toContain('enabled: false')
  })
})

describe('parseScheduledJobsFromYaml', () => {
  it('extracts name/cron/enabled per entry, stripping quotes + comments', () => {
    const yaml = `scheduled_jobs:
  - name: nightly
    cron: "0 2 * * *"  # daily
    enabled: true
  - name: weekly
    cron: 0 3 * * 0
    enabled: false
`
    expect(parseScheduledJobsFromYaml(yaml)).toEqual([
      { name: 'nightly', cron: '0 2 * * *', enabled: true },
      { name: 'weekly', cron: '0 3 * * 0', enabled: false },
    ])
  })

  it('defaults enabled to true when absent', () => {
    expect(parseScheduledJobsFromYaml('scheduled_jobs:\n  - name: a\n    cron: "* * * * *"\n')).toEqual([
      { name: 'a', cron: '* * * * *', enabled: true },
    ])
  })

  it('returns [] when there is no scheduled_jobs block', () => {
    expect(parseScheduledJobsFromYaml('max_episodes: 5\n')).toEqual([])
  })
})
