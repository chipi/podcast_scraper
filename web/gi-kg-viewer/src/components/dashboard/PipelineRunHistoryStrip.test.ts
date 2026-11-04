// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import type { CorpusRunSummaryItem } from '../../api/corpusMetricsApi'
import PipelineRunHistoryStrip from './PipelineRunHistoryStrip.vue'

function run(id: string, over: Partial<CorpusRunSummaryItem> = {}): CorpusRunSummaryItem {
  return {
    relative_path: `feeds/a/${id}/run.json`,
    run_id: id,
    created_at: '2026-06-10T10:00:00Z',
    run_duration_seconds: 125,
    episodes_scraped_total: 12,
    errors_total: 0,
    gi_artifacts_generated: null,
    kg_artifacts_generated: null,
    time_scraping_seconds: null,
    time_parsing_seconds: null,
    time_normalizing_seconds: null,
    time_io_and_waiting_seconds: null,
    episode_outcomes: { ok: 12 },
    ads_filtered_count: null,
    dialogue_insights_dropped_count: null,
    topics_normalized_count: null,
    entity_kinds_repaired_count: null,
    ad_chars_excised_preroll: null,
    ad_chars_excised_postroll: null,
    ad_episodes_with_excision_count: null,
    ...over,
  }
}

type Props = InstanceType<typeof PipelineRunHistoryStrip>['$props']

function mountStrip(props: Props) {
  return mount(PipelineRunHistoryStrip, { props })
}

const options = (w: ReturnType<typeof mountStrip>) => w.findAll('[role="option"]')
const selected = (w: ReturnType<typeof mountStrip>) =>
  options(w)
    .filter((o) => o.attributes('aria-selected') === 'true')
    .map((o) => o.text())

describe('PipelineRunHistoryStrip', () => {
  it('shows the empty message when there are no runs', () => {
    const w = mountStrip({ runs: [] })
    expect(w.text()).toContain('No run.json summaries found for this corpus yet.')
    expect(w.find('[role="listbox"]').exists()).toBe(false)
  })

  it('shows the card heading unless embedded', () => {
    expect(mountStrip({ runs: [] }).find('h3').text()).toBe('Run history')
    const emb = mountStrip({ runs: [], embedded: true })
    expect(emb.find('h3').exists()).toBe(false)
    expect(emb.get('[data-testid="pipeline-run-history-strip"]').classes()).not.toContain('border')
  })

  it('lists runs newest first, tie-breaking on path descending, and selects the newest', () => {
    const w = mountStrip({
      runs: [
        run('old', { created_at: '2026-06-01T00:00:00Z' }),
        run('a', { created_at: '2026-06-10T10:00:00Z' }),
        run('b', { created_at: '2026-06-10T10:00:00Z' }),
        run('none', { created_at: null }),
      ],
    })
    const text = options(w).map((o) => o.text())
    expect(text).toHaveLength(4)
    expect(text[0]).toContain('feeds/a/b/run.json')
    expect(text[1]).toContain('feeds/a/a/run.json')
    expect(text[2]).toContain('feeds/a/old/run.json')
    expect(text[3]).toContain('feeds/a/none/run.json')
    expect(selected(w)).toEqual([text[0]])
  })

  it('formats each row label with episode count, outcome and path', () => {
    const w = mountStrip({ runs: [run('a', { episodes_scraped_total: 1500 })] })
    expect(options(w)[0].text()).toContain(`${(1500).toLocaleString()} ep · succeeded · feeds/a/a/run.json`)
  })

  it('uses a dash when the run has no timestamp and truncates very long paths', () => {
    const longPath = `feeds/${'x'.repeat(100)}/run.json`
    const w = mountStrip({ runs: [run('a', { created_at: null, relative_path: longPath })] })
    const label = options(w)[0].text()
    expect(label.startsWith('— ·')).toBe(true)
    expect(label).toContain('…')
    expect(label).not.toContain(longPath)
  })

  describe('outcome classification', () => {
    it.each([
      ['succeeded', { ok: 5 }, 0, 'bg-success'],
      ['partial', { ok: 5, failed: 1 }, 0, 'bg-warning'],
      ['partial', { skipped: 2, failed: 1 }, 0, 'bg-warning'],
      ['failed', { failed: 2 }, 0, 'bg-danger'],
      ['failed', {}, 3, 'bg-danger'],
      ['succeeded', {}, 0, 'bg-success'],
    ])('%s for outcomes %j errors %d', (key, outcomes, errors, cls) => {
      const w = mountStrip({ runs: [run('a', { episode_outcomes: outcomes, errors_total: errors })] })
      const dot = w.get('[data-testid="pipeline-run-dot"]')
      expect(dot.attributes('data-run-outcome')).toBe(key)
      expect(dot.classes()).toContain(cls)
      expect(dot.attributes('title')).toContain(key)
    })

    it('puts duration and episode count in the dot tooltip', () => {
      const w = mountStrip({ runs: [run('a')] })
      expect(w.get('[data-testid="pipeline-run-dot"]').attributes('title')).toMatch(
        /^Run .+ · 12 episodes · 2m 5s · succeeded$/,
      )
    })

    it('uses dashes in the tooltip for missing time and duration', () => {
      const w = mountStrip({ runs: [run('a', { created_at: null, run_duration_seconds: null })] })
      expect(w.get('[data-testid="pipeline-run-dot"]').attributes('title')).toBe(
        'Run — · 12 episodes · — · succeeded',
      )
    })
  })

  describe('insight line', () => {
    it('says every loaded run succeeded', () => {
      const w = mountStrip({ runs: [run('a'), run('b')] })
      expect(w.get('[data-testid="pipeline-run-history-summary-line"]').text()).toContain(
        'All 2 loaded runs succeeded',
      )
    })

    it('breaks down success, partial and failed counts', () => {
      const w = mountStrip({
        runs: [
          run('a'),
          run('b', { episode_outcomes: { ok: 1, failed: 1 } }),
          run('c', { episode_outcomes: { failed: 1 } }),
        ],
      })
      expect(w.text()).toContain('3 loaded runs: 1 success, 1 partial, 1 failed')
    })

    it('summarises the selected run: short label, episodes, wall time and date', () => {
      const w = mountStrip({ runs: [run('abcdefghijklmnop')] })
      const line = w.get('[data-testid="pipeline-run-history-summary-line"]').text()
      expect(line).toContain('abcdefgh…')
      expect(line).toContain('12 ep')
      expect(line).toContain('wall 2m 5s')
    })

    it('omits the date when the run has no timestamp and dashes the wall time', () => {
      const w = mountStrip({ runs: [run('a', { created_at: null, run_duration_seconds: null })] })
      const line = w.get('[data-testid="pipeline-run-history-summary-line"]').text()
      expect(line).toContain('wall —')
    })
  })

  describe('shortRunLabel fallbacks', () => {
    it('keeps a short run id verbatim', () => {
      const w = mountStrip({ runs: [run('short')] })
      expect(w.get('[data-testid="pipeline-run-history-summary-line"]').text()).toContain('short')
    })

    it('falls back to the folder name for a run.json path when run_id is blank', () => {
      const w = mountStrip({
        runs: [run('', { relative_path: 'feeds/a/run_20260610/run.json' })],
      })
      expect(w.get('[data-testid="pipeline-run-history-summary-line"]').text()).toContain('run_20260610')
    })

    it('truncates a long folder name', () => {
      const w = mountStrip({
        runs: [run('', { relative_path: `feeds/a/${'f'.repeat(20)}/run.json` })],
      })
      expect(w.get('[data-testid="pipeline-run-history-summary-line"]').text()).toContain(
        `${'f'.repeat(12)}…`,
      )
    })

    it('uses the path tail for non-run.json files, truncating long ones', () => {
      const short = mountStrip({ runs: [run('', { relative_path: 'a/summary.json' })] })
      expect(short.get('[data-testid="pipeline-run-history-summary-line"]').text()).toContain('summary.json')
      const long = mountStrip({ runs: [run('', { relative_path: `a/${'z'.repeat(30)}.json` })] })
      expect(long.get('[data-testid="pipeline-run-history-summary-line"]').text()).toContain(
        `${'z'.repeat(16)}…`,
      )
    })

    it('falls back to a bare run.json path', () => {
      const w = mountStrip({ runs: [run('', { relative_path: 'run.json' })] })
      expect(w.get('[data-testid="pipeline-run-history-summary-line"]').text()).toContain('run.json')
    })
  })

  describe('detail panel', () => {
    it('shows path and outcome breakdown', () => {
      const w = mountStrip({ runs: [run('a', { episode_outcomes: { ok: 3, failed: 1, skipped: 2 } })] })
      const d = w.get('[data-testid="pipeline-run-history-detail"]').text()
      expect(d).toContain('Path: feeds/a/a/run.json')
      expect(d).toContain('Episodes: 3 ok · 1 failed · 2 skipped')
    })

    it('shows episodes scraped when there are no outcomes', () => {
      const w = mountStrip({ runs: [run('a', { episode_outcomes: {}, episodes_scraped_total: 7 })] })
      expect(w.get('[data-testid="pipeline-run-history-detail"]').text()).toContain('Episodes scraped: 7')
    })

    it('shows errors only when non-zero', () => {
      expect(
        mountStrip({ runs: [run('a', { errors_total: 0 })] }).get('[data-testid="pipeline-run-history-detail"]').text(),
      ).not.toContain('Errors:')
      expect(
        mountStrip({ runs: [run('a', { errors_total: 1234 })] })
          .get('[data-testid="pipeline-run-history-detail"]')
          .text(),
      ).toContain(`Errors: ${(1234).toLocaleString()}`)
    })

    it('shows artifact counts with a dash for the missing side', () => {
      const w = mountStrip({ runs: [run('a', { gi_artifacts_generated: 4, kg_artifacts_generated: null })] })
      expect(w.get('[data-testid="pipeline-run-history-detail"]').text()).toMatch(/Artifacts:\s+GI 4\s+· KG —/)
      expect(
        mountStrip({ runs: [run('a')] }).get('[data-testid="pipeline-run-history-detail"]').text(),
      ).not.toContain('Artifacts:')
    })

    it('lists only the stage timings that are present', () => {
      const w = mountStrip({
        runs: [run('a', { time_scraping_seconds: 65, time_io_and_waiting_seconds: 3 })],
      })
      const d = w.get('[data-testid="pipeline-run-history-detail"]').text()
      expect(d).toContain('Stage time (metrics)')
      expect(d).toContain('Scraping 1m 5s')
      expect(d).toContain('I/O and wait 3s')
      expect(d).not.toContain('Parsing')
      expect(d).not.toContain('Normalizing')
    })

    it('shows parsing and normalizing timings when present', () => {
      const w = mountStrip({
        runs: [run('a', { time_parsing_seconds: 2, time_normalizing_seconds: 0 })],
      })
      const d = w.get('[data-testid="pipeline-run-history-detail"]').text()
      expect(d).toContain('Parsing 2s')
      expect(d).toContain('Normalizing 0s')
    })

    it('hides the stage section when no timing is a valid number', () => {
      const w = mountStrip({ runs: [run('a', { time_scraping_seconds: -1 })] })
      expect(w.get('[data-testid="pipeline-run-history-detail"]').text()).not.toContain('Stage time')
    })
  })

  describe('selection', () => {
    const runs = () => [
      run('first', { created_at: '2026-06-12T00:00:00Z' }),
      run('second', { created_at: '2026-06-11T00:00:00Z' }),
      run('third', { created_at: '2026-06-10T00:00:00Z' }),
    ]

    it('selects a run on click and updates the detail panel', async () => {
      const w = mountStrip({ runs: runs() })
      await options(w)[2].trigger('click')
      expect(selected(w)).toEqual([options(w)[2].text()])
      expect(w.get('[data-testid="pipeline-run-history-detail"]').text()).toContain('feeds/a/third/run.json')
    })

    it('moves with ArrowDown/ArrowUp, clamped at the ends, and jumps with Home/End', async () => {
      const w = mountStrip({ runs: runs() })
      const list = w.get('[role="listbox"]')
      const pathNow = () => w.get('[data-testid="pipeline-run-history-detail"]').text()
      await list.trigger('keydown', { key: 'ArrowDown' })
      expect(pathNow()).toContain('/second/')
      await list.trigger('keydown', { key: 'ArrowDown' })
      await list.trigger('keydown', { key: 'ArrowDown' })
      expect(pathNow()).toContain('/third/')
      await list.trigger('keydown', { key: 'ArrowUp' })
      expect(pathNow()).toContain('/second/')
      await list.trigger('keydown', { key: 'Home' })
      expect(pathNow()).toContain('/first/')
      await list.trigger('keydown', { key: 'ArrowUp' })
      expect(pathNow()).toContain('/first/')
      await list.trigger('keydown', { key: 'End' })
      expect(pathNow()).toContain('/third/')
      await list.trigger('keydown', { key: 'x' })
      expect(pathNow()).toContain('/third/')
    })

    it('prefers highlightRelativePath and follows it when it changes', async () => {
      const w = mountStrip({ runs: runs(), highlightRelativePath: 'feeds/a/third/run.json' })
      expect(w.get('[data-testid="pipeline-run-history-detail"]').text()).toContain('/third/')
      await w.setProps({ highlightRelativePath: 'feeds/a/second/run.json' })
      expect(w.get('[data-testid="pipeline-run-history-detail"]').text()).toContain('/second/')
    })

    it('falls back to the newest run when the highlight path is unknown', () => {
      const w = mountStrip({ runs: runs(), highlightRelativePath: 'nope' })
      expect(w.get('[data-testid="pipeline-run-history-detail"]').text()).toContain('/first/')
    })

    it('keeps the current selection when the runs list refreshes', async () => {
      const w = mountStrip({ runs: runs() })
      await options(w)[1].trigger('click')
      await w.setProps({ runs: [...runs(), run('fourth', { created_at: '2026-06-13T00:00:00Z' })] })
      expect(w.get('[data-testid="pipeline-run-history-detail"]').text()).toContain('/second/')
    })

    it('reselects the newest run when the selected run disappears', async () => {
      const w = mountStrip({ runs: runs() })
      await options(w)[2].trigger('click')
      await w.setProps({ runs: runs().slice(0, 2) })
      expect(w.get('[data-testid="pipeline-run-history-detail"]').text()).toContain('/first/')
    })

    it('clears the detail when runs become empty', async () => {
      const w = mountStrip({ runs: runs() })
      await w.setProps({ runs: [] })
      expect(w.find('[data-testid="pipeline-run-history-detail"]').exists()).toBe(false)
      expect(w.text()).toContain('No run.json summaries found')
    })

    it('ignores keyboard navigation when the filter matches nothing', async () => {
      const w = mountStrip({ runs: runs() })
      await w.get('[data-testid="pipeline-run-history-filter"]').setValue('zzz-no-match')
      expect(options(w)).toHaveLength(1) // the selected run is kept visible
      await w.get('[role="listbox"]').trigger('keydown', { key: 'End' })
      expect(w.get('[data-testid="pipeline-run-history-detail"]').text()).toContain('/first/')
    })
  })

  describe('filtering and caps', () => {
    it('filters by path, run id or date substring, case-insensitively', async () => {
      const w = mountStrip({
        runs: [
          run('alpha', { created_at: '2026-01-01T00:00:00Z' }),
          run('beta', { created_at: '2026-02-02T00:00:00Z' }),
        ],
      })
      const input = w.get('[data-testid="pipeline-run-history-filter"]')
      await input.setValue('ALPHA')
      expect(options(w).map((o) => o.text()).filter((t) => t.includes('/alpha/'))).toHaveLength(1)
      await input.setValue('2026-02')
      expect(options(w).some((o) => o.text().includes('/beta/'))).toBe(true)
      await input.setValue('feeds/a/')
      expect(options(w)).toHaveLength(2)
    })

    it('always keeps the selected run visible even when the filter excludes it', async () => {
      const w = mountStrip({
        runs: [
          run('alpha', { created_at: '2026-02-01T00:00:00Z' }),
          run('beta', { created_at: '2026-01-01T00:00:00Z' }),
        ],
      })
      await w.get('[data-testid="pipeline-run-history-filter"]').setValue('beta')
      const texts = options(w).map((o) => o.text())
      expect(texts).toHaveLength(2)
      expect(texts[0]).toContain('/alpha/')
    })

    it('caps the unfiltered list at 120 and explains it', () => {
      const many = Array.from({ length: 130 }, (_, i) =>
        run(`r${String(i).padStart(3, '0')}`, {
          created_at: new Date(Date.UTC(2026, 0, 1) + i * 60_000).toISOString(),
        }),
      )
      const w = mountStrip({ runs: many })
      expect(options(w)).toHaveLength(120)
      expect(w.get('[data-testid="pipeline-run-history-picker-hint"]').text()).toContain(
        'Showing newest 120 of 130',
      )
    })

    it('caps filtered results at 250 and explains it', async () => {
      const many = Array.from({ length: 260 }, (_, i) =>
        run(`r${i}`, { created_at: new Date(Date.UTC(2026, 0, 1) + i * 60_000).toISOString() }),
      )
      const w = mountStrip({ runs: many })
      await w.get('[data-testid="pipeline-run-history-filter"]').setValue('feeds/a')
      expect(options(w)).toHaveLength(250)
      expect(w.get('[data-testid="pipeline-run-history-picker-hint"]').text()).toContain(
        'Showing first 250 matches',
      )
    })

    it('shows no hint for a small list', () => {
      const w = mountStrip({ runs: [run('a')] })
      expect(w.find('[data-testid="pipeline-run-history-picker-hint"]').exists()).toBe(false)
    })
  })
})
