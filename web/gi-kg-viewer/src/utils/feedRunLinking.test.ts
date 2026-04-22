import { describe, expect, it } from 'vitest'
import type { CorpusRunSummaryItem } from '../api/corpusMetricsApi'
import {
  feedRowRunRelativePaths,
  runPathContainsFeedDir,
  stableFeedDirForUrl,
  uniqueRunForFeedDir,
} from './feedRunLinking'

describe('runPathContainsFeedDir', () => {
  it('matches typical feeds/<dir>/run.json layout', () => {
    expect(runPathContainsFeedDir('feeds/pod/run_2024/run.json', 'pod')).toBe(true)
    expect(runPathContainsFeedDir('feeds/pod/run_2024/run.json', 'run_2024')).toBe(true)
    expect(runPathContainsFeedDir('feeds/x/run.json', 'y')).toBe(false)
  })
})

describe('stableFeedDirForUrl', () => {
  it('resolves manifest entry by normalized URL', () => {
    const dir = stableFeedDirForUrl(
      [{ feed_url: 'https://HOST/Podcast/', stable_feed_dir: 'abc' }],
      'https://host/podcast',
    )
    expect(dir).toBe('abc')
  })
})

describe('uniqueRunForFeedDir', () => {
  it('returns null when zero or multiple runs match', () => {
    const runs = [
      { relative_path: 'feeds/a/run.json', run_id: '1', created_at: null, episode_outcomes: {} },
      { relative_path: 'feeds/a/other/run.json', run_id: '2', created_at: null, episode_outcomes: {} },
    ] as CorpusRunSummaryItem[]
    expect(uniqueRunForFeedDir(runs, 'a')).toBe(null)
    expect(uniqueRunForFeedDir([], 'a')).toBe(null)
  })

  it('returns the single matching run', () => {
    const one = {
      relative_path: 'feeds/only/run.json',
      run_id: 'z',
      created_at: null,
      episode_outcomes: {},
    } as CorpusRunSummaryItem
    expect(uniqueRunForFeedDir([one], 'only')).toEqual(one)
  })
})

describe('feedRowRunRelativePaths', () => {
  it('maps feed URLs to run paths only with unique manifest + run hits', () => {
    const manifest = [{ feed_url: 'https://x/feed', stable_feed_dir: 'fd' }]
    const runs = [
      { relative_path: 'feeds/fd/run.json', run_id: '1', created_at: null, episode_outcomes: {} },
    ] as CorpusRunSummaryItem[]
    const out = feedRowRunRelativePaths(['https://x/feed'], runs, manifest)
    expect(out).toEqual(['feeds/fd/run.json'])
  })
})
