import { describe, expect, it } from 'vitest'
import type { CorpusManifestFeed, CorpusRunSummaryItem } from '../api/corpusMetricsApi'
import {
  feedRowRunRelativePaths,
  runPathContainsFeedDir,
  stableFeedDirForUrl,
  uniqueRunForFeedDir,
} from './feedRunLinking'

function run(relativePath: string, run_id = 'r'): CorpusRunSummaryItem {
  return {
    relative_path: relativePath,
    run_id,
    created_at: null,
    episode_outcomes: {},
  } as CorpusRunSummaryItem
}

describe('runPathContainsFeedDir', () => {
  it('matches the typical feeds/<dir>/run.json layout', () => {
    expect(runPathContainsFeedDir('feeds/pod/run_2024/run.json', 'pod')).toBe(true)
    expect(runPathContainsFeedDir('feeds/pod/run_2024/run.json', 'run_2024')).toBe(true)
    expect(runPathContainsFeedDir('feeds/x/run.json', 'y')).toBe(false)
  })

  it('matches when feeds/<dir>/ is the leading path segment', () => {
    expect(runPathContainsFeedDir('feeds/pod/run.json', 'pod')).toBe(true)
  })

  it('matches feeds/<dir>/ nested deeper in the path (the /feeds/<s>/ branch)', () => {
    expect(runPathContainsFeedDir('corpus/feeds/pod/run.json', 'pod')).toBe(true)
  })

  it('matches a path ending in /<dir>/run.json without a feeds/ prefix', () => {
    expect(runPathContainsFeedDir('something/else/pod/run.json', 'pod')).toBe(true)
  })

  it('matches a path ending in <dir>/run.json with no leading slash (root-relative)', () => {
    expect(runPathContainsFeedDir('pod/run.json', 'pod')).toBe(true)
  })

  it('matches /<dir>/run.json appearing mid-path (the includes branch)', () => {
    expect(runPathContainsFeedDir('a/pod/run.json/trailing', 'pod')).toBe(true)
  })

  it('is case-insensitive', () => {
    expect(runPathContainsFeedDir('Feeds/POD/run.json', 'pod')).toBe(true)
    expect(runPathContainsFeedDir('feeds/pod/run.json', 'POD')).toBe(true)
  })

  it('normalizes Windows backslash separators', () => {
    expect(runPathContainsFeedDir('feeds\\pod\\run.json', 'pod')).toBe(true)
  })

  it('trims and strips surrounding slashes from the feed dir segment', () => {
    expect(runPathContainsFeedDir('feeds/pod/run.json', '  /pod/  ')).toBe(true)
  })

  it('returns false when the feed dir is empty or whitespace/slashes only', () => {
    expect(runPathContainsFeedDir('feeds/pod/run.json', '')).toBe(false)
    expect(runPathContainsFeedDir('feeds/pod/run.json', '   ')).toBe(false)
    expect(runPathContainsFeedDir('feeds/pod/run.json', '///')).toBe(false)
  })

  it('returns false when no segment matches', () => {
    expect(runPathContainsFeedDir('feeds/abc/run.json', 'pod')).toBe(false)
    expect(runPathContainsFeedDir('totally/unrelated/path.json', 'pod')).toBe(false)
  })

  it('does not match a dir name that is only a substring of a segment', () => {
    // ``pod`` must be a whole path segment, not a substring of ``podcast``.
    expect(runPathContainsFeedDir('feeds/podcast/run.json', 'pod')).toBe(false)
  })
})

describe('stableFeedDirForUrl', () => {
  it('resolves a manifest entry by normalized URL (host case + trailing slash)', () => {
    const dir = stableFeedDirForUrl(
      [{ feed_url: 'https://HOST/Podcast/', stable_feed_dir: 'abc' }],
      'https://host/podcast',
    )
    expect(dir).toBe('abc')
  })

  it('trims the returned stable_feed_dir', () => {
    const dir = stableFeedDirForUrl(
      [{ feed_url: 'https://x/feed', stable_feed_dir: '  fd  ' }],
      'https://x/feed',
    )
    expect(dir).toBe('fd')
  })

  it('matches non-URL feed strings via the lowercase/strip-slash fallback', () => {
    const dir = stableFeedDirForUrl(
      [{ feed_url: 'Not A URL///', stable_feed_dir: 'fd' }],
      'not a url',
    )
    expect(dir).toBe('fd')
  })

  it('ignores the query string and hash (origin + pathname only)', () => {
    const dir = stableFeedDirForUrl(
      [{ feed_url: 'https://x/feed?a=1#frag', stable_feed_dir: 'fd' }],
      'https://x/feed?b=2',
    )
    expect(dir).toBe('fd')
  })

  it('returns null for an empty / whitespace feed URL', () => {
    expect(stableFeedDirForUrl([{ feed_url: 'https://x/feed', stable_feed_dir: 'fd' }], '')).toBe(
      null,
    )
    expect(stableFeedDirForUrl([{ feed_url: 'https://x/feed', stable_feed_dir: 'fd' }], '   ')).toBe(
      null,
    )
  })

  it('returns null when no manifest entry matches', () => {
    expect(
      stableFeedDirForUrl(
        [{ feed_url: 'https://x/feed', stable_feed_dir: 'fd' }],
        'https://y/other',
      ),
    ).toBe(null)
  })

  it('skips entries with missing / non-string feed_url or stable_feed_dir', () => {
    const feeds = [
      { stable_feed_dir: 'fd' }, // no feed_url
      { feed_url: 'https://x/feed' }, // no stable_feed_dir
      { feed_url: 123 as unknown as string, stable_feed_dir: 'fd' }, // non-string url
      { feed_url: 'https://x/feed', stable_feed_dir: 42 as unknown as string }, // non-string dir
      { feed_url: 'https://x/feed', stable_feed_dir: '   ' }, // blank dir
      { feed_url: 'https://x/feed', stable_feed_dir: 'good' }, // the real match
    ] as CorpusManifestFeed[]
    expect(stableFeedDirForUrl(feeds, 'https://x/feed')).toBe('good')
  })

  it('returns null for an empty manifest list', () => {
    expect(stableFeedDirForUrl([], 'https://x/feed')).toBe(null)
  })
})

describe('uniqueRunForFeedDir', () => {
  it('returns null when zero or multiple runs match', () => {
    const runs = [run('feeds/a/run.json', '1'), run('feeds/a/other/run.json', '2')]
    expect(uniqueRunForFeedDir(runs, 'a')).toBe(null)
    expect(uniqueRunForFeedDir([], 'a')).toBe(null)
  })

  it('returns null when no run matches the dir', () => {
    expect(uniqueRunForFeedDir([run('feeds/b/run.json')], 'a')).toBe(null)
  })

  it('returns the single matching run', () => {
    const one = run('feeds/only/run.json', 'z')
    expect(uniqueRunForFeedDir([one], 'only')).toEqual(one)
  })
})

describe('feedRowRunRelativePaths', () => {
  it('maps feed URLs to run paths only with unique manifest + run hits', () => {
    const manifest: CorpusManifestFeed[] = [{ feed_url: 'https://x/feed', stable_feed_dir: 'fd' }]
    const runs = [run('feeds/fd/run.json', '1')]
    expect(feedRowRunRelativePaths(['https://x/feed'], runs, manifest)).toEqual(['feeds/fd/run.json'])
  })

  it('returns null per row when the manifest list is undefined', () => {
    const runs = [run('feeds/fd/run.json')]
    expect(feedRowRunRelativePaths(['a', 'b'], runs, undefined)).toEqual([null, null])
  })

  it('returns null per row when the manifest list is empty', () => {
    const runs = [run('feeds/fd/run.json')]
    expect(feedRowRunRelativePaths(['a', 'b'], runs, [])).toEqual([null, null])
  })

  it('returns null per row when there are no runs', () => {
    const manifest: CorpusManifestFeed[] = [{ feed_url: 'https://x/feed', stable_feed_dir: 'fd' }]
    expect(feedRowRunRelativePaths(['a', 'b'], [], manifest)).toEqual([null, null])
  })

  it('returns an empty array when there are no row URLs', () => {
    const manifest: CorpusManifestFeed[] = [{ feed_url: 'https://x/feed', stable_feed_dir: 'fd' }]
    const runs = [run('feeds/fd/run.json')]
    expect(feedRowRunRelativePaths([], runs, manifest)).toEqual([])
  })

  it('yields null for a row whose URL is not in the manifest', () => {
    const manifest: CorpusManifestFeed[] = [{ feed_url: 'https://x/feed', stable_feed_dir: 'fd' }]
    const runs = [run('feeds/fd/run.json')]
    expect(feedRowRunRelativePaths(['https://unknown/feed'], runs, manifest)).toEqual([null])
  })

  it('yields null for a mapped dir that has no unique run', () => {
    const manifest: CorpusManifestFeed[] = [{ feed_url: 'https://x/feed', stable_feed_dir: 'fd' }]
    const runs = [run('feeds/fd/run.json', '1'), run('feeds/fd/again/run.json', '2')]
    expect(feedRowRunRelativePaths(['https://x/feed'], runs, manifest)).toEqual([null])
  })

  it('handles a mix of resolvable and unresolvable rows', () => {
    const manifest: CorpusManifestFeed[] = [
      { feed_url: 'https://x/feed', stable_feed_dir: 'fd' },
      { feed_url: 'https://y/feed', stable_feed_dir: 'gd' },
    ]
    const runs = [run('feeds/fd/run.json', '1')] // only fd has a run
    expect(
      feedRowRunRelativePaths(['https://x/feed', 'https://y/feed', 'https://z/feed'], runs, manifest),
    ).toEqual(['feeds/fd/run.json', null, null])
  })
})
