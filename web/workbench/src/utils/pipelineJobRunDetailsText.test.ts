import { describe, expect, it } from 'vitest'
import { pipelineJobRunDetailsText } from './pipelineJobRunDetailsText'
import type { PipelineJobRow } from '../api/jobsApi'

function row(over: Partial<PipelineJobRow> = {}): PipelineJobRow {
  return {
    job_id: 'j1',
    command_type: 'pipeline',
    status: 'succeeded',
    created_at: '2026-01-01T00:00:00Z',
    started_at: null,
    ended_at: null,
    pid: null,
    argv_summary: '',
    exit_code: null,
    log_relpath: '',
    error_reason: null,
    ...over,
  }
}

describe('pipelineJobRunDetailsText', () => {
  it('renders a minimal row as a single command_type line with an em-dash on empty', () => {
    const text = pipelineJobRunDetailsText(row({ command_type: '' }))
    expect(text).toBe('command_type: —')
  })

  it('includes error_reason on its own line when the row failed', () => {
    const text = pipelineJobRunDetailsText(
      row({ command_type: 'ingest', error_reason: 'timeout waiting for feed' }),
    )
    expect(text).toContain('command_type: ingest')
    expect(text).toContain('error_reason: timeout waiting for feed')
  })

  it('parses argv_summary as JSON and prints each element on its own indexed line', () => {
    const text = pipelineJobRunDetailsText(
      row({ argv_summary: JSON.stringify(['scrape', '--config', 'ops/x.yaml']) }),
    )
    expect(text).toContain('argv (exact subprocess):')
    expect(text).toContain('  [0] scrape')
    expect(text).toContain('  [1] --config')
    expect(text).toContain('  [2] ops/x.yaml')
    expect(text).toContain('operator_yaml (--config): ops/x.yaml')
  })

  it('falls back to argv_summary raw when the payload is not valid JSON', () => {
    const text = pipelineJobRunDetailsText(row({ argv_summary: 'not-json-{{' }))
    expect(text).toContain('argv_summary (raw): not-json-{{')
    expect(text).not.toContain('argv (exact subprocess):')
  })

  it('falls back to argv_summary raw when the payload is JSON but not an array', () => {
    const text = pipelineJobRunDetailsText(
      row({ argv_summary: JSON.stringify({ not: 'an-array' }) }),
    )
    expect(text).toContain('argv_summary (raw):')
    expect(text).not.toContain('argv (exact subprocess):')
  })

  it('omits the operator_yaml line when --config has no following value', () => {
    const text = pipelineJobRunDetailsText(row({ argv_summary: JSON.stringify(['scrape', '--config']) }))
    expect(text).toContain('argv (exact subprocess):')
    expect(text).not.toContain('operator_yaml')
  })

  it('appends log_relpath when present', () => {
    const text = pipelineJobRunDetailsText(row({ log_relpath: 'logs/j1.log' }))
    expect(text).toContain('log_relpath: logs/j1.log')
  })
})
