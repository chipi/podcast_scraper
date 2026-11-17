import type { PipelineJobRow } from '../api/jobsApi'

function parsedArgv(row: PipelineJobRow): string[] {
  try {
    const raw = row.argv_summary?.trim()
    if (!raw) {
      return []
    }
    const a = JSON.parse(raw) as unknown
    if (!Array.isArray(a)) {
      return []
    }
    return a.map((x) => String(x))
  } catch {
    return []
  }
}

function operatorYamlPathFromArgv(argv: string[]): string | null {
  const i = argv.indexOf('--config')
  if (i >= 0 && i + 1 < argv.length) {
    const p = argv[i + 1]?.trim()
    return p || null
  }
  return null
}

/**
 * Multi-line text for the job “Command line and paths” disclosure (Jobs + Job history).
 *
 * When ``viewer_operator.yaml`` has a ``profile:`` line, the queued subprocess argv includes
 * ``--profile <name>`` before ``--config`` (same order as the README quick-start).
 */
export function pipelineJobRunDetailsText(row: PipelineJobRow): string {
  const lines: string[] = [`command_type: ${row.command_type || '—'}`]
  const err = row.error_reason?.trim()
  if (err) {
    lines.push(`error_reason: ${err}`)
  }
  const argv = parsedArgv(row)
  if (argv.length) {
    lines.push('argv (exact subprocess):')
    argv.forEach((s, i) => {
      lines.push(`  [${i}] ${s}`)
    })
    const opYaml = operatorYamlPathFromArgv(argv)
    if (opYaml) {
      lines.push(`operator_yaml (--config): ${opYaml}`)
    }
    lines.push(
      'Preset: --profile is copied from profile: in that YAML when present; explicit keys in the operator file still override the preset after merge.',
    )
  } else if (row.argv_summary) {
    lines.push(`argv_summary (raw): ${row.argv_summary}`)
  }
  if (row.log_relpath) {
    lines.push(`log_relpath: ${row.log_relpath}`)
  }
  return lines.join('\n')
}
