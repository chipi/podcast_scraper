/**
 * Flip the `enabled:` field of one `scheduled_jobs[]` entry in the operator YAML
 * text, by a line-rewrite that leaves surrounding lines (and comments) intact
 * (#709). Returns the new YAML, or `null` when the block / named entry isn't
 * found (caller then surfaces "edit in Job Configuration").
 *
 * Deliberately a targeted text edit, not a full YAML round-trip: it touches only
 * the one `enabled:` line (or inserts one), preserving operator comments — which
 * a parse-and-redump would strip.
 */
function indentOf(line: string): number {
  const m = /^( *)/.exec(line)
  return m ? m[1].length : 0
}

function stripQuotes(s: string): string {
  const t = s.trim()
  if ((t.startsWith('"') && t.endsWith('"')) || (t.startsWith("'") && t.endsWith("'"))) {
    return t.slice(1, -1)
  }
  return t
}

export interface ParsedScheduledJob {
  name: string
  cron: string
  enabled: boolean
}

/** Read the `key: value` for one entry line, stripping quotes + trailing comment. */
function entryValue(line: string, key: string): string | null {
  const m = new RegExp(`^ *(?:- +)?${key} *: *(.*)$`).exec(line)
  if (!m) return null
  let v = (m[1] ?? '').trim()
  if (v.startsWith('"') || v.startsWith("'")) {
    return stripQuotes(v.replace(/\s+#.*$/, ''))
  }
  const hash = v.indexOf('#')
  if (hash >= 0) v = v.slice(0, hash)
  return v.trim()
}

/**
 * Best-effort parse of the `scheduled_jobs:` block into `{name, cron, enabled}`
 * for the live cron preview in the Job Configuration editor (#709). Same
 * block-sequence assumptions as the toggle rewrite; returns `[]` when absent.
 */
export function parseScheduledJobsFromYaml(yamlText: string): ParsedScheduledJob[] {
  const lines = yamlText.split('\n')
  const sjIdx = lines.findIndex((l) => /^ *scheduled_jobs *:/.test(l))
  if (sjIdx < 0) return []
  const sjIndent = indentOf(lines[sjIdx]!)

  let blockEnd = lines.length
  for (let i = sjIdx + 1; i < lines.length; i += 1) {
    const l = lines[i]!
    if (l.trim() === '' || /^ *#/.test(l)) continue
    if (indentOf(l) <= sjIndent) {
      blockEnd = i
      break
    }
  }

  const itemStarts: number[] = []
  for (let i = sjIdx + 1; i < blockEnd; i += 1) {
    if (/^ *-(?: |$)/.test(lines[i]!) && indentOf(lines[i]!) > sjIndent) {
      itemStarts.push(i)
    }
  }

  const out: ParsedScheduledJob[] = []
  for (let k = 0; k < itemStarts.length; k += 1) {
    const start = itemStarts[k]!
    const end = k + 1 < itemStarts.length ? itemStarts[k + 1]! : blockEnd
    let name = ''
    let cron = ''
    let enabled = true
    for (let i = start; i < end; i += 1) {
      const nm = entryValue(lines[i]!, 'name')
      if (nm != null) name = nm
      const cr = entryValue(lines[i]!, 'cron')
      if (cr != null) cron = cr
      const en = entryValue(lines[i]!, 'enabled')
      if (en != null) enabled = en.toLowerCase() !== 'false'
    }
    if (name || cron) out.push({ name, cron, enabled })
  }
  return out
}

export function toggleScheduledJobEnabled(
  yamlText: string,
  name: string,
  enabled: boolean,
): string | null {
  const lines = yamlText.split('\n')
  const sjIdx = lines.findIndex((l) => /^ *scheduled_jobs *:/.test(l))
  if (sjIdx < 0) return null
  const sjIndent = indentOf(lines[sjIdx]!)

  // The block runs until the next non-blank, non-comment line at <= sjIndent.
  let blockEnd = lines.length
  for (let i = sjIdx + 1; i < lines.length; i += 1) {
    const l = lines[i]!
    if (l.trim() === '' || /^ *#/.test(l)) continue
    if (indentOf(l) <= sjIndent) {
      blockEnd = i
      break
    }
  }

  const itemStarts: number[] = []
  for (let i = sjIdx + 1; i < blockEnd; i += 1) {
    if (/^ *-(?: |$)/.test(lines[i]!) && indentOf(lines[i]!) > sjIndent) {
      itemStarts.push(i)
    }
  }
  if (itemStarts.length === 0) return null

  for (let k = 0; k < itemStarts.length; k += 1) {
    const start = itemStarts[k]!
    const end = k + 1 < itemStarts.length ? itemStarts[k + 1]! : blockEnd
    const itemIndent = indentOf(lines[start]!)

    let nameLine = -1
    for (let i = start; i < end; i += 1) {
      const m = /^ *(?:- +)?name *: *(.+?) *$/.exec(lines[i]!)
      if (m && stripQuotes(m[1]!) === name) {
        nameLine = i
        break
      }
    }
    if (nameLine < 0) continue

    // Replace an existing enabled: line (keep prefix + any trailing comment).
    for (let i = start; i < end; i += 1) {
      const m = /^( *(?:- +)?enabled *: *)([^#]*?)( *#.*)?$/.exec(lines[i]!)
      if (m) {
        lines[i] = `${m[1]}${enabled ? 'true' : 'false'}${m[3] ?? ''}`
        return lines.join('\n')
      }
    }

    // No enabled: field — insert one right after the name line.
    const fieldIndent = itemIndent + 2
    lines.splice(nameLine + 1, 0, `${' '.repeat(fieldIndent)}enabled: ${enabled ? 'true' : 'false'}`)
    return lines.join('\n')
  }
  return null
}
