/** Split / merge top-level ``profile:`` for operator YAML (no full YAML parser). */

/** Quote a scalar for a one-line ``profile:`` value when needed (safe YAML subset). */
export function yamlScalarForProfileLine(name: string): string {
  const s = name.trim()
  if (!s) {
    return '""'
  }
  if (/^[A-Za-z0-9_.-]+$/.test(s)) {
    return s
  }
  return JSON.stringify(s)
}

export function splitOperatorYamlProfile(content: string): { profile: string; body: string } {
  const raw = content.replace(/\r\n/g, '\n')
  const lines = raw.split('\n')
  for (let i = 0; i < lines.length; i += 1) {
    const m = lines[i].match(/^\s*profile:\s*(.+?)\s*$/)
    if (m) {
      let v = m[1].trim()
      const hash = v.indexOf('#')
      if (hash >= 0) {
        v = v.slice(0, hash).trim()
      }
      if (
        (v.startsWith('"') && v.endsWith('"')) ||
        (v.startsWith("'") && v.endsWith("'"))
      ) {
        v = v.slice(1, -1)
      }
      const rest = [...lines.slice(0, i), ...lines.slice(i + 1)].join('\n')
      return { profile: v, body: rest.replace(/^\n+/, '').replace(/\n+$/, '') }
    }
  }
  return { profile: '', body: raw.replace(/\n+$/, '') }
}

export function mergeOperatorYamlProfile(profile: string, body: string): string {
  const b = body.replace(/^\n+/, '').replace(/\n+$/, '')
  if (!profile.trim()) {
    return b ? `${b}\n` : ''
  }
  const p = yamlScalarForProfileLine(profile.trim())
  if (!b) {
    return `profile: ${p}\n`
  }
  return `profile: ${p}\n${b}\n`
}
