/**
 * Copy ``text`` to the clipboard. Returns ``true`` on success, ``false`` when
 * neither the async Clipboard API nor the legacy ``execCommand`` fallback works
 * (insecure context, permission denied, headless environment).
 *
 * Extracted verbatim from EpisodeDetailPanel / NodeDetail (which carried
 * identical copies) so the copy affordance behaves the same everywhere — e.g.
 * the pipeline job-log viewer (#695).
 */
export async function copyTextToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    try {
      const ta = document.createElement('textarea')
      ta.value = text
      ta.setAttribute('readonly', '')
      ta.style.position = 'fixed'
      ta.style.opacity = '0'
      ta.style.left = '-9999px'
      document.body.appendChild(ta)
      ta.focus()
      ta.select()
      const ok = document.execCommand('copy')
      document.body.removeChild(ta)
      return ok
    } catch {
      return false
    }
  }
}
