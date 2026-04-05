/**
 * Chart.js external tooltip mounted on document.body so it is not clipped by
 * scrollable parents (e.g. dashboard overflow-y-auto).
 */
import type { Chart, TooltipModel } from 'chart.js'

const Z = 10000

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

function tooltipElementForChart(chart: Chart): HTMLDivElement {
  const id = `chartjs-external-tip-${String(chart.id)}`
  let el = document.getElementById(id) as HTMLDivElement | null
  if (!el) {
    el = document.createElement('div')
    el.id = id
    el.setAttribute('role', 'tooltip')
    Object.assign(el.style, {
      position: 'fixed',
      zIndex: String(Z),
      pointerEvents: 'none',
      opacity: '0',
      transition: 'opacity 0.12s ease',
      background: 'rgba(15, 23, 42, 0.94)',
      color: '#e2e8f0',
      borderRadius: '6px',
      border: '1px solid rgba(148, 163, 184, 0.35)',
      padding: '8px 12px',
      fontSize: '12px',
      lineHeight: '1.45',
      maxWidth: 'min(280px, calc(100vw - 16px))',
      boxShadow: '0 10px 40px rgba(0,0,0,0.35)',
    })
    document.body.appendChild(el)
  }
  return el
}

export function removeChartExternalTooltip(chart: Chart): void {
  document.getElementById(`chartjs-external-tip-${String(chart.id)}`)?.remove()
}

export function chartExternalTooltipHandler(context: {
  chart: Chart
  tooltip: TooltipModel<'bar'>
}): void {
  const { chart, tooltip } = context
  const el = tooltipElementForChart(chart)

  if (tooltip.opacity === 0) {
    el.style.opacity = '0'
    return
  }

  const blocks: string[] = []
  const titles = tooltip.title || []
  if (titles.length) {
    blocks.push(
      `<div style="font-weight:600;margin-bottom:4px;white-space:pre-wrap">${escapeHtml(
        titles.join('\n'),
      )}</div>`,
    )
  }
  for (const body of tooltip.body || []) {
    const line = body.lines.join('\n')
    if (line) {
      blocks.push(`<div style="white-space:pre-wrap">${escapeHtml(line)}</div>`)
    }
  }
  el.innerHTML = blocks.join('') || '&nbsp;'

  const rect = chart.canvas.getBoundingClientRect()
  const vw = window.innerWidth
  const vh = window.innerHeight
  const PAD = 8
  const GAP = 12

  el.style.opacity = '1'
  el.style.maxWidth = `min(280px, calc(100vw - ${String(2 * PAD)}px))`

  // Prefer below/right of caret; measure then clamp / flip so nothing leaves the viewport
  let left = rect.left + tooltip.caretX + GAP
  let top = rect.top + tooltip.caretY + GAP
  el.style.left = `${left}px`
  el.style.top = `${top}px`

  let w = el.offsetWidth
  let h = el.offsetHeight

  if (left + w > vw - PAD) {
    left = rect.left + tooltip.caretX - w - GAP
  }
  if (left + w > vw - PAD) {
    left = vw - w - PAD
  }
  if (left < PAD) {
    left = PAD
  }

  if (top + h > vh - PAD) {
    top = rect.top + tooltip.caretY - h - GAP
  }
  if (top + h > vh - PAD) {
    top = vh - h - PAD
  }
  if (top < PAD) {
    top = PAD
  }

  el.style.left = `${left}px`
  el.style.top = `${top}px`
}
