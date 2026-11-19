import { CronExpressionParser } from 'cron-parser'

/**
 * Cron helpers for the scheduled-jobs surface (#709): validate an expression and
 * preview its next fire times. The server's `next_run_at` is authoritative when
 * the scheduler is running; these run client-side to flag an **invalid cron**
 * (distinct from "disabled") and to preview upcoming runs.
 */
export function isValidCron(expr: string | null | undefined): boolean {
  const e = expr?.trim()
  if (!e) return false
  try {
    CronExpressionParser.parse(e)
    return true
  } catch {
    return false
  }
}

export function nextCronRuns(
  expr: string | null | undefined,
  count = 3,
  opts: { tz?: string; currentDate?: Date } = {},
): string[] | null {
  const e = expr?.trim()
  if (!e) return null
  try {
    const it = CronExpressionParser.parse(e, {
      tz: opts.tz,
      currentDate: opts.currentDate,
    })
    const out: string[] = []
    for (let i = 0; i < count; i += 1) {
      out.push(it.next().toDate().toISOString())
    }
    return out
  } catch {
    return null
  }
}
