/** One line on the Index `IndexTimeseriesChart` — `data` aligns to the shared
 *  `labels` month axis by index. */
export interface TimeseriesSeries {
  key: string
  label: string
  /** One value per `labels` entry, same order/length. */
  data: number[]
  /** Whether the series starts visible (default true). */
  defaultEnabled?: boolean
}
