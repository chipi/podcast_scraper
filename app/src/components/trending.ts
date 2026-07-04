/** A corpus topic that's "heating up" (temporal_velocity), shared by the Home
 *  trending views. `series` is its monthly counts aligned to a common month axis. */
export interface RisingTopic {
  id: string
  label: string
  /** velocity_last_over_6mo, rounded to 1dp (e.g. 2.1 → "2.1×"). */
  v: number
  total: number
  series: number[]
}
