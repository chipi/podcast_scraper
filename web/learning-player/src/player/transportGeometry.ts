/**
 * The size of EVERY secondary control in the transport row — transcript, capture, ↺15, 30↻,
 * output route, speed (operator 2026-09-30: "if the buttons are smaller, they all have to be
 * smaller").
 *
 * The row drifted to four sizes because each control carried its own: speed and transcript at 44px,
 * the skips at 40px, the route picker at its mini-player 32px (a `h-11` passed from outside lost to
 * the component's own `h-8`). One constant, handed to every one of them, is what keeps it from
 * drifting again. `lp-tap` keeps the HIT area at 44px on phones, where the ink is 40px.
 */
export const TRANSPORT_BUTTON_SIZE = "lp-tap h-10 w-10 sm:h-11 sm:w-11"
