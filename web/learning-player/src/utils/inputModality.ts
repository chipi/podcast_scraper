/**
 * Was the last input a tap/click or a key? For focus a SHEET or PANEL hands back when it closes.
 *
 * Closing returns focus to the control that opened it, so a keyboard or screen-reader user keeps
 * their place. A programmatic `focus()` draws the `:focus-visible` ring in WKWebView however the
 * close happened, so after a TAP the Brief's tab sat outlined in yellow as if still selected
 * (operator 2026-10-11, on the phone). The ring belongs to keyboard use only.
 */
let last: "pointer" | "key" | null = null

if (typeof window !== "undefined") {
  window.addEventListener("pointerdown", () => (last = "pointer"), { capture: true })
  window.addEventListener("keydown", () => (last = "key"), { capture: true })
}

export function lastInputWasPointer(): boolean {
  return last === "pointer"
}

/** Focus `el` without scrolling, with the keyboard ring only when the last input was a key. */
export function returnFocus(el: HTMLElement | null | undefined): void {
  el?.focus?.({ preventScroll: true, focusVisible: !lastInputWasPointer() } as FocusOptions)
}
