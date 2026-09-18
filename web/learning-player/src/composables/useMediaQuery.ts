import { onUnmounted, readonly, ref, type Ref } from 'vue'

/**
 * A reactive media query, for the cases CSS cannot reach.
 *
 * Layout belongs in Tailwind's responsive classes — this is for when a COMPONENT PROP has to
 * differ by viewport, which no stylesheet can express. Home caps the Trends list at three rows on
 * a phone and five on a desktop: `collapsed` is a number passed into `DiscoveryExplorer`, so
 * `lg:` cannot do it.
 *
 * Reach for CSS first. If a `lg:hidden` or a `lg:grid-cols-2` will do, use that instead: a media
 * query in JS is state that can disagree with the stylesheet.
 *
 * Guards `window.matchMedia` like `utils/motion` does — the unit suite renders components in
 * environments where it is absent, and a missing API must mean "no match", not a crash.
 */
export function useMediaQuery(query: string): Readonly<Ref<boolean>> {
  const matches = ref(false)
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
    return readonly(matches)
  }
  const mql = window.matchMedia(query)
  matches.value = mql.matches
  const onChange = (e: MediaQueryListEvent): void => {
    matches.value = e.matches
  }
  mql.addEventListener('change', onChange)
  onUnmounted(() => mql.removeEventListener('change', onChange))
  return readonly(matches)
}

/** Tailwind's `lg` breakpoint — the width at which this app becomes two-column (UXS-011). */
export function useIsDesktop(): Readonly<Ref<boolean>> {
  return useMediaQuery('(min-width: 1024px)')
}
