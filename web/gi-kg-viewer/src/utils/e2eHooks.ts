/**
 * Shared gate for the viewer's E2E inspection hooks
 * (``window.__GIKG_SUBJECT__`` / ``window.__GIKG_FSM__`` / etc).
 *
 * Two builds enable the hooks:
 *
 * 1. ``vite serve`` development — ``import.meta.env.DEV === true``.
 * 2. Stack-test production builds — produced by ``vite build`` with
 *    ``VITE_E2E_HOOKS=true`` set via the
 *    ``docker/viewer/Dockerfile`` build arg (wired in
 *    ``compose/docker-compose.stack-test.yml``). Production deploys leave
 *    the flag unset; the hooks are stripped via Vite's dead-code
 *    elimination on ``import.meta.env.VITE_E2E_HOOKS``.
 *
 * Always gate on ``typeof window !== 'undefined'`` in addition to this
 * helper to keep server-side / vitest-jsdom callers safe.
 */
export const e2eHooksEnabled =
  typeof import.meta !== 'undefined' &&
  (!!import.meta.env?.DEV || import.meta.env?.VITE_E2E_HOOKS === 'true')
