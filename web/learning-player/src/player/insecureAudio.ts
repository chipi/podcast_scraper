/**
 * Whether audio failed because an https page asked for an http-only source (BBC's feeds,
 * 2026-10-15). A browser blocks it as mixed content, or upgrades it to an https address the host
 * refuses (403); a retry cannot help. The apps load their own bundle and answer this differently,
 * so it is a web-only verdict.
 */
export function isInsecureAudio(pageProtocol: string, src: string, native: boolean): boolean {
  return !native && pageProtocol === 'https:' && src.startsWith('http:')
}
