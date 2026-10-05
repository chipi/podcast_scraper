/**
 * Which links people click in the emails we send (operator 2026-10-05).
 *
 * The email renderer (homelab delivery) tags every content link with
 * `utm_source=email&utm_campaign=<which email>&utm_content=<what kind of page>`. The click lands in
 * the web app or the installed app — the only place that can report it to Umami — so the app reads
 * the tags on arrival and reports ONE `email_link_opened` per click (`firstSighting`: a reload
 * in the same session is not a second click). The tags stay in the address bar: stripping them is
 * a second navigation, and it would race the landing scroll (`?t=`, `#notes`).
 *
 * Not Resend's click tracking, and not a redirect through our server: both bounce the tap through
 * another host first, and a phone only opens the installed app for a link that goes STRAIGHT to
 * closelistening.app (Universal Links / App Links).
 *
 * The values are closed enums — an unknown one is reported as `other`, never passed through — so a
 * hand-edited link cannot put free text into analytics. No user id rides the link: emails get
 * forwarded; who the person is comes from their session.
 */
export const EMAIL_CAMPAIGNS = [
  'your_week_digest',
  'recommendations_digest',
  'resurface_nudge',
  'daily_recap',
  'new_episodes',
  'other',
] as const
export type EmailCampaign = (typeof EMAIL_CAMPAIGNS)[number]

export const EMAIL_LINK_ELEMENTS = [
  'episode',
  'podcast',
  'topic',
  'person',
  'storyline',
  'theme',
  'other',
] as const
export type EmailLinkElement = (typeof EMAIL_LINK_ELEMENTS)[number]

export const TAG_KEYS = ['utm_source', 'utm_campaign', 'utm_content'] as const

export interface EmailLink {
  campaign: EmailCampaign
  element: EmailLinkElement
}

type Query = Record<string, unknown>

function oneOf<T extends string>(raw: unknown, allowed: readonly T[]): T | 'other' {
  return typeof raw === 'string' && (allowed as readonly string[]).includes(raw) ? (raw as T) : 'other'
}

/** The email tags on a route's query, or null when the visit did not come from one of our emails. */
export function emailLinkOf(query: Query | undefined): EmailLink | null {
  if (query?.utm_source !== 'email') return null
  return {
    campaign: oneOf(query.utm_campaign, EMAIL_CAMPAIGNS),
    element: oneOf(query.utm_content, EMAIL_LINK_ELEMENTS),
  }
}

const SEEN_PREFIX = 'lp.emailLinkSeen:'

/**
 * True the FIRST time this exact link is seen in this state this session — false on a reload or a
 * Back onto it. Keyed on signed-in state too, so the signed-out arrival at the gate and the
 * signed-in arrival after sign-in are both reported: that pair is the email → sign-in funnel.
 * Storage that throws (private mode) reports every time rather than never.
 */
export function firstSighting(fullPath: string, signedIn: boolean): boolean {
  const key = `${SEEN_PREFIX}${signedIn ? 1 : 0}:${fullPath}`
  try {
    if (sessionStorage.getItem(key)) return false
    sessionStorage.setItem(key, '1')
  } catch {
    /* no storage: better a rare double count than a lost click */
  }
  return true
}
