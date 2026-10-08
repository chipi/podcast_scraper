<script setup lang="ts">
/**
 * Privacy policy (#2210). Written from what the app ACTUALLY does — the per-user data audit of
 * 2026-10-04 (#2273) and the Play Data safety answers — not from a template. If the code changes
 * what it collects, this page and the store declarations must change with it.
 *
 * Deliberately English-only for now: a legal text translated by the i18n pipeline is a text nobody
 * reviewed in that language.
 *
 * DRAFT: the controller is a placeholder until the company exists (operator, 2026-10-04), and
 * backup retention is not yet enforced. `DRAFT_OPEN` drives the visible notice — empty it only
 * when both are true.
 */
const CONTACT = 'info@closelistening.app'
const DRAFT_OPEN = [
  'The company responsible for your data (the "controller") is being registered; its legal name and address will be added here before public release.',
  'How long backups are kept is being set; until then, deleted data can remain in backups.',
]
const UPDATED = '4 October 2026'
</script>

<template>
  <article class="lp-prose space-y-5 text-sm leading-relaxed" data-testid="privacy-policy">
    <div
      v-if="DRAFT_OPEN.length"
      class="rounded-2xl border border-border p-4"
      role="note"
      data-testid="privacy-draft-notice"
    >
      <p class="mb-2 font-bold">Draft — for beta testers</p>
      <ul class="list-disc space-y-1 pl-5">
        <li v-for="item in DRAFT_OPEN" :key="item">{{ item }}</li>
      </ul>
    </div>

    <p class="text-muted">Last updated {{ UPDATED }}.</p>

    <p>
      Close Listening helps you listen to podcasts more closely: transcripts, the people and ideas in
      each episode, and your own highlights and notes. This page explains what data the app
      collects, why, who else handles it, and what you can do about it.
    </p>

    <section>
      <h2 class="mb-1 text-base font-bold">Who is responsible</h2>
      <p>
        <strong>[Company name and registered address — to be added before public release.]</strong>
        For anything about your data, email <a :href="`mailto:${CONTACT}`">{{ CONTACT }}</a>.
      </p>
    </section>

    <section>
      <h2 class="mb-1 text-base font-bold">What we collect</h2>
      <ul class="list-disc space-y-1 pl-5">
        <li>
          <strong>Your account:</strong> your name, email address, how you sign in (Google, Apple
          or an email link) and the account identifier that sign-in method gives us, and your
          profile picture if you add one or your Google account has one.
        </li>
        <li>
          <strong>What you create:</strong> highlights, notes, boards, your queue, saved
          items, the shows, people and topics you follow, and your interests.
        </li>
        <li>
          <strong>What you listen to:</strong> playback positions, which episodes you finished,
          listening time, listening history and your searches.
        </li>
        <li>
          <strong>Notifications:</strong> your email and notification settings and, if you turn
          push notifications on, your device's push token.
        </li>
        <li>
          <strong>Crashes and diagnostics:</strong> error reports, and how and why the app last
          closed (for example, the system ending it to free memory), with the app version and
          device type.
        </li>
        <li>
          <strong>Usage statistics:</strong> which screens you open and what you do in the app,
          under a random identifier that is not your name or email. Your approximate location
          (country and city) is derived from your IP address for these statistics; we never use
          precise location. You can turn usage statistics off in <em>Settings › Privacy</em>.
        </li>
      </ul>
      <p class="mt-2">
        Note dictation uses your phone's own speech recognition (from Apple or Google); we receive
        only the resulting text, as a note. We do not access your contacts, photos (other than a
        profile picture you choose), calendar or files.
      </p>
    </section>

    <section>
      <h2 class="mb-1 text-base font-bold">Why we use it</h2>
      <ul class="list-disc space-y-1 pl-5">
        <li>To run the app and your account — signing you in, keeping your library, highlights and notes, resuming where you left off.</li>
        <li>To personalise it — recommendations, "Your week" and new-episode alerts come from what you listen to and follow.</li>
        <li>To send you what you asked for — sign-in links, digests and notifications, each of which you control in Profile.</li>
        <li>To keep it working — crash reports and diagnostics to find and fix problems; usage statistics to see what is used.</li>
      </ul>
      <p class="mt-2">
        We do not show ads, do not sell your data, and do not share it with advertisers or data
        brokers.
      </p>
    </section>

    <section>
      <h2 class="mb-1 text-base font-bold">Who else handles it</h2>
      <p class="mb-1">
        The app, its usage statistics and its error reports run on servers we operate. These
        service providers process data on our behalf, only for the purpose given:
      </p>
      <ul class="list-disc space-y-1 pl-5">
        <li><strong>Google</strong> and <strong>Apple</strong> — sign-in, if you choose them.</li>
        <li><strong>Apple Push Notification service</strong> and <strong>Google Firebase Cloud Messaging</strong> — delivering push notifications.</li>
        <li><strong>Resend</strong> — delivering emails.</li>
      </ul>
    </section>

    <section>
      <h2 class="mb-1 text-base font-bold">How long we keep it</h2>
      <ul class="list-disc space-y-1 pl-5">
        <li>Your account and what you created: until you delete it, or delete your account.</li>
        <li>Sent emails and notifications: 30 days; sign-in emails: 1 day.</li>
        <li>Records of security-relevant actions (such as an account being deleted) are kept to protect the service; they hold an internal identifier, not your email.</li>
        <li>Backups: <strong>[retention period — to be set before public release]</strong>.</li>
      </ul>
    </section>

    <section>
      <h2 class="mb-1 text-base font-bold">Your choices and rights</h2>
      <ul class="list-disc space-y-1 pl-5">
        <li><strong>Delete your account</strong> at any time: <em>Profile › Account › Delete account</em>. It takes effect immediately.</li>
        <li><strong>Delete some data</strong> without deleting the account: highlights, notes, boards, saved items and follows where they appear, and <em>Profile › Account › Clear listening history</em>.</li>
        <li><strong>Turn off</strong> usage statistics (Settings › Privacy), emails and push notifications (Profile).</li>
        <li><strong>Correct</strong> your name in Profile.</li>
        <li>
          Ask us for a copy of your data, or anything else about it, at
          <a :href="`mailto:${CONTACT}`">{{ CONTACT }}</a>. You can also complain to your data
          protection authority.
        </li>
      </ul>
    </section>

    <section>
      <h2 class="mb-1 text-base font-bold">Age</h2>
      <p>Close Listening is for people aged 16 and over.</p>
    </section>

    <section>
      <h2 class="mb-1 text-base font-bold">Security</h2>
      <p>All data travels over encrypted connections (HTTPS/TLS).</p>
    </section>

    <section>
      <h2 class="mb-1 text-base font-bold">Changes</h2>
      <p>If this policy changes, we will update the date above and, for significant changes, tell you in the app.</p>
    </section>
  </article>
</template>
