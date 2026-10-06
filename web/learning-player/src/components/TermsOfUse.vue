<script setup lang="ts">
/**
 * Terms of use — first version (2026-10-05). Written from what the app actually does, like the
 * privacy policy beside it (#2210): beta status, accounts, podcasts that belong to their creators,
 * automatically generated summaries, your own notes and highlights, connected agents, deletion.
 *
 * Deliberately English-only for now: a legal text translated by the i18n pipeline is a text nobody
 * reviewed in that language.
 *
 * DRAFT: the company is not yet registered (operator 2026-10-04), so the contracting party and the
 * governing law are placeholders, and this first version has not been reviewed by a lawyer.
 * `DRAFT_OPEN` drives the visible notice — empty it only when all three are settled.
 */
const CONTACT = 'info@closelistening.app'
const DRAFT_OPEN = [
  'The company you are agreeing with is being registered; its legal name and address will be added here before public release.',
  'The governing law and courts will be set once the company is registered.',
  'This is a first version and has not yet been reviewed by a lawyer.',
]
const UPDATED = '5 October 2026'
</script>

<template>
  <article class="lp-prose space-y-5 text-sm leading-relaxed" data-testid="terms-of-use">
    <div
      v-if="DRAFT_OPEN.length"
      class="rounded-2xl border border-border p-4"
      role="note"
      data-testid="terms-draft-notice"
    >
      <p class="mb-2 font-bold">Draft — for beta testers</p>
      <ul class="list-disc space-y-1 pl-5">
        <li v-for="item in DRAFT_OPEN" :key="item">{{ item }}</li>
      </ul>
    </div>

    <p class="text-muted">Last updated {{ UPDATED }}.</p>

    <p>
      These terms cover your use of Close Listening — the app and website that help you listen to
      podcasts more closely. By creating an account or using the app, you agree to them. How we
      handle your data is described separately in our
      <RouterLink :to="{ name: 'about-page', params: { page: 'privacy' } }">Privacy policy</RouterLink>.
    </p>

    <section>
      <h2 class="mb-1 text-base font-bold">Who you are agreeing with</h2>
      <p>
        <strong>[Company name and registered address — to be added before public release.]</strong>
        Questions about these terms: <a :href="`mailto:${CONTACT}`">{{ CONTACT }}</a>.
      </p>
    </section>

    <section>
      <h2 class="mb-1 text-base font-bold">A service in beta</h2>
      <p>
        Close Listening is in testing. Features will change, some will be removed, and the service
        may be interrupted or lose data. Please do not rely on it as the only copy of anything that
        matters to you.
      </p>
    </section>

    <section>
      <h2 class="mb-1 text-base font-bold">Your account</h2>
      <ul class="list-disc space-y-1 pl-5">
        <li>You must be 16 or older to use Close Listening.</li>
        <li>You sign in with Google, Apple or an email link. Keep access to that account secure — anyone who can sign in to it can use yours.</li>
        <li>An account is for one person. Give a name that is not misleading and do not impersonate anyone.</li>
      </ul>
    </section>

    <section>
      <h2 class="mb-1 text-base font-bold">Podcasts belong to their creators</h2>
      <p>
        The shows and episodes in the app are made and owned by their creators. We play them from the
        creators' own published feeds and do not claim them as ours. Close Listening is not
        affiliated with or endorsed by the shows it lists unless it says so. If you make a show and
        want it changed or removed, write to <a :href="`mailto:${CONTACT}`">{{ CONTACT }}</a>.
      </p>
    </section>

    <section>
      <h2 class="mb-1 text-base font-bold">Transcripts, summaries and insights</h2>
      <p>
        Transcripts, summaries, topics, people, insights and trends are produced automatically and can
        be wrong — a misheard word, a misattributed quote, a topic that is not really there. Use them
        as a guide to the episode, not as a substitute for it, and not as financial, medical, legal or
        any other professional advice.
      </p>
    </section>

    <section>
      <h2 class="mb-1 text-base font-bold">What you create stays yours</h2>
      <p>
        Your notes, highlights, collections and follows are yours. You allow us to store and process
        them only to run the app for you — to keep them, show them back to you, and personalise what
        you see. We do not publish them; something leaves your account only when you share it.
      </p>
    </section>

    <section>
      <h2 class="mb-1 text-base font-bold">Connected agents</h2>
      <p>
        You can create an access token in Settings to let an AI assistant or another tool use your
        account. Whatever it does with that token is done on your behalf, and you are responsible for
        it. Revoke a token in Settings the moment you no longer want it used.
      </p>
    </section>

    <section>
      <h2 class="mb-1 text-base font-bold">Using the service fairly</h2>
      <ul class="list-disc space-y-1 pl-5">
        <li>Do not copy or download the service's content in bulk, or scrape it by automated means beyond what a connected agent does for you personally.</li>
        <li>Do not try to break, overload or get around the service's security or limits.</li>
        <li>Do not use the service for anything unlawful, or to harass or harm anyone.</li>
      </ul>
      <p class="mt-2">We may suspend or close an account that breaks these rules.</p>
    </section>

    <section>
      <h2 class="mb-1 text-base font-bold">Leaving</h2>
      <p>
        You can delete your account at any time from <em>Profile › Account › Delete account</em>; it
        takes effect immediately. We may end the service, or the beta, with notice in the app.
      </p>
    </section>

    <section>
      <h2 class="mb-1 text-base font-bold">No guarantees</h2>
      <p>
        The service is provided as it is, without warranties of any kind. To the extent the law allows,
        we are not liable for indirect or consequential losses, or for loss of data, arising from your
        use of it. Nothing in these terms limits rights you have that the law does not allow to be
        limited.
      </p>
    </section>

    <section>
      <h2 class="mb-1 text-base font-bold">Changes</h2>
      <p>
        If these terms change, we will update the date above and, for significant changes, tell you in
        the app before they apply. Continuing to use the app after that means you accept them.
      </p>
    </section>
  </article>
</template>
