/**
 * Stack-test error-recovery + alternate-profile coverage scaffolding.
 *
 * Issue #678 PR-D8 surfaced four nice-to-have gaps in the stack-test
 * suite:
 *
 *   1. Error-recovery paths — pipeline jobs that fail mid-run; the
 *      ``JOB_TERMINAL_BAD`` set in ``stack-jobs-flow.spec.ts`` is
 *      defined but never exercised.
 *   2. Alternate-profile (``cloud_thin``) coverage — only
 *      ``airgapped_thin`` is exercised today.
 *   3. Feed validation failures — malformed RSS URLs / duplicate feed
 *      URLs / network-unreachable hosts.
 *   4. Job cancellation flow — mid-run cancel + post-completion
 *      idempotency end-to-end through the api.
 *
 * Each is a full-stack scenario with real containers, real fixtures,
 * and ~3-5 min per test of wall-clock cost. Implementation is
 * deferred to follow-up PRs once Phase 1 of RFC-081 stabilises;
 * the skipped tests below are intentional placeholders so the
 * gaps stay visible in test discovery output.
 *
 * Notes for the implementor:
 *
 *  - For error-recovery, the simplest reliable failure injection is a
 *    feed URL that returns 500 from mock-feeds Nginx (extend
 *    ``compose/docker-compose.stack-test.yml`` mock-feeds with a
 *    ``location /failing-feed/`` block that returns 500).
 *  - For ``cloud_thin``, the existing ``stack-test-build-cloud`` make
 *    target builds the LLM image; the spec needs to seed
 *    ``viewer_operator.yaml`` with ``profile: cloud_thin`` and
 *    require the operator's OPENAI_API_KEY env. Public CI does not
 *    run cloud_thin (recurring API cost).
 *  - For feed validation, the api's ``/api/feeds`` PUT enforces
 *    ``feeds.spec.yaml`` schema; submit a malformed URL via the
 *    Sources dialog and assert the dialog surfaces the error.
 *  - For cancellation, the stack-jobs spec already polls
 *    ``GET /api/jobs/<id>``; a new spec can ``POST /api/jobs/<id>/cancel``
 *    mid-run and verify status transitions to ``cancelled``.
 */

import { test } from '@playwright/test'

test.describe('stack-test error-recovery + alternate-profile (#678 PR-D8 scaffolding)', () => {
  test.skip('pipeline job that fails mid-run reports exit_code != 0 and surfaces in /api/jobs', () => {
    // Future: seed feeds.spec.yaml with one URL pointing at
    // mock-feeds /failing-feed/ that 500s. Trigger pipeline. Poll
    // /api/jobs until terminal. Assert status=failed, exit_code != 0,
    // error_reason populated. Verify the api job-state webhook fired
    // (when PODCAST_JOB_WEBHOOK_URL is set).
  })

  test.skip('cloud_thin profile end-to-end smoke (real OpenAI Whisper + Gemini)', () => {
    // Future: seed viewer_operator.yaml with profile=cloud_thin.
    // Requires OPENAI_API_KEY + GEMINI_API_KEY in the test env.
    // Skipped on public CI (recurring cloud cost). Operator-only
    // gate via ``make stack-test-cloud-thin``.
  })

  test.skip('malformed RSS URL submitted via Sources dialog surfaces validation error', () => {
    // Future: open Sources → Feeds → enter ``not-a-url`` → click Add.
    // Assert the dialog renders the validation error inline; assert
    // GET /api/feeds still has the original list (no partial write).
  })

  test.skip('duplicate feed URL via Sources dialog is rejected with feedback', () => {
    // Future: add a feed URL that already exists in the seed. Assert
    // the dialog surfaces a "duplicate" message; assert /api/feeds
    // count is unchanged.
  })

  test.skip('mid-run cancel transitions a running job to cancelled and stops the subprocess', () => {
    // Future: trigger a pipeline job. While it's RUNNING (status poll
    // shows running), POST /api/jobs/<id>/cancel. Wait for next poll.
    // Assert status=cancelled, exit_code reflects SIGTERM, the
    // docker compose run pipeline-llm container is gone.
  })

  test.skip('cancel-after-success is idempotent (no transition; returns terminal record)', () => {
    // Future: same as above but cancel after status=succeeded.
    // Assert second cancel returns 200 with status=succeeded,
    // exit_code=0 (unchanged). Mirrors the unit-tier
    // test_jobs_cancel_after_succeeded_is_noop_terminal in #678 PR-C6.
  })
})
