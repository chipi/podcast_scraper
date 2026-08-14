"""Pin the REAL contract of ``timeout_context`` — it observes, it does not enforce.

Issue #379 introduced this context manager "to prevent hangs". It prevents none: the
wrapped block runs to completion and ``TimeoutError`` is raised only after control returns
from the ``yield``. On 2026-08-12 a production run hung for 4h15m while nominally wrapped
in a 1200s ``timeout_context``.

These tests exist so nobody re-reads the name and assumes protection. If someone later
makes it genuinely interrupting, ``test_does_not_interrupt_a_slow_block`` will fail — and
that failure is the signal to delete these tests, not to weaken them.
"""

import logging
import os
import sys
import time
import unittest

PACKAGE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from podcast_scraper.utils import timeout as timeout_mod


class TestTimeoutContextObservesButDoesNotEnforce(unittest.TestCase):
    def test_does_not_interrupt_a_slow_block(self):
        """The block runs to completion despite blowing the deadline.

        This is the documented — and dangerous — behaviour. It is asserted deliberately.
        """
        completed = []
        with self.assertRaises(timeout_mod.TimeoutError):
            with timeout_mod.timeout_context(1, "slow-op"):
                time.sleep(1.4)
                completed.append("block finished anyway")

        self.assertEqual(
            completed,
            ["block finished anyway"],
            "the block must have run to completion — this manager cannot interrupt",
        )

    def test_raises_only_after_the_block_returns(self):
        """Ordering proof: the exception cannot arrive mid-operation."""
        events = []
        try:
            with timeout_mod.timeout_context(1, "ordering"):
                time.sleep(1.3)
                events.append("inside-end")
        except timeout_mod.TimeoutError:
            events.append("raised")

        self.assertEqual(events, ["inside-end", "raised"])

    def test_fast_block_does_not_raise(self):
        with timeout_mod.timeout_context(5, "fast-op"):
            time.sleep(0.05)

    def test_none_disables_observation(self):
        with timeout_mod.timeout_context(None, "no-deadline"):
            time.sleep(0.05)

    def test_zero_disables_observation(self):
        with timeout_mod.timeout_context(0, "no-deadline"):
            time.sleep(0.05)

    def test_deadline_breach_logs_at_error_level(self):
        """The deadline log is the only in-flight signal a stalled run produces.

        It was raised from WARNING to ERROR because during the 2026-08-12 wedge the
        pipeline emitted zero output for four hours, and alerting needs something to key
        on while the operation is still stuck.
        """
        with self.assertLogs(timeout_mod.logger, level=logging.ERROR) as captured:
            try:
                with timeout_mod.timeout_context(1, "stalled-op"):
                    time.sleep(1.3)
            except timeout_mod.TimeoutError:
                pass

        joined = "\n".join(captured.output)
        self.assertIn("DEADLINE EXCEEDED", joined)
        self.assertIn("stalled-op", joined)
        self.assertIn("STILL RUNNING", joined)

    def test_deadline_log_fires_while_the_block_is_still_running(self):
        """The signal must arrive during the stall, not after it clears.

        A log line that only appears once the operation finishes is worthless for
        detecting an operation that never finishes.
        """
        seen_during = []

        class _Probe(logging.Handler):
            def emit(self, record):
                seen_during.append(record.getMessage())

        probe = _Probe(level=logging.ERROR)
        timeout_mod.logger.addHandler(probe)
        try:
            with self.assertRaises(timeout_mod.TimeoutError):
                with timeout_mod.timeout_context(1, "long-op"):
                    time.sleep(1.4)
                    # By now the timer thread must already have logged.
                    self.assertTrue(
                        any("DEADLINE EXCEEDED" in m for m in seen_during),
                        "deadline log did not fire while the block was still executing",
                    )
        finally:
            timeout_mod.logger.removeHandler(probe)

    def test_timer_thread_is_daemon(self):
        """A pending deadline timer must never hold the interpreter open at exit."""
        import threading

        before = {t.name for t in threading.enumerate()}
        with timeout_mod.timeout_context(30, "daemon-check"):
            live = [t for t in threading.enumerate() if t.name not in before]
            for thread in live:
                self.assertTrue(
                    thread.daemon,
                    f"timer thread {thread.name!r} must be a daemon so it cannot delay shutdown",
                )


if __name__ == "__main__":
    unittest.main()
