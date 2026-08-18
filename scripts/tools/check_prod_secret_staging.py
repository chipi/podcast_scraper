#!/usr/bin/env python3
"""Every prod workflow that CREATES a container must stage the tmpfs secrets first.

Why this gate exists
--------------------
ADR-115 (#1250) moved prod's credentials out of ``.env`` into ``/dev/shm/podcast-secrets/`` —
RAM, never disk. Good security, but it split one fact into two:

    "prod is deployed"   is no longer the same as   "prod has credentials"

The directory does not persist. Docker copies secrets into a container at CREATE time, so
containers made during a deploy keep working — the API stays up, everything looks healthy. But
anything creating a NEW container later finds nothing there and runs with no credentials at
all: no Deepgram, no DeepSeek, no LiteLLM.

That requirement was recorded in one commit message (2317839e, 2026-08-10, "host dir not
persistent") and a comment in a single workflow. Nothing enforced it, so every workflow written
afterwards rediscovered it in production:

    2026-07-21  reprocess-prod        predates ADR-115, never migrated — broken for a month
    2026-08-10  recreate-operator-api broke on first use, patched the same day
    2026-08-18  gi-repair-prod        written without it
    2026-08-18  inspect-prod-corpus   written without it

On 2026-08-18 that cost an evening: a gateway 401 was read as a bad API key, a live production
key was deleted and re-minted to "fix" it, and three deploys ran before anyone checked whether
the key was on the box at all. It was not — nothing was.

A comment cannot enforce an invariant. This can.

What counts
-----------
CREATES A CONTAINER : ``docker compose ... run`` / ``docker run`` inside a workflow that also
                      SSHes to the prod box. ``compose up`` is covered too — deploy.sh does its
                      own check, but the workflow still has to have delivered the files.
STAGES SECRETS      : uses ``./.github/actions/stage-prod-secrets`` (preferred), or performs the
                      equivalent inline (``/dev/shm/podcast-secrets.staged``) for the two
                      workflows that predate the shared action.

The real fix is to make the directory persist so no workflow needs to know any of this. Until
then, this gate is what keeps the list above from growing.

Usage::

    python scripts/tools/check_prod_secret_staging.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

# Talks to the prod box at all.
TOUCHES_PROD = re.compile(r"PROD_SSH_PRIVATE_KEY|deploy@\$\{\{\s*steps\.ts_host", re.I)

# Creates a container. Matches the VERB, not "docker compose ... run" on one line: every real
# invocation here is split over continuation lines inside an ssh string, so a same-line pattern
# saw none of them and the gate reported OK after checking a single file — the same false-green
# it exists to prevent. Caught only because "1 prod workflow" was implausible.
CREATES_CONTAINER = re.compile(
    r"\brun\s+--rm\b|\brun\s+-T\b|\bcompose\s+run\b|\bdocker\s+run\b|\bup\s+-d\b", re.I
)

# Delivers the tmpfs secrets — the shared action, or the inline equivalent.
STAGES_SECRETS = re.compile(r"stage-prod-secrets|podcast-secrets\.staged", re.I)

# Workflows that reach prod but deliberately never create a container (pure file/API work).
# Listing one here is a claim that it cannot need credentials — keep it short and justified.
EXEMPT = {
    "backup-corpus-prod.yml",  # tar over ssh; no container, no LLM call
    "tailscale-acl.yml",  # ACL apply; never touches the box
    # Runs on the prod BOX but in a different compose project (-p vps-observability, under
    # /opt/vps-observability) that has no podcast services and reads none of these secrets.
    # Its container is the Alloy o11y agent.
    "deploy-vps-observability-endpoints.yml",
}


def _strip_comments(text: str) -> str:
    """Drop whole-line YAML comments before matching.

    Without this the gate reads prose: drill-deploy.yml deploys to the DR DRILL host, never
    prod, but a comment saying "same pattern as PROD_SSH_PRIVATE_KEY" made it look like a prod
    workflow. A gate whose false positives have to be silenced by an exemption list decays into
    the exemption list, so this is fixed at the match instead.
    """
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))


def main() -> int:
    problems: list[str] = []
    checked = 0

    for wf in sorted(WORKFLOWS.glob("*.yml")):
        if wf.name in EXEMPT:
            continue
        text = _strip_comments(wf.read_text(encoding="utf-8"))
        if not TOUCHES_PROD.search(text):
            continue
        if not CREATES_CONTAINER.search(text):
            continue
        checked += 1
        if not STAGES_SECRETS.search(text):
            problems.append(
                f"{wf.relative_to(REPO_ROOT)} creates a container on prod but never stages the "
                f"tmpfs secrets.\n"
                f"    The container will start with NO provider credentials — the run fails on a "
                f"401, or silently produces empty artifacts.\n"
                f"    Fix: add this step before the container is created, after the ts_host step:\n"
                f"        - name: Stage prod tmpfs secrets (REQUIRED before creating a container)\n"
                f"          if: vars.PODCAST_SECRETS_VIA_FILES == '1'\n"
                f"          uses: ./.github/actions/stage-prod-secrets\n"
                f"          with: {{ ssh_target: ..., ssh_identity: ..., <the 11 keys> }}\n"
                f"    See .github/actions/stage-prod-secrets/action.yml for the full input list."
            )

    if problems:
        print("PROD SECRET STAGING: FAIL\n")
        for p in problems:
            print(f"  - {p}\n")
        print(
            "A container created on prod without staged secrets has no credentials at all.\n"
            "This has now happened four times; the gate exists so it stops happening."
        )
        return 1

    print(
        f"PROD SECRET STAGING: OK — {checked} prod workflow(s) create containers, "
        f"all stage the tmpfs secrets first."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
