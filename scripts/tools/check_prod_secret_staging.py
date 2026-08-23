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
from typing import Any

import yaml

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

# MOUNTS them into the container. Delivery is only HALF the job: compose reads the files
# through docker-compose.secrets.yml, and docker/secrets-shim.sh (the entrypoint) exports them.
# A workflow that stages but never joins the overlay ships a container with the files on the
# HOST and nothing in the container — gi-repair-prod did exactly that on 2026-08-18 and died on
# "Deepgram API key required", one layer past the 401 it had just stopped producing. Checking
# only for staging let that pass, so the gate checks both halves now.
MOUNTS_SECRETS = re.compile(r"docker-compose\.secrets\.yml", re.I)

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


def _step_text(step: dict[str, Any]) -> str:
    """The matchable text of one workflow step: its run script + action ref + name."""
    parts: list[str] = []
    for key in ("run", "uses", "name"):
        val = step.get(key)
        if isinstance(val, str):
            parts.append(val)
    return "\n".join(parts)


def _proximity_problems(wf_name: str, doc: Any) -> list[str]:
    """Every container-creating STEP must be preceded by a (re)stage since the last create.

    Existence alone is not enough — that is the D5 false-green. On 2026-08-23 deploy-prod staged
    the tmpfs secrets early, ran ``compose up`` (create #1), then ran the D5 gateway probe (create
    #2) MANY steps later without re-staging. The whole-file gate saw a stage AND a create and
    passed — but ``/dev/shm/podcast-secrets`` had been reaped by systemd-logind ``RemoveIPC`` when
    the earlier ssh session ended, so create #2 got a 401 that looked like a bad key. The fix was
    a re-stage step immediately before the probe (deploy-prod.yml:672).

    Model: a container-creating step ENDS its ssh session, which reaps the RAM dir for every
    later step. So after any create step the staging is stale; the next create needs its own
    stage. Walk each job's steps in order — ``staged`` goes True on a stage step, and a create
    step with ``staged`` False is the D5 defect. ``staged`` resets to False AFTER a create step
    (multiple creates in ONE step share that step's session, so only cross-step creates trip it).
    Conservative: re-staging is idempotent + cheap, so a spurious "re-stage needed" is safe; a
    missed one costs an evening.
    """
    problems: list[str] = []
    jobs = (doc or {}).get("jobs") if isinstance(doc, dict) else None
    if not isinstance(jobs, dict):
        return problems
    for job_name, job in jobs.items():
        steps = (job or {}).get("steps") if isinstance(job, dict) else None
        if not isinstance(steps, list):
            continue
        staged = False
        prior_create: str | None = None
        for step in steps:
            if not isinstance(step, dict):
                continue
            # Strip shell/YAML comment lines: several deploy steps carry explanatory comments like
            # "# nested docker compose run pipeline-llm" inside their run script, which would match
            # CREATES_CONTAINER and flag a pure .env-staging step as a container creation.
            text = _strip_comments(_step_text(step))
            if STAGES_SECRETS.search(text):
                staged = True
            if CREATES_CONTAINER.search(text):
                label = str(step.get("name") or step.get("uses") or "<unnamed step>")
                if not staged:
                    where = (
                        f"after '{prior_create}'"
                        if prior_create
                        else "and no stage step precedes it"
                    )
                    problems.append(
                        f"{wf_name} (job '{job_name}'): step '{label}' creates a container "
                        f"{where} without a (re)stage in between.\n"
                        f"    The container-creating step before it ended its ssh session, so "
                        f"systemd RemoveIPC reaped /dev/shm/podcast-secrets — this container "
                        f"starts with NO credentials (the D5 401, 2026-08-23).\n"
                        f"    Fix: add a `uses: ./.github/actions/stage-prod-secrets` step "
                        f"immediately before this one (see deploy-prod.yml:672)."
                    )
                staged = False  # this step's session ends -> tmpfs reaped for later steps
                prior_create = label
    return problems


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
        if not MOUNTS_SECRETS.search(text):
            problems.append(
                f"{wf.relative_to(REPO_ROOT)} creates a container on prod but never joins "
                f"compose/docker-compose.secrets.yml.\n"
                f"    The files land on the HOST and nothing reaches the container — the run "
                f"dies on a missing provider key.\n"
                f"    Fix: add a conditional overlay to the compose invocation:\n"
                f"        SEC=''; if [ -d /dev/shm/podcast-secrets ] && "
                f'[ -n "$(ls -A /dev/shm/podcast-secrets 2>/dev/null)" ]; then '
                f"SEC='-f compose/docker-compose.secrets.yml'; fi\n"
                f"    ...then pass $SEC to `docker compose`, as deploy.sh:38 does."
            )
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

        # Proximity (P3): existence isn't enough — a re-stage must precede EACH cross-step
        # container creation (the D5 false-green). Parse the steps and walk them in order.
        try:
            doc = yaml.safe_load(wf.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            rel = wf.relative_to(REPO_ROOT)
            problems.append(f"{rel} could not be parsed for the proximity check: {exc}")
        else:
            problems.extend(_proximity_problems(str(wf.relative_to(REPO_ROOT)), doc))

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
        f"all stage the tmpfs secrets AND mount them via the secrets overlay."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
