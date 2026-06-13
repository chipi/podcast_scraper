# vLLM relocation — exit the provisioning business

**Status**: PLAN — awaiting go.
**Trigger**: 2026-06-12, operator moved vllm-autoresearch out of `podcast_scraper`
into the new `agentic-ai-homelab` public repo, checked out on the DGX at
`~/agentic-ai-homelab/`. Going forward, all DGX vllm changes are committed
to that repo and pulled on the DGX (gitops, single source of truth for
homelab compose stacks).

**Goal**: this repo (`podcast_scraper`) stops shipping the vllm compose,
deploy hook, verify hook, and operator docs. The runtime contract stays
identical (vllm serves on `http://<dgx-tailnet-host>:8003/`); autoresearch
eval scripts and `Config` are unchanged.

---

## Blast radius (19 hits across 7 files)

### Code (5 files)

1. **`infra/dgx/converge/deploy.py`** (lines 455–598, ~144 lines)
   - Whole `# #928 — vllm-autoresearch` block:
     - `VLLM_INSTALL_ROOT`, `VLLM_COMPOSE_FILE`, `VLLM_IMAGE`, `VLLM_PORT`,
       `VLLM_MODEL`, `VLLM_GPU_MEM_UTIL`, `VLLM_MAX_MODEL_LEN`, `VLLM_MAX_NUM_SEQS`
     - `files.directory(... /opt/vllm-autoresearch ...)`
     - `VLLM_COMPOSE_CONTENT` heredoc
     - 3 × `server.shell(...)` calls — write compose, pull image, `compose up -d`
   - **Action**: delete the entire block. Last block in deploy.py — no
     downstream code references the `VLLM_*` constants.

2. **`infra/dgx/converge/verify.py`** (lines 169–207)
   - 3 × `server.shell(...)` assertions — container up, API responsive,
     model-matches-compose. The last one reads
     `/opt/vllm-autoresearch/docker-compose.yml`.
   - **Action**: delete all three. The verify probe is moving to the
     homelab repo (along with the compose); a single optional probe could
     stay if we still want `podcast_scraper`'s converge step to fail loudly
     when vllm isn't reachable — see the **decision** at the bottom of this
     doc.

3. **`infra/dgx/vllm-autoresearch/README.md`** (177 lines)
   - Operator-facing handoff doc for the OLD `/opt/vllm-autoresearch/`
     install. Talks about NVIDIA tag history, Mamba cache constraints, the
     model-swap procedure, all the #928 Cell C debugging.
   - **Action**: one of:
     - (a) Delete the whole directory and let
       `agentic-ai-homelab/vllm-autoresearch/README.md` carry the content
       (preferred — single source of truth).
     - (b) Replace with a one-line redirect:
       `vllm-autoresearch lives in https://github.com/chipi/agentic-ai-homelab/tree/main/vllm-autoresearch/`.
     - Lean (a). Stale docs are worse than a missing dir.

### Eval reports (2 files, references only — they're historical)

4. **`docs/guides/eval-reports/EVAL_HYBRID_ROUTING_2026_06.md`** (1 line, 251)
   - Line 251: `infra/dgx/vllm-autoresearch/` (#928 prereq deploy)`
   - **Action**: append a parenthetical: `(moved to agentic-ai-homelab on 2026-06-12)`.
     Don't rewrite history — eval reports are point-in-time and the original
     reference was correct at the time.

5. **`docs/guides/eval-reports/EVAL_SUMMARY_DGX_LOCAL_2026_06.md`** (3 lines: 16, 113, 439, 446)
   - Multiple historical references to `infra/dgx/vllm-autoresearch/`.
   - **Action**: same parenthetical strategy (or leave as-is — these are
     dated reports and the original path was real at the time). Lean
     leave-as-is unless the reader actually clicks the path expecting to
     find it.

### WIP/planning docs (2 files)

6. **`docs/wip/AUTORESEARCH_LEARNINGS_FOR_V3.md`** (line 465)
   - References `infra/dgx/vllm-autoresearch/README.md` as a source.
   - **Action**: update the reference to the new repo URL.

7. **`docs/wip/NEXT_SESSION_PLAN.md`** (line 80)
   - Bullet referencing the old README.
   - **Action**: update to the new repo URL OR delete if the bullet is
     no longer needed.

### Untouched (intentionally — they're path-agnostic)

- `src/podcast_scraper/eval/autoresearch/openai_backend.py`,
  `autoresearch_track_a.py`, and friends — they hit the running endpoint
  at port 8003, never reference the filesystem path.
- `src/podcast_scraper/providers/ml/model_registry.py` — no vllm install
  path references; the registry's vllm endpoint is `http://...:8003/v1`
  (host-template only).
- All profile YAMLs under `config/profiles/` — no `/opt/vllm-autoresearch`
  references.
- Tests under `tests/` — no install-path references; the autoresearch
  backend tests use mocked endpoints.

### NOT in the search results but worth double-checking before edits

- `Makefile` — any `dgx-deploy` / `dgx-verify` targets that gate on vllm
  being up. (Survey didn't hit any; verify before the actual edits.)
- `infra/dgx/converge/README.md` (if it exists) — could mention vllm.
- `AGENTS.md` / `CLAUDE.md` / `CONTRIBUTING.md` — could mention vllm-autoresearch
  in DGX operator guidance.

---

## Key decision: keep a vllm reachability probe in `verify.py`?

The full verify block today does 3 things:
1. Container is up (docker ps grep)
2. API responsive on :8003
3. Served model matches compose declaration

(1) and (3) are homelab-repo concerns now — they belong in
`agentic-ai-homelab`'s own verify script. (2) is a **podcast_scraper
runtime concern**: if vllm isn't reachable on :8003, autoresearch evals
fail. Two options:

- **Option A — full deletion**: drop all 3 probes. If vllm is down,
  autoresearch sweeps explode at runtime, not at converge time. Cleaner
  separation; aligns with "podcast_scraper exits provisioning".
- **Option B — keep the reachability ping**: keep only the
  `curl :8003/health` line. Cheap, useful, doesn't read any homelab-owned
  files. Doesn't violate the separation (we're a client, not a provisioner).

Lean **Option A** if autoresearch sweeps run rarely (then converge-time
verify of vllm is wrong-place-wrong-time). Lean **Option B** if we want
`make dgx-verify` to be a single "is everything we depend on up?" probe.

**Recommendation**: Option B. The cost is one curl line; the upside is
that `make dgx-verify` stays a complete operator-side health check.

---

## Order of execution (when authorized)

1. Confirm new repo URL is `https://github.com/chipi/agentic-ai-homelab/`
   (or whatever the operator's actual handle/repo path is).
2. Choose Option A or B for verify.py.
3. Edit `deploy.py` — delete lines 455–598 (the whole vllm block).
4. Edit `verify.py` — delete lines 169–177 + 190–207 (Option B keeps
   179–188); update the index comment numbering on subsequent blocks if
   any are renumbered downstream of vllm.
5. Delete `infra/dgx/vllm-autoresearch/` directory entirely.
6. Update `docs/wip/AUTORESEARCH_LEARNINGS_FOR_V3.md` and
   `docs/wip/NEXT_SESSION_PLAN.md` to point at the new repo URL.
7. Leave the two eval reports alone (or add a brief
   "moved 2026-06-12" note — operator's call).
8. `make ci-fast` — must stay green (no test touches the vllm block).
9. Commit + push + PR — only when explicitly authorized.

## What this does NOT do

- It does NOT migrate any of the actual vllm runtime config. The
  operator has already done that in `agentic-ai-homelab` and on the
  DGX. This is purely a follow-up cleanup of the now-orphaned plumbing
  in `podcast_scraper`.
- It does NOT change the runtime contract — port 8003, OpenAI-compatible
  API. Autoresearch scripts continue to work unchanged.
- It does NOT touch the `whisper-openai`, `speaches`, or `pyannote`
  services. They stay in `podcast_scraper` (for now — same pattern could
  apply later if they also move).

## Open questions for the operator

1. Confirm new repo URL: `https://github.com/chipi/agentic-ai-homelab/` ?
2. Option A or B on `verify.py` reachability ping?
3. Eval reports: leave as-is, or add `(moved 2026-06-12)` parentheticals?
4. Do you want the `infra/dgx/vllm-autoresearch/` dir DELETED, or replaced
   with a 1-line redirect README?
