# Naming decision trace — reconstruct the speaker-naming ladder for any episode

Operator goal (2026-10-04): every episode's sidecars hold enough to reconstruct how each voice got
its name and role — which rung proposed what, what won, what was overridden and why — so the
priority order can be measured instead of argued, patterns found across the corpus, and new
solutions designed from data.

Status: DESIGN. Nothing implemented yet.

## Why (what the inventory found)

The ladder has **31 rungs** from feed host detection to the published roster (inventory of
2026-10-04, file:line per rung kept in the session notes). Today:

- Only the **inputs** (`tried`: `known_hosts`, `metadata_named`, `detected_guests`, self-intros) and
  the **final** per-voice result (`voices[]`: name, role, `source`, `voice_type`) are recorded, plus
  `resolution_attribution` (rules-only vs final counts and the LLM's added/removed names).
- `source` is **lossy**: `self_intro` is the catch-all for every `voice_intro` entry that is not
  publisher or LLM — intro-reader names, co-host-formula names and merged-host-owner names included —
  and downstream code treats `self_intro` as "the voice said it" (metadata show-name filter, m0014,
  the copresence prior). `forced` does not say which forced variant fired; `feed` is unreachable.
- **Not recorded at all**: rejected candidates and reasons (placeholder / organisation /
  corroboration drops, LLM invented or refuted names, refused introductions, publish-gate
  rejections); losing names (an LLM name shadowed by a self-intro, pre-snap spellings,
  one-name-per-person losers); which host-seat step (1, 1b, 1c, 2-6) seated or refused a voice;
  role-only LLM verdicts; guest hosts added to the pool.
- `roster.py` has no logging at all; the corroboration rejection reason is only a log line.
- Migrations (m0012-m0020) rewrite metadata/segments/graphs but never the diagnostics sidecar, so
  the two drift.

Worked example (Conversations with Tyler, "Julia Ioffe on…", gold dev c087): the rules named the
2,735 s voice "Julia Ioffe" (guest, her own self-introduction — correct); the LLM then added
"Julia Ioffe" as HOST on Tyler's 380 s voice (the show states no host). The sidecar shows the
added name but not that it duplicated a self-introduced name, nor that corroboration had dropped
her from the guest list because "X on …" in a title is not an accepted cue.

## Design

### 1. A collector threaded through the roster (rungs 9-27, replayable)

`resolve_speaker_roster(..., trace: Optional[NamingTrace] = None)` — `None` is a no-op, so the
default behaviour and every existing caller are unchanged. Same pattern as the existing
`refused_out` out-parameter on `_intro_reader_voice_names`. Every decision site calls
`trace.voice(voice, rung, proposed=..., decision=..., reason=...)` or
`trace.name(name, rung, decision=..., reason=...)`.

Decision sites to instrument (the inventory's override points first): stated-seed (publisher
labels) replacing a self-intro; intro-reader refusals; show-mononym drop; LLM merge (and LLM names
skipped because the voice was already named); co-host formula; host-pool construction (with each
name's source); host seats steps 1/1b/1c/2/3/4/5/6 (seated AND refused, with the veto); host naming
(intro vs forced, and which veto blocked a forced name); guest pool harvest; guest naming
(intro/LLM/publisher vs forced / cameo-floor / dominant / host-elimination, and the third-person /
refused-intro guards); talkative-host re-run; stated-spelling snaps; recover-stated-names; publish
gate (rejected name + reason); one-name-per-person (groups, losers, role unification);
voice-type classification and edge-ad demotion.

Because the roster is deterministic and replayable, `roster_replay.py --trace` reconstructs this
part of the ladder for **every stored episode** offline, with no re-ingest.

### 2. Ingest-side records (rungs 1-7, 10, 28 — forward only)

Not replayable, so recorded at ingest into the same sidecar:
- detection: the raw `detect_speakers` answer; names dropped by the placeholder / organisation
  filters; corroboration rejects **with reason** (e.g. `no_interview_cue`); per host-pool name its
  sub-source (feed statement, author tag, episode description, config, recurrence, guest host).
- LLM resolution: the raw per-voice verdict (name + role), names refused by the closed list /
  third-person rule / one-per-voice rule, and whether the two-voice complement or swap fired.
- the rules-only (baseline) roster per voice, not just its counts, and which voices
  `_reconcile_non_regression` restored.

### 3. Sidecar shape

```json
"decision_trace": {
  "version": 1,
  "inputs": {
    "detection_raw": {"speakers": [], "hosts": [], "guests": []},
    "dropped": [{"name": "…", "stage": "placeholder|organisation|corroboration", "reason": "…"}],
    "host_pool": [{"name": "…", "sources": ["feed_statement", "author_tag"]}],
    "llm_raw": {"SPEAKER_00": {"name": "…", "role": "host"}},
    "llm_refused": [{"voice": "…", "name": "…", "reason": "not_in_list|third_person|duplicate"}],
    "baseline": {"SPEAKER_00": {"name": "…", "role": "…"}}
  },
  "voices": {
    "SPEAKER_00": [
      {"rung": "self_intro", "proposed": "Julia Ioffe", "decision": "accepted"},
      {"rung": "llm_merge", "proposed": "Julia Ioffe", "decision": "accepted", "reason": "…"}
    ]
  },
  "names": {
    "Julia Ioffe": [{"rung": "corroboration", "decision": "rejected", "reason": "no_interview_cue"}]
  },
  "post_hoc": [{"migration": "0019", "change": "removed name 'Pulitzer Prize-winning'"}]
}
```

`post_hoc`: migrations that rewrite a speaker surface append here, so the trace never silently
disagrees with the published roster.

## Phases

1. **Roster trace + `roster_replay --trace`.** Pure observability. Gate: replaying the gold sets
   and the full corpus with the trace on changes **0** published names and roles (byte-identical
   rosters); unit tests per decision site.
2. **Analysis on existing data**: per-rung precision against the gold labels (how often each rung's
   decision matches the label; how often it overrode an earlier rung that was right); failure
   chains across ~2,300 prod episodes, per show. This is what sets the priority order.
3. **Ingest-side records** (detection raw, drops with reasons, LLM raw/refused, baseline): forward
   only; ships with a deploy.
4. **Then** priority / ordering changes, each gated on dev + validation with the trace explaining
   every voice that moved. Candidates already visible: title/description guest phrases ("X on …",
   "— X", "with X", "In conversation with X") as corroboration; an LLM name may not be reused on a
   second voice once the episode bound it by self-introduction; `self_intro` split into
   self_intro / intro_reader / cohost_formula (a behaviour change — downstream treats it as "the
   voice said it"; gated separately).
