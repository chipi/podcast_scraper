# Plan — LLM host/guest role on the voice-resolution call (ADR-135)

POC on Gemini, run on the v2.3.x pilot subset, iterate to stabilize. Backward-compatible: the
existing name-only contract and its tests stay green.

## Steps

1. **`_parse` (resolution.py)** — accept BOTH the legacy `{"voices": {"S0": "Name"}}` string form and
   the new `{"voices": {"S0": {"name": "...", "role": "host|guest|null"}}}` object form. Return
   `Dict[str, LLMVoice]` where `LLMVoice = {name: str|None, role: str|None}`. String form → role None.

2. **`build_resolution_prompt`** — add optional `episode_title`, `episode_description`, `intro_block`
   params. Render each as "(not provided)" when absent (backward-compatible shape). Switch the output
   spec to the `{name, role}` object; add the role rules + the "abstain on crowd voices" rule.

3. **`resolve_voices_and_roles(...)`** (new) — the combined resolver: build prompt, call `complete`,
   `_parse`, enforce name closed-list + `_refuted_by_third_person` (as today) + role vocabulary
   `{host,guest,null}`. Returns `Dict[str, LLMVoice]`.
   **`resolve_voices_from_conversation(...)`** — becomes a thin wrapper projecting to `{voice: name}`
   (unchanged signature + return → all existing tests pass).

4. **`_resolve_voices_via_llm` (pipeline.py)** — call `resolve_voices_and_roles`; return
   `(names: Dict[str,str], roles: Dict[str,str])`. New kwargs: `episode_title`, `episode_description`,
   `intro_block`.

5. **Single cleaning classifier (option D — the ordering fix).** `roster.classify_voices(diarization,
   ad_intervals, …) -> VoiceCleaning{ad, cameo, commercial, real}` is computed ONCE in
   `apply_diarization_to_result` right after diarization, and consumed by **both** the LLM call and
   the roster — one definition of "which voices are noise", never replicated.
   - `_labeled_intro_block(aligned, cleaning.real)` builds the first ~500 words, speaker-labeled,
     restricted to real voices (no inline cameo/ad logic — it consumes the classifier).
   - the LLM is asked only about `real_voice_texts` (real voices).
   - `resolve_speaker_roster(..., cleaning=cleaning, llm_voice_roles=...)` uses `cleaning.ad` for its
     ad set and `cleaning.commercial/cameo` for voice-typing instead of recomputing; standalone
     callers pass no `cleaning` and get the identical inline computation (behaviour-neutral).
   - new optional `episode_title`/`episode_description` kwargs on `apply_diarization_to_result`.

6. **`resolve_speaker_roster` (roster.py)** — new `llm_voice_roles: Optional[Dict[str,str]]` kwarg.
   Consume as:
   - **veto**: voices with LLM role `guest` join `conv_guests` in the step-2/3/4 exclusions
     (alongside the existing `stated_non_host_voices` guard);
   - **anchor**: on empty `host_pool`, voices with LLM role `host` may seat (a new host source,
     capped/guarded). Never override step-1 self-intro'd known hosts.
   Record provenance in the diagnostics (`source="llm_role"`).

7. **Thread title/description** from the main transcribe call site (episode_processor.py ~2438) only;
   relabel/failover paths pass None (degrade to intro + candidates, still correct).

8. **Tests** (airgapped, canned completion — no network):
   - `_parse` both forms; role vocabulary enforcement; abstain (null) no-op.
   - `resolve_voices_and_roles`: canned role JSON → veto/anchor semantics; a `host` for a
     `conv_guests`/`_refuted` voice is discarded.
   - roster: LLM `guest` blocks seat-fill; LLM `host` anchors an empty pool; LLM never unseats a
     self-intro'd known host.
   - existing `test_resolve_after_diarization.py` stays green (wrapper).

9. **Run** the pilot subset — Planet Money (anchor narrator-hosts), Latent Space (bind Eiso Kant +
   name hosts), Tariffs (`role=unknown`). Diff `.speakers.diagnostics.json` before/after. Iterate.

## Verification bar

- Deterministic path unchanged when the LLM abstains or is absent (airgapped).
- No category-4 crowd voice gets a forced name.
- ci-fast green at the end.
