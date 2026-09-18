"""#2075 golden set — validate a speaker-naming change in SECONDS.

Why this exists: the DGX harness takes hours because it runs the whole pipeline per
episode — CLI startup, cleaning, GI, KG, summary — while a naming change touches only
name binding and record building. Everything else is overhead for the question asked.

So: load each episode's stored artifacts from a READ-ONLY corpus snapshot, call the
roster in-process, and assert what must hold. No CLI, no DGX, no enrichment, no LLM.
Runtime is seconds, so a change can be tried, measured and reverted inside one thought.

Each case names the defect it pins in the language of the TRANSCRIPT rather than of the
code, so every expectation is something a person can check by reading the episode.

    .venv/bin/python scripts/measure/2075_speaker_golden_set.py --snapshot /path/to/corpus
    ... --only dwarkesh          # one case
    ... --src /other/checkout/src  # compare two trees

``--snapshot`` defaults to ``$PODCAST_CORPUS_SNAPSHOT``. Point it at a corpus directory
containing ``run_*/transcripts/*.speakers.diagnostics.json`` — a restored production
snapshot, never a corpus the pipeline is writing to.

See docs/guides/VALIDATING_CORPUS_FIXES_FAST.md for how this fits the wider loop and how
to add cases.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

# --- the cases --------------------------------------------------------------------
# kind:
#   "absent"    — no voice may carry this name (it is wrong, or nobody states it)
#   "present"   — some voice must carry this name
#   "on"        — this exact voice must carry this name
#   "absent-on" — this exact voice must NOT carry this name
#   "record"    — the name must reach the episode's speaker record (placed or not)
#
# llm=True replays the model verdicts the diagnostics recorded. Use it ONLY to test
# whether a GUARD refuses a bad answer: injecting yesterday's answer defeats any fix that
# works by changing what the model is ASKED, which is how `Norman Conquest` came back
# from the dead the first time it was tried.
CASES = [
    # --- a name the episode never states must not be published (#2095) --------------
    dict(
        id="dwarkesh-authored-asr-name",
        ep="Grant Sanderson _ AI and the fut",
        kind="absent",
        name="Drance Anderson",
        why='host says "chatting with Drance Anderson" — the ASR mangling of Grant Sanderson',
    ),
    dict(
        id="dwarkesh-correct-guest",
        ep="Grant Sanderson _ AI and the fut",
        kind="record",
        name="Grant Sanderson",
        why="the episode states him in its title and description",
    ),
    # --- a stated guest must reach the record on a two-host show (#2078) -------------
    #
    # READ THIS BEFORE "FIXING" THE NEXT CASE. It fails, and this harness CANNOT make it pass.
    # Detection runs at INGEST: every stored diagnostics file already holds the CAPPED
    # `detected_guests`, and this harness replays those stored artifacts. So the case reads
    # identically before and after the fix. It is verified instead by
    #   tests/unit/podcast_scraper/speaker_detectors/test_stated_guest_survives_the_seat_cap.py
    #   tests/unit/podcast_scraper/providers/openai/test_stated_guest_survives_the_seat_cap.py
    #   tests/unit/podcast_scraper/workflow/stages/test_stated_and_arithmetic_lists_are_separate.py
    # and by a real `relabel_only` run, which re-runs detection.
    #
    # The two provider files are not duplication: the cap existed on BOTH the spaCy detector and
    # the OpenAI-compatible one, and production runs the latter (`prod_dgx_full` -> `vllm`).
    # Fixing only the first was reported as a complete fix by every offline check here.
    #
    # The case is kept because it states something true about the corpus, and because a harness
    # that silently drops what it cannot check is worse than one that fails honestly.
    dict(
        id="hardfork-guest-in-record",
        ep="A.I. School Is in Session",
        kind="record",
        name="Mackenzie Price",
        why='"let\'s bring in Alpha School cofounder, Mackenzie Price" -> "Thanks for having me"',
    ),
    dict(
        id="trip-stubb-in-record",
        ep="President Stubb",
        kind="record",
        name="Alexander Stubb",
        why="episode titled for him; the 66% voice is him describing his bilingual childhood",
    ),
    # --- must NOT regress: names the branch currently gets right ---------------------
    # KNOWN GAP, and a warning about this harness. These two FAIL today and that is real:
    # with no LLM the forced-host path puts the one known host's name on whichever voice is
    # dominant, and here the dominant voice is the GUEST (42 min of author vs 12 min of
    # host), so `Eric Topol` lands on Matthew Cobb. That reaches the no-LLM profiles
    # (airgapped, local, dev, reprocess_dgx_no_llm).
    #
    # Do NOT read these as "the swap fix is broken". The swap lives in
    # `resolve_voices_and_roles`, which this harness never calls: `llm=True` injects
    # POST-resolution names, downstream of the swap. Against the real recorded verdict the
    # swap does correct this episode — see
    # test_two_stated_names_on_each_other_s_voices_are_swapped_back.
    dict(
        id="ground-truths-host",
        ep="The Story of Francis Crick",
        kind="absent-on",
        voice="SPEAKER_01",
        name="Eric Topol",
        why="SPEAKER_01 is the 42-min author answering about Crick, not the host",
    ),
    dict(
        id="ground-truths-guest",
        ep="The Story of Francis Crick",
        kind="absent-on",
        voice="SPEAKER_02",
        name="Matthew Cobb",
        why='SPEAKER_02 is the host: "this is Matthew Cobb\'s seventh book"',
    ),
    dict(
        id="pravda-host-says-who-he-is",
        ep="Pavel Durov_s Russian biographer",
        kind="absent-on",
        voice="SPEAKER_00",
        name="Boris Goryachev",
        why='SPEAKER_00 says "your host, Kevin Rothrock" — it cannot be another person',
    ),
    dict(
        id="ilb-stated-spelling",
        ep="Ep182_ Andy Rachleff",
        kind="absent",
        name="Andy Ratcliffe",
        why="the episode states `Andy Rachleff`; Ratcliffe is the ASR's spelling of him",
    ),
    dict(
        id="rest-is-history-no-event",
        ep="The Troubles",
        kind="absent",
        name="Norman Conquest",
        why="a historical event is not a person",
    ),
    # --- must NOT regress: the seats the bleed rescue won ----------------------------
    dict(
        id="netflix-not-gergely",
        ep="Netflix_s Engineering Culture",
        kind="absent-on",
        voice="SPEAKER_00",
        name="Gergely Orosz",
        why="SPEAKER_00 is Elizabeth Stone; Gergely narrates and reads the outro",
    ),
    dict(
        id="journal-not-ryan",
        ep="Novo Nordisk_s CEO",
        kind="absent-on",
        voice="SPEAKER_01",
        name="Ryan Knutson",
        why="SPEAKER_01 is the Novo Nordisk CEO answering",
    ),
]


def load_episode(snapshot: Path, fragment: str):
    """Everything the roster needs for one episode, from the read-only snapshot."""
    for dp in snapshot.rglob("run_*/transcripts/*.speakers.diagnostics.json"):
        if fragment not in dp.name or ".adfree." in dp.name:
            continue
        stem = dp.name[: -len(".speakers.diagnostics.json")]
        sp = dp.parent / f"{stem}.segments.json"
        if not sp.is_file():
            continue
        try:
            segs = json.loads(sp.read_text(encoding="utf-8"))
            diag_all = json.loads(dp.read_text(encoding="utf-8")) or {}
        except (OSError, ValueError):
            continue
        tried = diag_all.get("tried") or {}
        if isinstance(segs, dict):
            segs = segs.get("segments") or []
        segs = [
            s for s in segs if isinstance(s, dict) and s.get("speaker") and s.get("end") is not None
        ]
        if not segs:
            continue
        # Replay the MODEL's answers instead of calling it. The diagnostics record, per
        # voice, the name that was accepted and the path it came from, so
        # `llm_resolution` entries reconstruct what the resolver was told —
        # deterministically, offline, in milliseconds.
        # LIMIT: these are the verdicts that SURVIVED the guards, not the raw model
        # output. This can prove a new guard refuses something previously published, but
        # it cannot exercise a proposal the old guards already threw away.
        llm_names: dict[str, str] = {}
        llm_roles: dict[str, str] = {}
        for v in diag_all.get("voices") or []:
            if isinstance(v, dict) and v.get("source") == "llm_resolution" and v.get("voice"):
                if v.get("resolved_name"):
                    llm_names[str(v["voice"])] = str(v["resolved_name"])
                if v.get("role"):
                    llm_roles[str(v["voice"])] = str(v["role"])
        feed: dict = {}
        ep: dict = {}
        for mp in (dp.parent.parent / "metadata").glob("*.metadata.json"):
            try:
                md = json.loads(mp.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            path = str((md.get("content") or {}).get("transcript_file_path") or "")
            if path.endswith(stem + ".txt"):
                feed, ep = md.get("feed") or {}, md.get("episode") or {}
                break
        return dict(
            stem=stem,
            segs=segs,
            tried=tried,
            feed=feed,
            episode=ep,
            llm_names=llm_names,
            llm_roles=llm_roles,
        )
    return None


def run_case(case, mods, snapshot: Path):
    ep = load_episode(snapshot, case["ep"])
    if ep is None:
        return "NO DATA", f"no episode matching {case['ep']!r} under {snapshot}"
    roster_mod, hosts_mod, base = mods
    # RECOMPUTE the hosts; do NOT read `tried.known_hosts`. This looks like the "verify the
    # instrument" trap and is the opposite of it, so it is written down.
    #
    # The stage this harness imitates is `relabel_only`, and that path recomputes:
    # `_relabel_existing_transcript` calls `_feed_hosts_from_sibling_metadata`
    # (episode_processor.py:2631), which reads the sibling metadata's `feed` block and runs
    # `detect_hosts_from_feed` over it. It never reads the stored diagnostics.
    #
    # `tried.known_hosts` is that computation's OUTPUT, recorded at the time of the original run —
    # so it carries whatever host detection believed back then. Feeding it back in pins the
    # harness to pre-fix behaviour and hides every improvement to host detection. Concretely: The
    # Rest Is History's stored value is `['Norman Conquest']`, a historical event that the
    # branch's `_NOT_A_MONONYM` / demonym guards now reject. Replaying the stored value republishes
    # it on two voices and reports a fixed defect as live.
    #
    # An earlier version of the full replay harness had the mirror-image bug — it reused stored
    # values where the pipeline recomputes — and that cost 89 episodes across 3 feeds.
    # Same principle, opposite direction: imitate the CALLER, not the record.
    known_hosts = list(
        hosts_mod.drop_non_person_names(
            sorted(
                hosts_mod.detect_hosts_from_feed(
                    ep["feed"].get("title"),
                    ep["feed"].get("description"),
                    ep["feed"].get("authors") or [],
                )
            ),
            ep["feed"].get("title"),
        )
    )
    voice_texts: dict[str, str] = collections.defaultdict(str)
    turns: list[tuple[str, str]] = []
    for s in ep["segs"]:
        voice_texts[s["speaker"]] += s.get("text") or ""
        if turns and turns[-1][0] == s["speaker"]:
            turns[-1] = (s["speaker"], turns[-1][1] + (s.get("text") or ""))
        else:
            turns.append((s["speaker"], s.get("text") or ""))
    diarization = base.DiarizationResult(
        segments=[
            base.DiarizationSegment(
                start=float(s["start"]), end=float(s["end"]), speaker=s["speaker"]
            )
            for s in ep["segs"]
        ],
        num_speakers=len(voice_texts),
    )
    episode_text = (
        " ".join(
            x
            for x in (
                ep["episode"].get("title") or "",
                ep["episode"].get("description") or "",
            )
            if x
        )
        or None
    )
    roster = roster_mod.resolve_speaker_roster(
        diarization,
        " ".join(t for _, t in turns),
        known_hosts=known_hosts,
        detected_guests=ep["tried"].get("detected_guests") or [],
        metadata_named=ep["tried"].get("metadata_named") or [],
        voice_texts=dict(voice_texts),
        ordered_turns=turns,
        episode_text=episode_text,
        llm_voice_names=(ep["llm_names"] or None) if case.get("llm") else None,
        llm_voice_roles=(ep["llm_roles"] or None) if case.get("llm") else None,
    )
    named = {v: r.name for v, r in roster.by_voice.items() if r.named}
    want = case["name"].lower()
    published = {n.lower() for n in named.values()}

    if case["kind"] == "absent":
        carriers = [v for v, n in named.items() if n.lower() == want]
        return ("PASS", "") if not carriers else ("FAIL", f"published on {carriers}")
    if case["kind"] == "present":
        return ("PASS", "") if want in published else ("FAIL", f"not published; named={named}")
    if case["kind"] == "on":
        got = named.get(case["voice"])
        ok = (got or "").lower() == want
        return ("PASS", "") if ok else ("FAIL", f"{case['voice']}={got!r}")
    if case["kind"] == "absent-on":
        got = named.get(case["voice"])
        ok = (got or "").lower() != want
        return ("PASS", "") if ok else ("FAIL", f"{case['voice']} carries it")
    if case["kind"] == "record":
        # The record carries people no voice was matched to, as `placed: false`. Reproduce the
        # pipeline's own definition rather than inventing one: `build_speaker_diagnostics`
        # computes `unbound_names` as `_clean_person_names(metadata_named)` minus the names the
        # roster bound, and `_unplaced_speakers` writes exactly that.
        #
        # This used to read `roster.diagnostics`, which does not exist — `SpeakerRoster` has only
        # `by_voice` and `num_speakers`. The lookup silently returned {} and the case then passed
        # whenever the name appeared in the STORED metadata, making it an assertion about the
        # snapshot rather than about the code under test. Caught in review; the guide's own
        # "verify the instrument" rule, missed inside the instrument.
        unbound = {
            str(n).lower()
            for n in roster_mod._clean_person_names(ep["tried"].get("metadata_named") or ())
            if str(n).lower() not in published
        }
        if want in published or want in unbound:
            return ("PASS", "")
        stated = sorted(
            {
                str(n)
                for n in (ep["tried"].get("metadata_named") or [])
                + (ep["tried"].get("detected_guests") or [])
            }
        )
        return (
            "FAIL",
            f"neither bound to a voice nor unplaced in the record (stated={stated[:4]})",
        )
    return ("SKIP", f"unknown kind {case['kind']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--snapshot",
        default=os.environ.get("PODCAST_CORPUS_SNAPSHOT", ""),
        help="read-only corpus snapshot root (env: PODCAST_CORPUS_SNAPSHOT)",
    )
    ap.add_argument(
        "--src",
        default=str(Path(__file__).resolve().parents[2] / "src"),
        help="checkout to import the roster from (compare two trees)",
    )
    ap.add_argument("--only", default="", help="substring filter over case ids")
    args = ap.parse_args()

    if not args.snapshot:
        print(
            "no snapshot given: pass --snapshot or set PODCAST_CORPUS_SNAPSHOT to a "
            "corpus directory holding run_*/transcripts/*.speakers.diagnostics.json",
            file=sys.stderr,
        )
        return 2
    snapshot = Path(args.snapshot)
    if not snapshot.is_dir():
        print(f"snapshot is not a directory: {snapshot}", file=sys.stderr)
        return 2

    sys.path.insert(0, args.src)
    from podcast_scraper.providers.ml.diarization import base, roster as roster_mod
    from podcast_scraper.speaker_detectors import hosts as hosts_mod

    if not roster_mod.__file__.startswith(args.src):
        print(
            f"--src {args.src} was shadowed; roster imported from {roster_mod.__file__}",
            file=sys.stderr,
        )
        return 2

    mods = (roster_mod, hosts_mod, base)
    cases = [c for c in CASES if not args.only or args.only in c["id"]]
    if not cases:
        print(f"no case id matches {args.only!r}", file=sys.stderr)
        return 2
    width = max(len(c["id"]) for c in cases)
    marks = {"PASS": "ok  ", "FAIL": "FAIL", "NO DATA": "n/a ", "SKIP": "skip"}
    failed = 0
    for case in cases:
        verdict, detail = run_case(case, mods, snapshot)
        if verdict == "FAIL":
            failed += 1
        print(f"  {marks[verdict]} {case['id']:{width}s}  {case['name']:20s} {detail}")
        if verdict == "FAIL":
            print(f"       why it matters: {case['why']}")
    print(f"\n{failed} failing of {len(cases)} cases")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
