#!/usr/bin/env python3
"""Generate Deepgram nova-3 silver reference (transcript + diarization RTTM) for prod episodes."""

import os
import sys
import time

import requests

KEY = os.environ["DEEPGRAM_API_KEY"]
AUDIO_DIR = ".test_outputs/diar-prod90/audio"
OUT = ".test_outputs/moss-eval/prod/deepgram_ref"
os.makedirs(OUT, exist_ok=True)

URL = (
    "https://api.deepgram.com/v1/listen"
    "?model=nova-3&smart_format=true&diarize=true&utterances=true&punctuate=true"
)

episodes = sys.argv[1:] or ["e001", "e011", "e041", "e051", "e071", "e081"]
for ep in episodes:
    path = f"{AUDIO_DIR}/{ep}.mp3"
    t0 = time.time()
    with open(path, "rb") as fh:
        r = requests.post(
            URL,
            headers={"Authorization": f"Token {KEY}", "Content-Type": "audio/mpeg"},
            data=fh,
            timeout=600,
        )
    r.raise_for_status()
    j = r.json()
    alt = j["results"]["channels"][0]["alternatives"][0]
    transcript = alt.get("transcript", "")
    with open(f"{OUT}/{ep}.txt", "w") as f:
        f.write(transcript + "\n")
    # diarization RTTM from utterances (speaker turns)
    utts = j["results"].get("utterances", [])
    lines = []
    for u in utts:
        start, end, spk = float(u["start"]), float(u["end"]), u.get("speaker", 0)
        lines.append(f"SPEAKER {ep} 1 {start:.3f} {end-start:.3f} <NA> <NA> spk{spk} <NA> <NA>")
    with open(f"{OUT}/{ep}.rttm", "w") as f:
        f.write("\n".join(lines) + "\n")
    spk_ct = len({u.get("speaker", 0) for u in utts})
    print(
        f"{ep}: {len(utts)} utts, {spk_ct} speakers, "
        f"{len(transcript.split())} words ({time.time()-t0:.1f}s)",
        flush=True,
    )
print("DONE")
