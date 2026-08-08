"""Blind A/B judge: Claude Opus 4.8 scores Gemini-2.4 vs deepseek-iter2 on the 9 episodes.
Judge is told nothing about which system produced A or B (A/B randomized per episode by index
parity for reproducibility). Scores 3 dimensions 1-10 + picks winner. Aggregates win/loss/tie."""

import glob
import json
import os
import re
import sys
import urllib.request

AK = None
for line in open(os.path.join(os.getcwd(), ".env")):
    if line.startswith("AUTORESEARCH_JUDGE_ANTHROPIC_API_KEY="):
        AK = line.split("=", 1)[1].strip().strip('"').strip("'")
JUDGE = "claude-opus-4-8"

import os as _os

ROOT = _os.environ.get("PODCAST_EVAL_ROOT", ".test_outputs/manual")
BASE = f"{ROOT}/prod-v2.4-relabel-fixed/feeds"
# candidate dir name via argv[1] (default deepseek iter-2); argv[2] = label for output
CAND = sys.argv[1] if len(sys.argv) > 1 else "v2.5-deepseek-10ep-iter2"
CAND_LABEL = sys.argv[2] if len(sys.argv) > 2 else "deepseek"
IT2 = f"{ROOT}/{CAND}/feeds"
print(f"JUDGE: Gemini-2.4  vs  {CAND_LABEL}  ({CAND})")


def newest(root, feed, suffix):
    hits = sorted(
        glob.glob(f"{root}/{feed}/run_*/metadata/0001 - *{suffix}"),
        key=os.path.getmtime,
        reverse=True,
    )
    return hits[0] if hits else None


def newest_tx(root, feed):
    hits = sorted(
        glob.glob(f"{root}/{feed}/run_*/transcripts/0001 - *.adfree.txt"),
        key=os.path.getmtime,
        reverse=True,
    )
    return hits[0] if hits else None


def load(p):
    try:
        return json.loads(open(p).read())
    except Exception:
        return None


def extract(root, feed):
    m = load(newest(root, feed, ".metadata.json")) or {}
    g = load(newest(root, feed, ".gi.json")) or {}
    summ = m.get("summary")
    if isinstance(summ, str):
        try:
            summ = json.loads(summ)
        except Exception:
            pass
    if isinstance(summ, dict):
        bullets = summ.get("bullets") or []
        summ = summ.get("title", "") + "\n- " + "\n- ".join(bullets)
    # FAIR COMPARISON: the pipeline stores ALL insights ranked (ADR-135: never truncate; the viewer
    # shows top-N at view-time). Judge the production view = top-12 by rank, not the
    # full stored list.
    inodes = [n.get("properties", {}) for n in g.get("nodes", []) if n.get("type") == "Insight"]
    inodes.sort(key=lambda p: p.get("rank", 999) if isinstance(p.get("rank"), int) else 999)
    ins = [p.get("text") or p.get("label") for p in inodes[:12]]
    ins = [i for i in ins if i]
    tops = [
        n.get("properties", {}).get("label") for n in g.get("nodes", []) if n.get("type") == "Topic"
    ]
    tops = [t for t in tops if t]
    return {"summary": str(summ)[:2500], "insights": ins, "topics": tops}


PROMPT = """You are a strict editorial judge comparing two AI systems that each
summarized and extracted insights from the SAME podcast episode. You are NOT told which is which.

Below is the episode transcript (may be truncated), then System A's and System B's outputs.

CRITICAL LENGTH-CONTROL RULE: Do NOT reward length. A longer summary or a longer insight list is
NOT better. Judge substance-per-word. If two outputs convey the same substance, the SHORTER one is
BETTER. Heavily penalize padding, filler, hedging, repetition, and low-value additions. A system
that says more is only better if the extra material is genuinely significant and non-redundant.

Score each system 1-10 on THREE dimensions, judging against the transcript:
1. SUMMARY — faithful, well-organized, captures the most important points, no
fluff/hallucination. Length is NOT a virtue.
2. INSIGHTS — genuinely significant + accurate + non-redundant claims a reader would
want; penalize noise, repetition, trivia, padding heavily. A longer list is NOT better.
3. TOPICS — clean canonical subject headings (good: "monetary policy"; bad: sentence fragments).

Return ONLY JSON:
{"summary":{"A":n,"B":n,"winner":"A|B|tie","why":"<=25 words"},
 "insights":{"A":n,"B":n,"winner":"A|B|tie","why":"<=25 words"},
 "topics":{"A":n,"B":n,"winner":"A|B|tie","why":"<=25 words"},
 "overall_winner":"A|B|tie"}

=== TRANSCRIPT ===
@@TRANSCRIPT@@

=== SYSTEM A ===
SUMMARY:
@@A_SUMMARY@@
INSIGHTS (@@A_NINS@@): @@A_INSIGHTS@@
TOPICS: @@A_TOPICS@@

=== SYSTEM B ===
SUMMARY:
@@B_SUMMARY@@
INSIGHTS (@@B_NINS@@): @@B_INSIGHTS@@
TOPICS: @@B_TOPICS@@
"""


def call_judge(prompt):
    body = json.dumps(
        {"model": JUDGE, "max_tokens": 1500, "messages": [{"role": "user", "content": prompt}]}
    ).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": AK,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    txt = d["content"][0]["text"]
    mobj = re.search(r"\{.*\}", txt, re.S)
    return json.loads(mobj.group(0)) if mobj else None


feeds = sorted([os.path.basename(d) for d in glob.glob(IT2 + "/*") if os.path.isdir(d)])
agg = {
    "summary": {"gem": 0, "ds": 0, "tie": 0},
    "insights": {"gem": 0, "ds": 0, "tie": 0},
    "topics": {"gem": 0, "ds": 0, "tie": 0},
    "overall": {"gem": 0, "ds": 0, "tie": 0},
}
scoresum = {"summary": [0, 0], "insights": [0, 0], "topics": [0, 0]}  # [gem, ds]
rows = []
import time as _t

_NOW = _t.time()
for idx, feed in enumerate(feeds):
    _cg = newest(IT2, feed, ".gi.json")
    if not _cg:
        print(f"  {feed.split('.')[1][:16]:16} SKIP (candidate GI not regenerated)")
        continue
    gem = extract(BASE, feed)
    ds = extract(IT2, feed)
    txp = newest_tx(IT2, feed) or newest_tx(BASE, feed)
    tx = open(txp).read()[:45000] if txp else ""
    ds_is_A = idx % 2 == 1  # FLIPPED parity vs first run — controls for position bias
    A, B = (ds, gem) if ds_is_A else (gem, ds)
    repl = {
        "@@TRANSCRIPT@@": tx,
        "@@A_SUMMARY@@": A["summary"],
        "@@A_NINS@@": str(len(A["insights"])),
        "@@A_INSIGHTS@@": " | ".join(A["insights"][:40]),
        "@@A_TOPICS@@": ", ".join(A["topics"]),
        "@@B_SUMMARY@@": B["summary"],
        "@@B_NINS@@": str(len(B["insights"])),
        "@@B_INSIGHTS@@": " | ".join(B["insights"][:40]),
        "@@B_TOPICS@@": ", ".join(B["topics"]),
    }
    p = PROMPT
    for k, val in repl.items():
        p = p.replace(k, val)
    try:
        v = call_judge(p)
    except Exception as e:
        print(f"  {feed[:30]}: JUDGE ERROR {e}")
        continue
    if not v:
        print(f"  {feed[:30]}: no JSON")
        continue

    # un-blind
    def resolve(dim):
        w = v[dim]["winner"]
        a_score, b_score = v[dim]["A"], v[dim]["B"]
        gem_s, ds_s = (b_score, a_score) if ds_is_A else (a_score, b_score)
        scoresum[dim][0] += gem_s
        scoresum[dim][1] += ds_s
        if w == "tie":
            agg[dim]["tie"] += 1
            return "tie", gem_s, ds_s
        win_ds = (w == "A") == ds_is_A
        agg[dim]["ds" if win_ds else "gem"] += 1
        return ("ds" if win_ds else "gem"), gem_s, ds_s

    r = {d: resolve(d) for d in ("summary", "insights", "topics")}
    ow = v["overall_winner"]
    if ow == "tie":
        agg["overall"]["tie"] += 1
        owr = "tie"
    else:
        owr = "ds" if ((ow == "A") == ds_is_A) else "gem"
        agg["overall"][owr] += 1
    rows.append((feed.split(".")[1][:14], r, owr))
    print(
        f"  {feed.split('.')[1][:16]:16} sum:{r['summary'][0]:>3} ins:{r['insights'][0]:>3} top:{r['topics'][0]:>3} | overall:{owr}"  # noqa: E501
    )

print("\n=== AGGREGATE (ds=deepseek wins, gem=Gemini wins) over", len(rows), "episodes ===")
for dim in ("summary", "insights", "topics", "overall"):
    a = agg[dim]
    print(f"  {dim:9}: deepseek {a['ds']}  |  Gemini {a['gem']}  |  tie {a['tie']}")
print("\n=== mean scores (1-10) ===")
n = max(1, len(rows))
for dim in ("summary", "insights", "topics"):
    print(f"  {dim:9}: Gemini {scoresum[dim][0]/n:.2f}  vs  deepseek {scoresum[dim][1]/n:.2f}")
